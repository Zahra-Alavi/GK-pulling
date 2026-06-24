#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- user settings ----------
T = 310.0                      # K
kB = 0.008314462618            # kJ/mol/K
beta = 1.0 / (kB * T)

v = 0.0005                     # nm/ps
x_ref0 = 4.05746               # nm
MAX_TIME_PS = 3000.0           # first 3 ns of each pull trajectory

pullf_name = "pullf.xvg"
pullx_name = "pullx.xvg"

# If work comes out negative for opening/stretching, change this to -1.0
SIGN = 1.0

OUTDIR = "jarzynski_analysis"
os.makedirs(OUTDIR, exist_ok=True)

def read_xvg(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("@") or not line.strip():
                continue
            data.append([float(x) for x in line.split()])
    return np.array(data)

def cumulative_trapezoid(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out

rows = []
all_work_profiles = []

for rep in sorted(glob.glob("rep_??")):
    pullf_path = os.path.join(rep, pullf_name)
    pullx_path = os.path.join(rep, pullx_name)

    if not os.path.exists(pullf_path):
        print(f"Missing {pullf_path}, skipping")
        continue

    fdata = read_xvg(pullf_path)

    if fdata.shape[1] < 2:
        print(f"Bad pullf file: {pullf_path}")
        continue

    t_all = fdata[:, 0]       # ps
    F_all = fdata[:, 1]       # kJ/mol/nm, GROMACS pull force

    # Analyze only the first MAX_TIME_PS from the start of this trajectory.
    # This uses relative time so it also works if the XVG time column does not
    # start exactly at zero.
    t_start = t_all[0]
    time_mask = t_all <= t_start + MAX_TIME_PS
    if np.count_nonzero(time_mask) < 2:
        print(f"Not enough pullf points in first {MAX_TIME_PS:g} ps for {rep}, skipping")
        continue

    t = t_all[time_mask]
    F = F_all[time_mask]

    # Protocol work:
    # W(t) = integral F(t) dx_ref = integral F(t) v dt
    W_t = SIGN * cumulative_trapezoid(F * v, t)
    W_final = W_t[-1]

    x_first = np.nan
    x_last = np.nan
    x_mean = np.nan

    if os.path.exists(pullx_path):
        xdata = read_xvg(pullx_path)
        xmask = xdata[:, 0] <= xdata[0, 0] + MAX_TIME_PS
        x = xdata[xmask, 1]
        x_first = x[0]
        x_last = x[-1]
        x_mean = np.mean(x)

    rows.append({
        "rep": rep,
        "W_kJmol": W_final,
        "time_final_ps": t[-1],
        "force_mean_kJmol_nm": np.mean(F),
        "force_std_kJmol_nm": np.std(F),
        "force_min_kJmol_nm": np.min(F),
        "force_max_kJmol_nm": np.max(F),
        "x_first_nm": x_first,
        "x_last_nm": x_last,
        "x_mean_nm": x_mean,
    })

    all_work_profiles.append((rep, t, W_t))

df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No replicas found. Check pullf.xvg file names and directory structure.")

works = df["W_kJmol"].values

# Jarzynski estimator
exp_terms = np.exp(-beta * works)
DeltaF = -(1.0 / beta) * np.log(np.mean(exp_terms))

W_mean = np.mean(works)
W_std = np.std(works, ddof=1)
W_diss_mean = W_mean - DeltaF

df["DeltaF_Jarzynski_kJmol"] = DeltaF
df["W_diss_i_kJmol"] = df["W_kJmol"] - DeltaF

summary = {
    "N_replicas": len(works),
    "Temperature_K": T,
    "beta_mol_per_kJ": beta,
    "max_analysis_time_ps": MAX_TIME_PS,
    "pulling_rate_nm_per_ps": v,
    "work_sign_used": SIGN,
    "mean_work_kJmol": W_mean,
    "std_work_kJmol": W_std,
    "min_work_kJmol": np.min(works),
    "max_work_kJmol": np.max(works),
    "DeltaF_Jarzynski_kJmol": DeltaF,
    "mean_W_diss_kJmol": W_diss_mean,
    "mean_exp_minus_betaW": np.mean(exp_terms),
}

df.to_csv(os.path.join(OUTDIR, "work_by_replica.csv"), index=False)

with open(os.path.join(OUTDIR, "jarzynski_summary.txt"), "w") as f:
    for k, val in summary.items():
        f.write(f"{k}: {val}\n")

print("\nJarzynski summary")
print("-----------------")
for k, val in summary.items():
    print(f"{k}: {val}")

print("\nPer-replica work")
print("----------------")
print(df[["rep", "W_kJmol", "W_diss_i_kJmol", "x_first_nm", "x_last_nm"]])

# Plot work histogram
plt.figure(figsize=(6, 4))
plt.hist(works, bins=10)
plt.xlabel("Work W / kJ mol$^{-1}$")
plt.ylabel("Count")
plt.title("Work distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "work_histogram.png"), dpi=200)
plt.close()

# Plot cumulative work profiles
plt.figure(figsize=(7, 5))
for rep, t, W_t in all_work_profiles:
    plt.plot(t, W_t, linewidth=1, alpha=0.8, label=rep)
plt.xlabel("Time / ps")
plt.ylabel("Cumulative work W(t) / kJ mol$^{-1}$")
plt.title("Cumulative work profiles")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "cumulative_work_profiles.png"), dpi=200)
plt.close()

# Plot per-replica final work
plt.figure(figsize=(8, 4))
plt.plot(df["rep"], df["W_kJmol"], marker="o")
plt.axhline(DeltaF, linestyle="--", label="Jarzynski ΔF")
plt.xticks(rotation=45)
plt.ylabel("Work / kJ mol$^{-1}$")
plt.title("Final work by replica")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "work_by_replica.png"), dpi=200)
plt.close()

# Plot dissipated work by replica
plt.figure(figsize=(8, 4))
plt.plot(df["rep"], df["W_diss_i_kJmol"], marker="o")
plt.axhline(W_diss_mean, linestyle="--", label="Mean dissipated work")
plt.xticks(rotation=45)
plt.ylabel("W - ΔF / kJ mol$^{-1}$")
plt.title("Dissipated work by replica")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "dissipated_work_by_replica.png"), dpi=200)
plt.close()

print(f"\nSaved outputs in: {OUTDIR}/")
