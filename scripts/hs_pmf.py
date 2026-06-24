#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- user settings ----------------
T = 310.0
kB = 0.008314462618      # kJ/mol/K
beta = 1.0 / (kB * T)

k_pull = 1000.0          # kJ/mol/nm^2
v = 0.0005               # nm/ps
x_ref0 = 4.05746         # nm

pullx_name = "pullx.xvg"
pullf_name = "pullf.xvg"

# Use 3000 ps if you want early opening before unfolding.
# Use None for full trajectory.
TMAX_PS = 3000.0

xmin = 3.9
xmax = 7.2               
dx = 0.02

SIGN = 1.0               # change to -1 if your work sign is wrong

OUTDIR = "hummer_szabo_analysis"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- helper functions ----------------
def read_xvg(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("@") or not line.strip():
                continue
            data.append([float(x) for x in line.split()])
    return np.array(data)

def cumulative_trapz(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out

def logsumexp(vals):
    vals = np.array(vals)
    if len(vals) == 0:
        return np.nan
    m = np.max(vals)
    return m + np.log(np.sum(np.exp(vals - m)))

# ---------------- load trajectories ----------------
traj = []
work_final_rows = []

for rep in sorted(glob.glob("rep_??")):
    px = os.path.join(rep, pullx_name)
    pf = os.path.join(rep, pullf_name)

    if not os.path.exists(px) or not os.path.exists(pf):
        print(f"Skipping {rep}: missing pullx or pullf")
        continue

    xdata = read_xvg(px)
    fdata = read_xvg(pf)

    n = min(len(xdata), len(fdata))
    t = xdata[:n, 0]
    x = xdata[:n, 1]
    F = fdata[:n, 1]

    if TMAX_PS is not None:
        mask = t <= t[0] + TMAX_PS
        t = t[mask]
        x = x[mask]
        F = F[mask]
        if len(t) < 2:
            print(f"Skipping {rep}: not enough points in first {TMAX_PS:g} ps")
            continue

    xref = x_ref0 + v * t

    # Protocol work: W(t) = integral F d x_ref = integral F v dt
    W = SIGN * cumulative_trapz(F * v, t)

    Vbias = 0.5 * k_pull * (x - xref) ** 2

    traj.append({
        "rep": rep,
        "t": t,
        "x": x,
        "F": F,
        "xref": xref,
        "W": W,
        "Vbias": Vbias,
    })

    work_final_rows.append({
        "rep": rep,
        "W_final_kJmol": W[-1],
        "time_final_ps": t[-1],
        "x_first_nm": x[0],
        "x_last_nm": x[-1],
        "xref_first_nm": xref[0],
        "xref_last_nm": xref[-1],
    })

if len(traj) == 0:
    raise RuntimeError("No trajectories found.")

work_df = pd.DataFrame(work_final_rows)
work_df.to_csv(os.path.join(OUTDIR, "hs_input_work_summary.csv"), index=False)

print("\nLoaded trajectories:")
print(work_df)

# ---------------- Hummer-Szabo reconstruction ----------------
bins = np.arange(xmin, xmax + dx, dx)
centers = 0.5 * (bins[:-1] + bins[1:])

# We use a time-slice-normalized Hummer-Szabo estimator:
# each snapshot contributes approximately exp[-beta W(t) + beta Vbias(x,t)]
# and each time slice is normalized by the Jarzynski factor at that time.
#
# This gives a practical PMF reconstruction from nonequilibrium pulls.

log_weights_by_bin = [[] for _ in centers]

# Need common time indexing. Use shortest trajectory length.
min_len = min(len(tr["t"]) for tr in traj)

for j in range(min_len):
    Wj = np.array([tr["W"][j] for tr in traj])

    # Jarzynski free energy at this time:
    # F(t) = -kBT ln < exp(-beta W(t)) >
    log_jarz_factor = logsumexp(-beta * Wj) - np.log(len(Wj))
    F_t = -(1.0 / beta) * log_jarz_factor

    for tr in traj:
        x = tr["x"][j]
        W = tr["W"][j]
        Vbias = tr["Vbias"][j]

        b = np.searchsorted(bins, x) - 1
        if 0 <= b < len(centers):
            # normalized nonequilibrium unbiasing weight
            logw = -beta * (W - F_t) + beta * Vbias
            log_weights_by_bin[b].append(logw)

pmf = np.full(len(centers), np.nan)
counts = np.zeros(len(centers), dtype=int)

for b, logs in enumerate(log_weights_by_bin):
    counts[b] = len(logs)
    if len(logs) > 0:
        pmf[b] = -(1.0 / beta) * logsumexp(logs)

# shift minimum to zero
pmf -= np.nanmin(pmf)

out = pd.DataFrame({
    "x_nm": centers,
    "PMF_kJmol": pmf,
    "count": counts,
})
out.to_csv(os.path.join(OUTDIR, "hs_pmf.csv"), index=False)

# ---------------- plots ----------------
plt.figure(figsize=(7, 5))
plt.plot(centers, pmf, marker="o", markersize=3, linewidth=1)
plt.xlabel("Pull coordinate x / nm")
plt.ylabel("PMF / kJ mol$^{-1}$")
plt.title("Hummer-Szabo PMF")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "hs_pmf.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 4))
plt.bar(centers, counts, width=dx)
plt.xlabel("Pull coordinate x / nm")
plt.ylabel("Number of snapshots")
plt.title("Sampling along pull coordinate")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "hs_sampling_counts.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 5))
for tr in traj:
    plt.plot(tr["t"], tr["x"], linewidth=1, alpha=0.8)
plt.xlabel("Time / ps")
plt.ylabel("Actual pull coordinate x / nm")
plt.title("pullx trajectories")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pullx_trajectories.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 5))
for tr in traj:
    plt.plot(tr["t"], tr["W"], linewidth=1, alpha=0.8)
plt.xlabel("Time / ps")
plt.ylabel("Cumulative work / kJ mol$^{-1}$")
plt.title("Cumulative work trajectories")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "cumulative_work.png"), dpi=200)
plt.close()

print(f"\nSaved Hummer-Szabo outputs in: {OUTDIR}/")
print("Main files:")
print("  hs_pmf.csv")
print("  hs_pmf.png")
print("  hs_sampling_counts.png")
print("  pullx_trajectories.png")
print("  cumulative_work.png")
