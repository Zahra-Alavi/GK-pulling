#!/usr/bin/env python3
"""
plot_and_summarize_eq.py
Reads per-replica GROMACS output files produced by analyze_eq_ensemble.sh,
builds a summary CSV, and writes overlay + histogram plots to analysis/.

Usage:
    python plot_and_summarize_eq.py <BASE_DIR>
    BASE_DIR should be the root directory that contains rep_01 … rep_20.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# First argument is the base directory containing rep_01 … rep_20
BASE = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

REPS = [f"rep_{i:02d}" for i in range(1, 21)]

OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# XVG reader
# ---------------------------------------------------------------------------

def read_xvg(path):
    """
    Parse a GROMACS .xvg file and return (time_array, data_array).
    Lines starting with '#' or '@' are skipped (GROMACS comments/legends).
    Returns None if the file is missing or has no numeric data.
    """
    if not os.path.exists(path):
        return None

    xs, ys = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("@") or not line.strip():
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            if len(vals) >= 2:
                xs.append(vals[0])
                ys.append(vals[1:])

    if not xs:
        return None
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Per-replica summary
# ---------------------------------------------------------------------------

# Map of short key -> xvg filename for each metric we want to summarize
XVG_FILES = {
    "pullx":         "eq_pullx.xvg",       # pull coordinate (v=0 equilibration)
    "pullf":         "eq_pullf.xvg",       # pull force
    "rmsd_backbone": "rmsd_backbone.xvg",  # backbone RMSD vs starting structure
    "rg":            "rg_protein.xvg",     # radius of gyration (whole protein)
    "rmsd_anchor":   "rmsd_anchor_r75.xvg",  # anchor residue r_75 RMSD
    "rmsd_pull":     "rmsd_pull_r171.xvg",   # pulled residue r_171 RMSD
}


def summarize_one(rep):
    """
    For a single replica directory, compute mean/std/first/last for each
    metric and return them as a flat dict (one row of the summary table).
    NaN is stored when a file is missing or unparseable.
    """
    row = {"rep": rep}

    for key, fname in XVG_FILES.items():
        path = os.path.join(BASE, rep, fname)
        data = read_xvg(path)

        if data is None:
            # File missing or empty — fill with NaN so the CSV stays rectangular
            row[f"{key}_mean"]  = np.nan
            row[f"{key}_std"]   = np.nan
            row[f"{key}_first"] = np.nan
            row[f"{key}_last"]  = np.nan
            continue

        t, y = data
        y1 = y[:, 0]  # first data column (time is index 0 in the xvg, already split out)
        row[f"{key}_mean"]  = np.mean(y1)
        row[f"{key}_std"]   = np.std(y1)
        row[f"{key}_first"] = y1[0]
        row[f"{key}_last"]  = y1[-1]

    return row


# Build summary table and write CSV
summary = pd.DataFrame([summarize_one(rep) for rep in REPS])
csv_path = os.path.join(OUT, "eq_ensemble_summary.csv")
summary.to_csv(csv_path, index=False)
print(summary.to_string())


# ---------------------------------------------------------------------------
# Overlay line plots (one line per replica)
# ---------------------------------------------------------------------------

# Each entry: (xvg filename, plot title, y-axis label, output PNG filename)
OVERLAY_PLOTS = [
    ("eq_pullx.xvg",         "Pull coordinate during v=0 equilibration", "x / nm",    "pullx_overlay.png"),
    ("eq_pullf.xvg",         "Pull force during v=0 equilibration",      "Force",      "pullf_overlay.png"),
    ("rmsd_backbone.xvg",    "Backbone RMSD",                            "RMSD / nm",  "rmsd_backbone_overlay.png"),
    ("rg_protein.xvg",       "Protein radius of gyration",               "Rg / nm",    "rg_overlay.png"),
    ("rmsd_anchor_r75.xvg",  "Anchor group RMSD: r_75",                  "RMSD / nm",  "rmsd_anchor_overlay.png"),
    ("rmsd_pull_r171.xvg",   "Pull group RMSD: r_171",                   "RMSD / nm",  "rmsd_pull_group_overlay.png"),
]


def overlay_plot(fname, title, ylabel, outfile):
    """Draw all replicas on a single axes and save to analysis/."""
    plt.figure(figsize=(8, 5))
    for rep in REPS:
        path = os.path.join(BASE, rep, fname)
        data = read_xvg(path)
        if data is None:
            continue
        t, y = data
        plt.plot(t, y[:, 0], linewidth=1, alpha=0.75, label=rep)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, outfile), dpi=200)
    plt.close()


for args in OVERLAY_PLOTS:
    overlay_plot(*args)


# ---------------------------------------------------------------------------
# Histograms of final / mean values across replicas
# ---------------------------------------------------------------------------

# Each entry: (summary column, plot title, x-axis label, output PNG filename)
HIST_PLOTS = [
    ("pullx_last",         "Final pull coordinate across replicas",    "x_final / nm",       "hist_final_pullx.png"),
    ("pullf_mean",         "Mean pull force across replicas",          "mean force",          "hist_mean_pullf.png"),
    ("rmsd_backbone_last", "Final backbone RMSD across replicas",      "RMSD / nm",           "hist_final_rmsd.png"),
    ("rg_last",            "Final radius of gyration across replicas", "Rg / nm",             "hist_final_rg.png"),
    ("rmsd_anchor_last",   "Final anchor RMSD across replicas",        "Anchor RMSD / nm",    "hist_final_anchor_rmsd.png"),
]

for col, title, xlabel, outfile in HIST_PLOTS:
    if col not in summary:
        continue
    vals = summary[col].dropna().values
    if len(vals) == 0:
        continue
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=10)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, outfile), dpi=200)
    plt.close()


print(f"\nSaved summary and plots in: {OUT}")
