#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(pwd)"
OUTDIR="$BASE_DIR/analysis"
mkdir -p "$OUTDIR"

echo "Analyzing replicas in: $BASE_DIR"
echo "Output directory: $OUTDIR"

for d in rep_{01..20}; do
    echo "=== $d ==="

    if [[ ! -d "$d" ]]; then
        echo "Missing $d, skipping"
        continue
    fi

    if [[ ! -f "$d/eq_start.xtc" || ! -f "$d/eq_start.tpr" ]]; then
        echo "$d missing eq_start.xtc or eq_start.tpr, skipping"
        continue
    fi

    # Check completion
    if grep -q "Finished mdrun" "$d/eq_start.log" 2>/dev/null; then
        echo "$d finished normally"
    else
        echo "WARNING: $d may not have finished normally"
    fi

    # Protein/backbone RMSD
    # first selection = fitting group, second selection = RMSD group
    echo "Backbone Backbone" | gmx_mpi rms \
        -s "$d/eq_start.tpr" \
        -f "$d/eq_start.xtc" \
        -n "$d/index.ndx" \
        -o "$d/rmsd_backbone.xvg" \
        -tu ns >/dev/null 2>&1 || echo "RMSD failed for $d"

    # Radius of gyration
    echo "Protein" | gmx_mpi gyrate \
        -s "$d/eq_start.tpr" \
        -f "$d/eq_start.xtc" \
        -n "$d/index.ndx" \
        -o "$d/rg_protein.xvg" >/dev/null 2>&1 || echo "Rg failed for $d"

    # Anchor group RMSD: checks stability of r_75 group
    echo "r_75 r_75" | gmx_mpi rms \
        -s "$d/eq_start.tpr" \
        -f "$d/eq_start.xtc" \
        -n "$d/index.ndx" \
        -o "$d/rmsd_anchor_r75.xvg" \
        -tu ns >/dev/null 2>&1 || echo "Anchor RMSD failed for $d"

    # Pull group RMSD: optional check of pulled residue local stability
    echo "r_171 r_171" | gmx_mpi rms \
        -s "$d/eq_start.tpr" \
        -f "$d/eq_start.xtc" \
        -n "$d/index.ndx" \
        -o "$d/rmsd_pull_r171.xvg" \
        -tu ns >/dev/null 2>&1 || echo "Pull-group RMSD failed for $d"

    # Energies
    # This extracts Temperature, Potential, Total Energy if names match.
    printf "Temperature\nPotential\nTotal Energy\n0\n" | gmx_mpi energy \
        -f "$d/eq_start.edr" \
        -o "$d/energy_T_Pot_Total.xvg" >/dev/null 2>&1 || echo "Energy extraction failed for $d"

done

cat > "$OUTDIR/plot_and_summarize.py" << 'PYEOF'
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.getcwd()
reps = [f"rep_{i:02d}" for i in range(1, 21)]
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

def read_xvg(path):
    xs = []
    ys = []
    if not os.path.exists(path):
        return None
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

def summarize_one(rep):
    row = {"rep": rep}

    files = {
        "pullx": "eq_pullx.xvg",
        "pullf": "eq_pullf.xvg",
        "rmsd_backbone": "rmsd_backbone.xvg",
        "rg": "rg_protein.xvg",
        "rmsd_anchor": "rmsd_anchor_r75.xvg",
        "rmsd_pull": "rmsd_pull_r171.xvg",
    }

    for key, fname in files.items():
        path = os.path.join(BASE, rep, fname)
        data = read_xvg(path)
        if data is None:
            row[f"{key}_mean"] = np.nan
            row[f"{key}_std"] = np.nan
            row[f"{key}_first"] = np.nan
            row[f"{key}_last"] = np.nan
            continue

        t, y = data
        y1 = y[:, 0]
        row[f"{key}_mean"] = np.mean(y1)
        row[f"{key}_std"] = np.std(y1)
        row[f"{key}_first"] = y1[0]
        row[f"{key}_last"] = y1[-1]

    return row

summary = pd.DataFrame([summarize_one(rep) for rep in reps])
summary.to_csv(os.path.join(OUT, "eq_ensemble_summary.csv"), index=False)

print(summary)

def overlay_plot(fname, title, ylabel, outfile):
    plt.figure(figsize=(8, 5))
    for rep in reps:
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

overlay_plot("eq_pullx.xvg", "Pull coordinate during v=0 equilibration", "x / nm", "pullx_overlay.png")
overlay_plot("eq_pullf.xvg", "Pull force during v=0 equilibration", "Force", "pullf_overlay.png")
overlay_plot("rmsd_backbone.xvg", "Backbone RMSD", "RMSD / nm", "rmsd_backbone_overlay.png")
overlay_plot("rg_protein.xvg", "Protein radius of gyration", "Rg / nm", "rg_overlay.png")
overlay_plot("rmsd_anchor_r75.xvg", "Anchor group RMSD: r_75", "RMSD / nm", "rmsd_anchor_overlay.png")
overlay_plot("rmsd_pull_r171.xvg", "Pull group RMSD: r_171", "RMSD / nm", "rmsd_pull_group_overlay.png")

# Histograms of final values across replicas
hist_items = [
    ("pullx_last", "Final pull coordinate across replicas", "x_final / nm", "hist_final_pullx.png"),
    ("pullf_mean", "Mean pull force across replicas", "mean force", "hist_mean_pullf.png"),
    ("rmsd_backbone_last", "Final backbone RMSD across replicas", "RMSD / nm", "hist_final_rmsd.png"),
    ("rg_last", "Final radius of gyration across replicas", "Rg / nm", "hist_final_rg.png"),
    ("rmsd_anchor_last", "Final anchor RMSD across replicas", "Anchor RMSD / nm", "hist_final_anchor_rmsd.png"),
]

for col, title, xlabel, outfile in hist_items:
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
PYEOF

python "$OUTDIR/plot_and_summarize.py"

echo ""
echo "Done."
echo "Check:"
echo "  analysis/eq_ensemble_summary.csv"
echo "  analysis/pullx_overlay.png"
echo "  analysis/pullf_overlay.png"
echo "  analysis/rmsd_backbone_overlay.png"
echo "  analysis/rg_overlay.png"
echo "  analysis/rmsd_anchor_overlay.png"