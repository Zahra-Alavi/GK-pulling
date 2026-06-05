#!/usr/bin/env bash
# analyze_eq_ensemble.sh
# Runs GROMACS analysis tools (rms, gyrate, energy) across all rep_XX directories,
# then calls plot_and_summarize_eq.py to build overlay plots and a summary CSV.
#
# Run from the directory that contains rep_01 … rep_20.
#
# Inputs  (per rep_XX/): eq_start.{xtc,tpr,edr,log}, index.ndx
# Outputs (per rep_XX/): rmsd_backbone.xvg, rg_protein.xvg,
#                        rmsd_anchor_r75.xvg, rmsd_pull_r171.xvg,
#                        energy_T_Pot_Total.xvg
# Outputs (analysis/):   eq_ensemble_summary.csv, *_overlay.png, hist_*.png
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

echo "Running plot_and_summarize_eq.py ..."
python "$SCRIPT_DIR/plot_and_summarize_eq.py" "$BASE_DIR"

echo ""
echo "Done."
echo "Check:"
echo "  analysis/eq_ensemble_summary.csv"
echo "  analysis/pullx_overlay.png"
echo "  analysis/pullf_overlay.png"
echo "  analysis/rmsd_backbone_overlay.png"
echo "  analysis/rg_overlay.png"
echo "  analysis/rmsd_anchor_overlay.png"