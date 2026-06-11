#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# setup_eq_start_dmso.sh
#
# Run biased starting equilibration for DMSO replicas.
#
# Each rep_XX/ should already contain:
#   eq_dmso_5ns.gro
#   eq_dmso_5ns.cpt
#   topol.top
#   posre.itp
#   posre_anchor_CA.itp
#   amber99sb-ildn.ff
#   eq_start_k1000.mdp
#
# This script:
#   1. creates a fresh index.ndx for each DMSO replica
#   2. adds these groups:
#        anchor_73_83_CA = alpha carbons of residues 73-83
#        r_75            = all atoms of residue 75
#        r_171           = all atoms of residue 171
#   3. runs grompp for eq_start
#   4. runs mdrun for eq_start on 4 GPUs, 4 replicas at a time
#
# Outputs per replica:
#   index.ndx
#   eq_start.tpr
#   eq_start.gro
#   eq_start.cpt
#   eq_start.edr
#   eq_start.log
#   eq_pullf.xvg
#   eq_pullx.xvg
#   eq_start.runlog
###############################################################################

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GMX="gmx_mpi"

NGPU=4
NTOMP=16

INPUT_GRO="eq_dmso_5ns.gro"
INPUT_CPT="eq_dmso_5ns.cpt"
MDP="eq_start_k1000.mdp"

OUTPUT_DEFFNM="eq_start"
OUTPUT_TPR="eq_start.tpr"

SKIP_IF_DONE="yes"


create_index() {
  local REP="$1"
  local REP_NAME
  REP_NAME="$(basename "$REP")"

  echo "=== $REP_NAME: creating fresh DMSO-compatible index.ndx ==="

  (
    cd "$REP"

    [[ -f "$INPUT_GRO" ]] || { echo "ERROR: missing $INPUT_GRO in $REP"; exit 1; }

    # First create a default GROMACS index for the DMSO-containing system.
    # This gives correct System, Protein, Water, Ion, etc. groups for the new atom count.
    printf "q\n" | "$GMX" make_ndx \
      -f "$INPUT_GRO" \
      -o index.ndx > make_ndx.log 2>&1

    # Then append the custom pulling/restraint groups by parsing the .gro file.
    python3 - << 'PY'
from pathlib import Path
import textwrap

gro_file = "eq_dmso_5ns.gro"
index_file = "index.ndx"

anchor = []
r75 = []
r171 = []

with open(gro_file) as f:
    lines = f.readlines()[2:-1]

for line in lines:
    try:
        resid = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnr = int(line[15:20])
    except ValueError:
        continue

    # Restrict to protein-like residues by excluding solvent/ions.
    # This prevents accidental residue-number overlap with solvent.
    if resname in {"SOL", "HOH", "WAT", "DMSO", "NA", "CL", "K", "MG", "CA"}:
        continue

    if 73 <= resid <= 83 and atomname == "CA":
        anchor.append(atomnr)

    if resid == 75:
        r75.append(atomnr)

    if resid == 171:
        r171.append(atomnr)

def format_group(name, atoms):
    if not atoms:
        raise SystemExit(f"ERROR: group {name} is empty. Check residue numbering in {gro_file}.")
    out = [f"\n[ {name} ]\n"]
    for i in range(0, len(atoms), 15):
        out.append(" ".join(f"{a:6d}" for a in atoms[i:i+15]) + "\n")
    return "".join(out)

with open(index_file, "a") as out:
    out.write(format_group("anchor_73_83_CA", anchor))
    out.write(format_group("r_75", r75))
    out.write(format_group("r_171", r171))

print(f"anchor_73_83_CA atoms: {len(anchor)}")
print(f"r_75 atoms: {len(r75)}")
print(f"r_171 atoms: {len(r171)}")
PY

    echo "--- custom groups added to index.ndx ---"
    grep -n "\[ anchor_73_83_CA \]\|\[ r_75 \]\|\[ r_171 \]" index.ndx
  )
}


run_rep() {
  local REP="$1"
  local GPU_ID="$2"
  local REP_NAME
  REP_NAME="$(basename "$REP")"

  echo "=== $REP_NAME: grompp eq_start ==="

  (
    cd "$REP"

    [[ -f "$INPUT_GRO" ]] || { echo "ERROR: missing $INPUT_GRO"; exit 1; }
    [[ -f "$INPUT_CPT" ]] || { echo "ERROR: missing $INPUT_CPT"; exit 1; }
    [[ -f "$MDP" ]] || { echo "ERROR: missing $MDP"; exit 1; }
    [[ -f "topol.top" ]] || { echo "ERROR: missing topol.top"; exit 1; }
    [[ -f "index.ndx" ]] || { echo "ERROR: missing index.ndx"; exit 1; }

    "$GMX" grompp \
      -f "$MDP" \
      -c "$INPUT_GRO" \
      -t "$INPUT_CPT" \
      -p topol.top \
      -r "$INPUT_GRO" \
      -n index.ndx \
      -o "$OUTPUT_TPR" \
      -maxwarn 1
  )

  echo "=== $REP_NAME: mdrun eq_start on GPU $GPU_ID ==="

  (
    cd "$REP"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    mpirun -np 1 "$GMX" mdrun \
      -s "$OUTPUT_TPR" \
      -deffnm "$OUTPUT_DEFFNM" \
      -pf eq_pullf.xvg \
      -px eq_pullx.xvg \
      -nb gpu \
      -ntomp "$NTOMP"
  ) > "$REP/eq_start.runlog" 2>&1 &
}


echo "Base directory: $BASE_DIR"
echo "Looking for replicas: $BASE_DIR/rep_01 ... $BASE_DIR/rep_20"

# ---------------------------------------------------------------------------
# Step 1: create fresh index.ndx in each DMSO replica
# ---------------------------------------------------------------------------

for REP in "$BASE_DIR"/rep_{01..20}; do
  [[ -d "$REP" ]] || continue

  if [[ "$SKIP_IF_DONE" == "yes" && -f "$REP/index.ndx" ]]; then
    echo "=== $(basename "$REP"): index.ndx exists; recreating anyway for safety ==="
    rm -f "$REP/index.ndx"
  fi

  create_index "$REP"
done


# ---------------------------------------------------------------------------
# Step 2: run eq_start in parallel batches of 4 GPUs
# ---------------------------------------------------------------------------

gpu_slot=0
batch_pids=()

for REP in "$BASE_DIR"/rep_{01..20}; do
  [[ -d "$REP" ]] || continue

  if [[ "$SKIP_IF_DONE" == "yes" && -f "$REP/eq_start.gro" ]]; then
    echo "=== $(basename "$REP"): eq_start.gro exists; skipping eq_start ==="
    continue
  fi

  run_rep "$REP" "$gpu_slot"
  pid=$!
  batch_pids+=("$pid")

  gpu_slot=$(( (gpu_slot + 1) % NGPU ))

  if [[ ${#batch_pids[@]} -eq $NGPU ]]; then
    echo "--- waiting for batch: ${batch_pids[*]} ---"
    for p in "${batch_pids[@]}"; do
      wait "$p" || { echo "ERROR: PID $p failed"; exit 1; }
    done
    batch_pids=()
  fi
done

if [[ ${#batch_pids[@]} -gt 0 ]]; then
  echo "--- waiting for final batch: ${batch_pids[*]} ---"
  for p in "${batch_pids[@]}"; do
    wait "$p" || { echo "ERROR: PID $p failed"; exit 1; }
  done
fi

echo "All DMSO eq_start equilibrations finished."