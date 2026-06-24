#!/usr/bin/env bash
#
# Runs biased equilibration (grompp + mdrun) in each pre-populated rep_XX directory.
# Assumes each rep_XX/ already contains: npt.gro, topol.top, index.ndx,
# posre.itp, posre_anchor_CA.itp, amber99sb-ildn.ff, and eq star mdp.
#
# Parallelizes across 4 GPUs (GPU IDs 0-3).
# At most 4 mdrun jobs run simultaneously; grompp steps are fast and run serially
# before each batch.
#
# Outputs (per rep_XX/): eq_start.tpr, eq_start.{trr,xtc,edr,log},
#   eq_pullf.xvg, eq_pullx.xvg, eq_start.runlog. The strucure (gro) and coordinate checkpoint (cpt)files generated here are used for the next step. 

#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPU=4
NTOMP=16

START_REP="${1:-1}"
END_REP="${2:-20}"

[[ "$START_REP" =~ ^[0-9]+$ && "$END_REP" =~ ^[0-9]+$ ]] || {
  echo "Usage: $0 [START_REP [END_REP]]"
  exit 1
}
(( START_REP >= 1 && START_REP <= END_REP && END_REP <= 99 )) || {
  echo "ERROR: require 1 <= START_REP <= END_REP <= 99"
  exit 1
}

run_rep () {
  local REP="$1"
  local GPU_ID="$2"
  local REP_NAME
  REP_NAME="$(basename "$REP")"

  echo "=== $REP_NAME: grompp ==="
  (
    cd "$REP"
    gmx_mpi grompp \
      -f eq_start_k1000.mdp \
      -c npt.gro \
      -p topol.top \
      -r npt.gro \
      -n index.ndx \
      -o eq_start.tpr \
      -maxwarn 1
  )

  echo "=== $REP_NAME: mdrun on GPU $GPU_ID ==="
  (
    cd "$REP"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    mpirun -np 1 gmx_mpi mdrun \
      -s eq_start.tpr \
      -deffnm eq_start \
      -pf eq_pullf.xvg \
      -px eq_pullx.xvg \
      -nb gpu \
      -ntomp "$NTOMP"
  ) > "$REP/eq_start.runlog" 2>&1 &
}

gpu_slot=0
batch_pids=()

for ((i = START_REP; i <= END_REP; i++)); do
  printf -v rep_name "rep_%02d" "$i"
  REP="$BASE_DIR/$rep_name"
  [[ -d "$REP" ]] || { echo "ERROR: replica directory not found: $REP"; exit 1; }

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

echo "All equilibrations finished."
