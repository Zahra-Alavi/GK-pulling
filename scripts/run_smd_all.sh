#!/usr/bin/env bash
# run_smd_all.sh
# Runs SMD pulling (grompp + mdrun) for each rep_XX directory, using eq_start.gro
# and eq_start.cpt as the starting structure and checkpoint.
# Parallelizes across 4 GPUs; at most one job per GPU at a time.
#
# Inputs:  rep_XX/{eq_start.gro, eq_start.cpt, topol.top, index.ndx}, pull_k1000_v0p0005.mdp
# Outputs (per rep_XX/): pull.tpr, pull.{trr,xtc,edr,log}, pullx.xvg, pullf.xvg, pull.runlog
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

PULL_MDP="$BASE_DIR/pull_k1000_v0p0005.mdp"

DEFFNM="pull"
PX="pullx.xvg"
PF="pullf.xvg"

run_one_rep () {
    local REP="$1"
    local GPU_ID="$2"
    local REP_NAME
    REP_NAME="$(basename "$REP")"

    echo "=== $REP_NAME: starting on GPU $GPU_ID ==="

    (
        cd "$REP"

        if [[ ! -f eq_start.gro ]]; then
            echo "ERROR: $REP_NAME missing eq_start.gro"
            exit 1
        fi

        if [[ ! -f eq_start.cpt ]]; then
            echo "ERROR: $REP_NAME missing eq_start.cpt"
            exit 1
        fi

        if [[ ! -f "$PULL_MDP" ]]; then
            echo "ERROR: $REP_NAME missing $PULL_MDP"
            exit 1
        fi

        gmx_mpi grompp \
            -f "$PULL_MDP" \
            -c eq_start.gro \
            -t eq_start.cpt \
            -p topol.top \
            -r eq_start.gro \
            -n index.ndx \
            -o "${DEFFNM}.tpr" \
            -maxwarn 1

        CUDA_VISIBLE_DEVICES="$GPU_ID" mpirun -np 1 gmx_mpi mdrun \
            -s "${DEFFNM}.tpr" \
            -deffnm "$DEFFNM" \
            -px "$PX" \
            -pf "$PF" \
            -nb gpu \
            -ntomp "$NTOMP"

        echo "=== $REP_NAME: finished on GPU $GPU_ID ==="
    ) > "$REP/${DEFFNM}.runlog" 2>&1
}

[[ -f "$PULL_MDP" ]] || { echo "ERROR: pull mdp not found: $PULL_MDP"; exit 1; }

declare -a RUNNING_PIDS=()
declare -a RUNNING_GPUS=()
declare -a REPS=()

for ((i = START_REP; i <= END_REP; i++)); do
    printf -v rep_name "rep_%02d" "$i"
    REP="$BASE_DIR/$rep_name"
    [[ -d "$REP" ]] || { echo "ERROR: replica directory not found: $REP"; exit 1; }
    REPS+=("$REP")
done

next_rep=0

while [[ $next_rep -lt ${#REPS[@]} || ${#RUNNING_PIDS[@]} -gt 0 ]]; do

    while [[ ${#RUNNING_PIDS[@]} -lt $NGPU && $next_rep -lt ${#REPS[@]} ]]; do
        GPU_ID=-1

        for g in $(seq 0 $((NGPU - 1))); do
            used=0
            for rg in ${RUNNING_GPUS[@]:+"${RUNNING_GPUS[@]}"}; do
                [[ "$rg" == "$g" ]] && used=1
            done
            if [[ $used -eq 0 ]]; then
                GPU_ID="$g"
                break
            fi
        done

        REP="${REPS[$next_rep]}"
        REP_NAME="$(basename "$REP")"

        echo "Launching $REP_NAME on GPU $GPU_ID"

        run_one_rep "$REP" "$GPU_ID" &
        pid=$!

        RUNNING_PIDS+=("$pid")
        RUNNING_GPUS+=("$GPU_ID")

        next_rep=$((next_rep + 1))
    done

    sleep 10

    new_pids=()
    new_gpus=()

    for i in "${!RUNNING_PIDS[@]}"; do
        pid="${RUNNING_PIDS[$i]}"
        gpu="${RUNNING_GPUS[$i]}"

        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_gpus+=("$gpu")
        else
            wait "$pid" || {
                echo "ERROR: one SMD run failed. Check the corresponding pull.runlog."
                exit 1
            }
            echo "PID $pid finished; GPU $gpu is now free"
        fi
    done

    RUNNING_PIDS=("${new_pids[@]}")
    RUNNING_GPUS=("${new_gpus[@]}")

done

echo ""
echo "All SMD runs finished."
