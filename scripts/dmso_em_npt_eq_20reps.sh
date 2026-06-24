#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# DMSO replica setup + EM + short NPT + 5 ns DMSO equilibration
#
# Run this script from inside template_files/
#
# It creates replica folders outside template_files:
#   ../dmso_replicas/rep_01 ... ../dmso_replicas/rep_20
#
# It runs 4 replicas in parallel.
###############################################################################

# ---------------- USER SETTINGS ----------------

GMX="gmx_mpi"

NREP=20
NDMSO=97

NGPU=4
MAX_PARALLEL=4

# CPU threads per replica.
# 4 replicas x 8 threads = 32 CPU threads total.
NTOMP=8

# Protein molecule name in [ molecules ] section of topol.top
PROTEIN_MOL="Protein_chain_A"

# Template input files
BASE_GRO="solv_ions.gro"
BASE_TOP="topol.top"

DMSO_GRO="DMSO.gro"
DMSO_ATOMTYPES="DMSO_atomtypes.itp"
DMSO_MOLECULE="DMSO_molecule.itp"

EM_MDP="minim.mdp"
NPT_MDP="npt.mdp"
DMSO_EQ_MDP="eq_dmso_5ns.mdp"

# Restart-friendly behavior.
# yes = skip a step if its expected .gro output already exists
# no  = redo steps even if outputs exist
SKIP_IF_DONE="yes"

# Replica root directory, outside template_files
TEMPLATE_DIR="$(pwd)"
REPLICA_ROOT="$(dirname "${TEMPLATE_DIR}")/dmso_replicas"

# ------------------------------------------------


die() {
    echo "ERROR: $*" >&2
    exit 1
}


check_file() {
    [[ -e "$1" ]] || die "Missing required file or directory: $1"
}


echo "=== Template directory ==="
echo "${TEMPLATE_DIR}"

echo "=== Replica root directory ==="
echo "${REPLICA_ROOT}"

mkdir -p "${REPLICA_ROOT}"


echo "=== Checking required template files ==="

check_file "${BASE_GRO}"
check_file "${BASE_TOP}"
check_file "${DMSO_GRO}"
check_file "${DMSO_ATOMTYPES}"
check_file "${DMSO_MOLECULE}"
check_file "${EM_MDP}"
check_file "${NPT_MDP}"
check_file "${DMSO_EQ_MDP}"
check_file "posre.itp"
check_file "posre_anchor_CA.itp"
check_file "amber99sb-ildn.ff"

echo "All required template files are present."


echo "=== Checking that topol.top already includes DMSO files ==="

grep -q 'DMSO_atomtypes.itp' "${BASE_TOP}" || die "topol.top does not include DMSO_atomtypes.itp"
grep -q 'DMSO_molecule.itp'  "${BASE_TOP}" || die "topol.top does not include DMSO_molecule.itp"

echo "DMSO include lines found in topol.top."


echo "=== Checking DMSO.gro ==="

python3 - << 'PY'
from pathlib import Path

gro = Path("DMSO.gro").read_text().splitlines()
natoms = int(gro[1].strip())
resnames = set(line[5:10].strip() for line in gro[2:-1])

print(f"DMSO.gro atom count: {natoms}")
print(f"DMSO.gro residue names: {sorted(resnames)}")

if natoms != 10:
    raise SystemExit("ERROR: This script assumes DMSO has 10 atoms.")
if "DMSO" not in resnames:
    raise SystemExit("ERROR: DMSO.gro residue name is not DMSO.")
PY


setup_and_run_replica() {
    local rep="$1"
    local seed="$2"
    local gpu="$3"

    local repdir="${REPLICA_ROOT}/${rep}"

    echo "=================================================================="
    echo "${rep}: starting on physical GPU ${gpu}, seed ${seed}"
    echo "${rep}: directory ${repdir}"
    echo "=================================================================="

    mkdir -p "${repdir}"

    # Copy template files that may be modified or should be local
    cp "${TEMPLATE_DIR}/${BASE_GRO}" "${repdir}/"
    cp "${TEMPLATE_DIR}/${BASE_TOP}" "${repdir}/topol.template.top"

    cp "${TEMPLATE_DIR}/posre.itp" "${repdir}/"
    cp "${TEMPLATE_DIR}/posre_anchor_CA.itp" "${repdir}/"

    cp "${TEMPLATE_DIR}/${DMSO_GRO}" "${repdir}/"
    cp "${TEMPLATE_DIR}/${DMSO_ATOMTYPES}" "${repdir}/"
    cp "${TEMPLATE_DIR}/${DMSO_MOLECULE}" "${repdir}/"

    cp "${TEMPLATE_DIR}/${EM_MDP}" "${repdir}/"
    cp "${TEMPLATE_DIR}/${NPT_MDP}" "${repdir}/"
    cp "${TEMPLATE_DIR}/${DMSO_EQ_MDP}" "${repdir}/"

    # Symlink force field instead of copying the directory
    if [[ ! -e "${repdir}/amber99sb-ildn.ff" ]]; then
        ln -s "${TEMPLATE_DIR}/amber99sb-ildn.ff" "${repdir}/amber99sb-ildn.ff"
    fi

    cd "${repdir}"

    export OMP_NUM_THREADS="${NTOMP}"

    # -------------------------------------------------------------------------
    # 1. Insert DMSO
    # -------------------------------------------------------------------------
    if [[ "${SKIP_IF_DONE}" == "yes" && -f solv_ions_dmso.gro ]]; then
        echo "${rep}: solv_ions_dmso.gro exists; skipping DMSO insertion."
    else
        echo "${rep}: inserting ${NDMSO} DMSO molecules..."

        "${GMX}" insert-molecules \
            -f "${BASE_GRO}" \
            -ci "${DMSO_GRO}" \
            -nmol "${NDMSO}" \
            -replace SOL \
            -seed "${seed}" \
            -o solv_ions_dmso.gro
    fi

    # -------------------------------------------------------------------------
    # 2. Count molecules and generate replica-specific topol.top
    # -------------------------------------------------------------------------
    echo "${rep}: generating replica-specific topol.top..."

    python3 - << PY
from pathlib import Path
from collections import Counter
import re

protein_mol = "${PROTEIN_MOL}"
expected_ndmso = ${NDMSO}

gro = "solv_ions_dmso.gro"
template = "topol.template.top"
outtop = "topol.top"

counts = Counter()

with open(gro) as f:
    lines = f.readlines()[2:-1]

for line in lines:
    resname = line[5:10].strip()
    counts[resname] += 1

nsol = counts["SOL"] // 3
ndmso = counts["DMSO"] // 10
nna = counts["NA"]
ncl = counts["CL"]

print(f"SOL  = {nsol}")
print(f"DMSO = {ndmso}")
print(f"NA   = {nna}")
print(f"CL   = {ncl}")

if ndmso != expected_ndmso:
    raise SystemExit(f"ERROR: Expected {expected_ndmso} DMSO molecules, found {ndmso}")

text = Path(template).read_text()

m = re.search(r'(?im)^\\s*\\[\\s*molecules\\s*\\]\\s*$', text)
if not m:
    raise SystemExit("ERROR: Could not find [ molecules ] section in topol.template.top")

# Keep everything before [ molecules ], then replace the molecule list.
prefix = text[:m.start()].rstrip()

# This order matches the observed coordinate order after insert-molecules:
# Protein -> SOL -> NA -> CL -> DMSO
new_mol_section = f"""
[ molecules ]
; Compound        #mols
{protein_mol:<18s} 1
SOL                {nsol}
NA                 {nna}
CL                 {ncl}
DMSO               {ndmso}
"""

Path(outtop).write_text(prefix + "\\n\\n" + new_mol_section)
PY

    echo "${rep}: final [ molecules ] section:"
    tail -n 8 topol.top

    # -------------------------------------------------------------------------
    # 3. Energy minimization
    # -------------------------------------------------------------------------
    if [[ "${SKIP_IF_DONE}" == "yes" && -f em_dmso.gro ]]; then
        echo "${rep}: em_dmso.gro exists; skipping EM."
    else
        echo "${rep}: grompp EM..."

        "${GMX}" grompp \
            -f "${EM_MDP}" \
            -c solv_ions_dmso.gro \
            -p topol.top \
            -o em_dmso.tpr

        echo "${rep}: running EM on physical GPU ${gpu}..."

        "${GMX}" mdrun \
            -deffnm em_dmso \
            -ntomp "${NTOMP}" \
            -nb gpu \
            -gpu_id "${gpu}"
    fi

    # -------------------------------------------------------------------------
    # 4. Short NPT using original npt.mdp
    # -------------------------------------------------------------------------
    if [[ "${SKIP_IF_DONE}" == "yes" && -f npt_dmso_short.gro ]]; then
        echo "${rep}: npt_dmso_short.gro exists; skipping short NPT."
    else
        echo "${rep}: grompp short NPT..."

        "${GMX}" grompp \
            -f "${NPT_MDP}" \
            -c em_dmso.gro \
            -r em_dmso.gro \
            -p topol.top \
            -o npt_dmso_short.tpr\
            -maxwarn 2

        echo "${rep}: running short NPT on physical GPU ${gpu}..."

        "${GMX}" mdrun \
            -deffnm npt_dmso_short \
            -ntomp "${NTOMP}" \
            -nb gpu \
            -pme gpu \
            -bonded gpu \
            -gpu_id "${gpu}"
    fi

    # -------------------------------------------------------------------------
    # 5. 5 ns DMSO relaxation
    # -------------------------------------------------------------------------
    if [[ "${SKIP_IF_DONE}" == "yes" && -f eq_dmso_5ns.gro ]]; then
        echo "${rep}: eq_dmso_5ns.gro exists; skipping 5 ns DMSO relaxation."
    else
        echo "${rep}: grompp 5 ns DMSO relaxation..."

        "${GMX}" grompp \
            -f "${DMSO_EQ_MDP}" \
            -c npt_dmso_short.gro \
            -t npt_dmso_short.cpt \
            -r npt_dmso_short.gro \
            -p topol.top \
            -o eq_dmso_5ns.tpr

        echo "${rep}: running 5 ns DMSO relaxation on physical GPU ${gpu}..."

        "${GMX}" mdrun \
            -deffnm eq_dmso_5ns \
            -ntomp "${NTOMP}" \
            -nb gpu \
            -pme gpu \
            -bonded gpu \
            -gpu_id "${gpu}"
    fi

    echo "${rep}: done through 5 ns DMSO equilibration."

    cd "${TEMPLATE_DIR}"
}


echo "=== Launching replicas: ${MAX_PARALLEL} at a time on ${NGPU} GPUs ==="

job_count=0

for i in $(seq -w 1 "${NREP}"); do
    rep="rep_${i}"
    seed=$((123450 + 10#$i))
    gpu=$(( (10#$i - 1) % NGPU ))

    setup_and_run_replica "${rep}" "${seed}" "${gpu}" > "${REPLICA_ROOT}/${rep}.log" 2>&1 &

    job_count=$((job_count + 1))

    if [[ "${job_count}" -eq "${MAX_PARALLEL}" ]]; then
        wait
        job_count=0
    fi
done

wait

echo "All ${NREP} replicas finished through 5 ns DMSO equilibration."
echo "Replica folders are in: ${REPLICA_ROOT}"