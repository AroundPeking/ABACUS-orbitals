#!/bin/bash
#SBATCH --partition=p1
#SBATCH --job-name=c_atom_solid_opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000M
#SBATCH --time=1-00:00:00
#SBATCH --exclusive

set -euo pipefail
trap 'status=$?; echo "C_ATOM_SOLID_BALANCED_OPT_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${REPO_ROOT:?}"
: "${CAMPAIGN_ROOT:?}"
: "${ATOM_PAIR_ROOT:?}"
: "${RUN_ROOT:?}"
: "${RUN_MODE:?}"
: "${EXPECTED_SIAB_COMMIT:?}"

candidate_profile=${CANDIDATE_PROFILE:-one_g}

case "$RUN_MODE" in
  pilot)
    max_steps=2
    minimum_steps=0
    plateau_patience=2
    ;;
  production)
    max_steps=500
    minimum_steps=200
    plateau_patience=250
    ;;
  *)
    echo "unsupported run mode: $RUN_MODE" >&2
    exit 2
    ;;
esac

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NNODES:?}" = 1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_NTASKS_PER_NODE:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test "$(cat "$REPO_ROOT/.git/HEAD")" = "$EXPECTED_SIAB_COMMIT"
test ! -e "$RUN_ROOT"

optimizer=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/optimize_periodic_basis.py
truncator=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/truncate_periodic_coefficients.py
module load python/3.9.22
runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
python_exe=$(command -v python3)
test -x "$python_exe"
test -s "$runtime/site-packages/torch/lib/libtorch_cpu.so"
export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages:$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
"$python_exe" - "$optimizer" <<'PY'
import runpy
import sys

import torch


runpy.run_path(sys.argv[1], run_name="siab_optimizer_import_smoke")
parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
torch.optim.Adam([parameter], lr=1.0e-3)
print("validated torch optimizer runtime", torch.__version__)
PY

producer_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q1=$producer_root/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
q2=$producer_root/c-diamond-g24-fd8-nfreq6-q2-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
source_initial=$CAMPAIGN_ROOT/runs/product-pca-20260825/nested-one-g-screen-545e108d/C_3s3p2d1f1g_keep_g1.txt
source_initial_sha=29a89abad8ffab7e55316a7e4ca908e14200cb5382274af6f8a6adaa91e895ca
atomic_response=/data/home/df_iopcas_ghj/gw/260823/c-atom-siab-reference-fixedocc-p1-8cf890e8/OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat
atomic_response_sha=e976c164595758029cb91ebe3913af6865780ef95fdb48954c07075bd0c7e3ff
atomic_source=$ATOM_PAIR_ROOT/OUT.C_ATOM_SIAB_SOURCE/STERNHEIMER_SIAB_SOURCE_V1.dat
pair_gate=$ATOM_PAIR_ROOT/SOURCE_ONLY_COMPLETE.json

case "$candidate_profile" in
  one_g)
    candidate_label=3s3p2d1f1g_keep_g1
    optimizer_nu=3,3,2,1,1
    optimizer_fixed_nu=2,2,1,0,0
    optimizer_max_l=4
    initial=$source_initial
    ;;
  no_f)
    candidate_label=3s3p2d
    optimizer_nu=3,3,2
    optimizer_fixed_nu=2,2,1
    optimizer_max_l=2
    initial=$RUN_ROOT/input/C_3s3p2d_joint_initial.txt
    ;;
  *)
    echo "unsupported candidate profile: $candidate_profile" >&2
    exit 2
    ;;
esac

test -s "$optimizer"
test -s "$truncator"
test -s "$source_initial"
test "$(sha256sum "$source_initial" | awk '{print $1}')" = "$source_initial_sha"
test -s "$atomic_response"
test "$(sha256sum "$atomic_response" | awk '{print $1}')" = "$atomic_response_sha"
test -s "$atomic_source"
test -s "$pair_gate"
for dataset in "$q1" "$q2"; do
  test -s "$dataset/manifest.dat"
  grep -qx 'status success' "$dataset/status.dat"
  grep -qx 'all_converged yes' "$dataset/status.dat"
  grep -qx 'auxiliary_basis_source product_pca' "$dataset/manifest.dat"
  grep -qx 'frequency_count 6' "$dataset/manifest.dat"
  grep -qx 'kernel full_coulomb' "$dataset/manifest.dat"
done
grep -qx 'selected_iq 1' "$q1/manifest.dat"
grep -qx 'selected_iq 22' "$q2/manifest.dat"

"$python_exe" - "$pair_gate" "$atomic_response" "$atomic_source" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


payload = json.load(open(sys.argv[1], encoding="ascii"))
if payload.get("status") != "success":
    raise SystemExit("source-only pair gate is not successful")
for key, path in (("target", sys.argv[2]), ("source", sys.argv[3])):
    record = payload.get(key, {})
    if Path(record.get("path", "")).resolve() != Path(path).resolve():
        raise SystemExit(f"source-only pair gate {key} path mismatch")
    if record.get("sha256") != sha256(path):
        raise SystemExit(f"source-only pair gate {key} SHA256 mismatch")
PY

mkdir -p "$RUN_ROOT"
if test "$candidate_profile" = no_f; then
  mkdir -p "$RUN_ROOT/input"
  "$python_exe" "$truncator" \
    --input "$source_initial" \
    --output "$initial" \
    --element C \
    --radial-rows 31 \
    --input-nu 3,3,2,1,1 \
    --target-lmax 2 \
    --report "$RUN_ROOT/input/TRUNCATION.json" \
    > "$RUN_ROOT/input/truncate.stdout"
fi
test -s "$initial"
{
  echo status=running
  echo purpose=C_atom_solid_family_balanced_response_optimization
  echo run_mode=$RUN_MODE
  echo siab_commit=$EXPECTED_SIAB_COMMIT
  echo candidate=$candidate_label
  echo candidate_profile=$candidate_profile
  echo fixed_basis=2s2p1d
  echo objective='0.5*C_atom_normalized_loss+0.5*C_solid_normalized_loss'
  echo solid_training_q_indices=1,22
  echo atomic_frequency_count=16
  echo solid_frequency_count=6
  echo maximum_steps=$max_steps
  echo python_executable=$python_exe
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$SLURM_CPUS_PER_TASK
  sha256sum "$0" "$optimizer" "$truncator" "$source_initial" "$initial" \
    "$atomic_response" "$atomic_source" \
    "$pair_gate" "$q1/manifest.dat" "$q2/manifest.dat" \
    "$runtime/site-packages/torch/lib/libtorch_cpu.so"
} > "$RUN_ROOT/provenance.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

/usr/bin/time -v -o "$RUN_ROOT/optimize.time" \
  "$python_exe" -u "$optimizer" \
    --dataset "$q1" \
    --dataset-family C_solid \
    --dataset "$q2" \
    --dataset-family C_solid \
    --atomic-response "$atomic_response" \
    --atomic-source "$atomic_source" \
    --atomic-family C_atom \
    --initial "$initial" \
    --output-directory "$RUN_ROOT/result" \
    --siab-commit "$EXPECTED_SIAB_COMMIT" \
    --element C \
    --nu "$optimizer_nu" \
    --fixed-nu "$optimizer_fixed_nu" \
    --radial-rows 31 \
    --max-l "$optimizer_max_l" \
    --learning-rate 1e-3 \
    --max-steps "$max_steps" \
    --minimum-steps "$minimum_steps" \
    --plateau-patience "$plateau_patience" \
    --plateau-relative-improvement 1e-7 \
    --maximum-backtracks 40 \
    --block-cache-workers 8 \
    --occupied-capture-reference fixed_prefix \
    --omitted-reference-projection-validation layout \
    > "$RUN_ROOT/optimize.log" 2>&1

grep -q '"status": "success"' "$RUN_ROOT/result/STATUS.json"
test -s "$RUN_ROOT/result/OPTIMIZATION_RESULT.json"
test -s "$RUN_ROOT/result/ORBITAL_RESULTS.txt"
"$python_exe" - "$RUN_ROOT/result/OPTIMIZATION_RESULT.json" <<'PY'
import json
import math
import sys


result = json.load(open(sys.argv[1], encoding="ascii"))
if result["dataset_families"] != ["C_solid", "C_solid"]:
    raise SystemExit("solid family labels are incorrect")
if result["atomic_family"] != "C_atom":
    raise SystemExit("atomic family label is incorrect")
for field in ("initial_family_losses", "best_family_losses"):
    losses = result[field]
    if set(losses) != {"C_atom", "C_solid"}:
        raise SystemExit(f"{field} does not contain both physical families")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in losses.values()):
        raise SystemExit(f"{field} contains a non-finite loss")
if result["best_loss"] > result["initial_loss"] + 1.0e-14:
    raise SystemExit("balanced objective did not improve")
for family in ("C_atom", "C_solid"):
    if (
        result["best_family_losses"][family]
        > result["initial_family_losses"][family] + 1.0e-14
    ):
        raise SystemExit(f"{family} loss did not improve")
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' "$RUN_ROOT/optimize.time"
  sha256sum "$RUN_ROOT/result/OPTIMIZATION_RESULT.json" "$RUN_ROOT/result/ORBITAL_RESULTS.txt"
} >> "$RUN_ROOT/provenance.txt"

echo "C_ATOM_SOLID_BALANCED_OPT_OK mode=$RUN_MODE profile=$candidate_profile output=$RUN_ROOT"
