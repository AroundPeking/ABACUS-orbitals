#!/bin/bash
#SBATCH --partition=640
#SBATCH --job-name=c_delta_response
#SBATCH --array=0-1%2
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180000M
#SBATCH --time=1-00:00:00
#SBATCH --exclusive

set -euo pipefail

root=${DELTA_GATE_ROOT:?}
abacus=${ABACUS_EXE:?}
expected_abacus_sha=${ABACUS_SHA256:?}
runtime_env=${ABACUS_ENV_SCRIPT:?}
python=${PYTHON_EXE:-python3}
source_dir=${DELTA_GATE_SOURCE:?}
source_commit=${DELTA_GATE_SOURCE_COMMIT:?}
branches=(fixed free)
branch=${branches[${SLURM_ARRAY_TASK_ID:?}]}
case_dir=$root/branches/$branch

test "${SLURM_JOB_PARTITION:?}" = 640
test "${SLURM_NNODES:?}" -eq 6
test "${SLURM_NTASKS:?}" -eq 6
test "${SLURM_NTASKS_PER_NODE:?}" -eq 1
test "${SLURM_CPUS_PER_TASK:?}" -eq 48
test -f "$root/PREPARATION_MANIFEST.json"
test -s "$root/SOURCE_COMMIT.txt"
test "$(cat "$root/SOURCE_COMMIT.txt")" = "$source_commit"
test -x "$abacus"
test -s "$runtime_env"
test -s "$source_dir/response_contract.py"
test -s "$source_dir/finalize_response_branch.py"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"

source "$runtime_env"
test ! -e "$case_dir/RESPONSE_COMPLETE.json"
test ! -e "$case_dir/.response-running"
touch "$case_dir/.response-running"
trap 'rm -f "$case_dir/.response-running"' EXIT

# RESULT_SUMMARY.json is the accepted PBE gate, not merely a completed producer.
"$python" - "$root/PREPARATION_MANIFEST.json" "$branch" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
pbe_result = Path(manifest["pbe_gate_root"]) / "RESULT_SUMMARY.json"
data = json.loads(pbe_result.read_text(encoding="ascii"))
assert data["status"] == "PBE_GATE_PASSED"
assert data["zero_field_comparison_status"] == "ZERO_FIELD_COMPARISON_PASSED"
assert data.get("blocked_on") is None
actual = hashlib.sha256(pbe_result.read_bytes()).hexdigest()
assert actual == manifest["pbe_result_sha256"]
assert sys.argv[2] in manifest["branches"]
PY

grep -Eq '^sternheimer_nfreq[[:space:]]+6$' "$case_dir/INPUT"
grep -Eq '^sternheimer_frequency_grid_file[[:space:]]+fixed_frequency_grid.dat$' "$case_dir/INPUT"
grep -Eq '^sternheimer_frequency_mpi[[:space:]]+1$' "$case_dir/INPUT"
grep -Eq '^sternheimer_fd_order[[:space:]]+8$' "$case_dir/INPUT"
grep -Eq '^sternheimer_delta[[:space:]]+1$' "$case_dir/INPUT"
grep -Eq '^sternheimer_delta_max_states[[:space:]]+0$' "$case_dir/INPUT"
grep -Eq '^exx_pca_threshold[[:space:]]+1e-4$' "$case_dir/INPUT"
grep -Eq '^exx_singularity_correction[[:space:]]+massidda$' "$case_dir/INPUT"
grep -Eq '^init_wfc[[:space:]]+file$' "$case_dir/INPUT"
grep -Eq '^init_chg[[:space:]]+file$' "$case_dir/INPUT"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export OMP_STACKSIZE=2G
export MKL_DYNAMIC=FALSE
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export ABACUS_STERNHEIMER_CHANNEL_THREADS=32
export ABACUS_STERNHEIMER_FD_ST_FREQ_RANK_SHIFT=1
export ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-6
export ABACUS_STERNHEIMER_FD_ST_MAX_ITER=300
export ABACUS_STERNHEIMER_FD_ST_SOLVER_MAX_ITER=300
export ABACUS_STERNHEIMER_DELTA_A_BLOCK=reference_value_gradient

cd "$case_dir"
{
  echo "status=running"
  echo "branch=$branch"
  echo "job_id=${SLURM_JOB_ID:?}"
  echo "nodes=$SLURM_NNODES"
  echo "mpi_ranks=$SLURM_NTASKS"
  echo "omp_threads=$OMP_NUM_THREADS"
  echo "abacus_sha256=$expected_abacus_sha"
  echo "workflow_source_commit=$source_commit"
  sha256sum INPUT STRU KPT fixed_frequency_grid.dat ./*.upf ./*.orb SOURCE_EIG_OCC.txt
} > RESPONSE_PROVENANCE.txt

ulimit -s unlimited
/usr/bin/time -v -o abacus.time \
  mpirun -np "$SLURM_NTASKS" -ppn 1 "$abacus" > abacus.out 2>&1

export PYTHONPATH=$source_dir${PYTHONPATH:+:$PYTHONPATH}
"$python" "$source_dir/finalize_response_branch.py" \
  --root "$root" \
  --branch "$branch" \
  --expected-abacus-sha "$expected_abacus_sha" \
  --producer-source-commit "$source_commit" \
  --finalizer-source-commit "$source_commit"

rm -f .response-running
trap - EXIT
