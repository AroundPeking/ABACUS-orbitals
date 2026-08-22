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

! grep -q '!!SCF IS NOT CONVERGED!!' abacus.out
grep -q 'convergence has been achieved' abacus.out
grep -q 'Read NAO wave functions from OUT.C_DELTA_RESPONSE_GATE/wfs1_nao.txt' abacus.out
grep -q 'Read NAO wave functions from OUT.C_DELTA_RESPONSE_GATE/wfs2_nao.txt' abacus.out
report=$(find . -path './OUT.*/STERNHEIMER_CHI0.dat' -type f -print -quit)
test -n "$report"
scf_log=${report%/STERNHEIMER_CHI0.dat}/running_scf.log
grep -q 'Read electron density from file: OUT.C_DELTA_RESPONSE_GATE/chgs1.cube' "$scf_log"
grep -q 'Read electron density from file: OUT.C_DELTA_RESPONSE_GATE/chgs2.cube' "$scf_log"
grep -q '^status success$' "$report"
grep -Eq '^format v1(_partial)?$' "$report"
grep -q '^frequency_grid_source file$' "$report"
grep -q '^nfreq 6$' "$report"
grep -q '^sternheimer_fd_order 8$' "$report"
# Historical molecular reports use "sternheimer_delta yes"; the current
# feature branch uses "sternheimer_mode delta" for the same physical mode.
grep -Eq '^sternheimer_(mode delta|delta yes)$' "$report"
grep -q '^perturbation_coulomb_kernel full_periodic_poisson$' "$report"
grep -q '^pca_threshold 0.0001$' "$report"
grep -q '^ccp_rmesh_times_input 1$' "$report"
grep -q '^sternheimer_frequency_mpi yes$' "$report"
grep -q '^mpi_ranks 6$' "$report"
grep -q '^all_converged yes$' "$report"
grep -q '^max_solver_relative_residual ' "$report"

# Accepted molecular output can be either consolidated or rank-suffixed.
# v1_sternheimer_chi0_iq_1_ifreq_*_rank*.dat
mapfile -t response_files < <(find . -maxdepth 1 -type f -name 'v1_sternheimer_chi0_iq_1_ifreq_*.dat' | sort)
test "${#response_files[@]}" -eq 6
test -s v1_coulomb_full_iq_1_rank0.dat
for name in stru_out bz_sampling_out basis_out basis_wfc_out basis_aux_out; do
  test -s "$name"
done

channels=$(awk '$1=="abfs_channels" {print $2}' "$report")
occupied=$(awk '$1=="occupied_bands_total" {print $2}' "$report")
solved=$(awk '$1=="solved_equations" {print $2}' "$report")
residual=$(awk '$1=="max_solver_relative_residual" {print $2}' "$report")
test -n "$channels"
test "$occupied" -eq 4
test "$solved" -eq $((6 * occupied * channels))
"$python" - "$residual" <<'PY'
import math
import sys
value = float(sys.argv[1])
assert math.isfinite(value) and value <= 1.0e-6
PY

out_dir=${report%/STERNHEIMER_CHI0.dat}
export PYTHONPATH=$source_dir${PYTHONPATH:+:$PYTHONPATH}
"$python" - "$out_dir/eig_occ.txt" <<'PY'
import sys
from pathlib import Path
from response_contract import parse_eig_occ
record = parse_eig_occ(Path(sys.argv[1]))
assert record.spin_counts == {1: 3, 2: 1}
PY

"$python" - "$branch" "$report" "$expected_abacus_sha" "$source_commit" "$residual" \
  "$channels" "$solved" "${response_files[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

branch, report, exe_sha, source_commit, residual, channels, solved, *files = sys.argv[1:]
def record(path):
    path = Path(path)
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "status": "RESPONSE_COMPLETE",
    "branch": branch,
    "abacus_sha256": exe_sha,
    "workflow_source_commit": source_commit,
    "frequency_grid_source": "file",
    "nfreq": 6,
    "finite_difference_order": 8,
    "sternheimer_delta": True,
    "virtual_source": "all_ks_bands_implicit_feature_branch",
    "pca_threshold": 1.0e-4,
    "coulomb_kernel": "full",
    "solver_tolerance": 1.0e-6,
    "max_solver_relative_residual": float(residual),
    "abfs_channels": int(channels),
    "solved_equations": int(solved),
    "report": record(report),
    "response_files": [record(path) for path in files],
    "full_coulomb": record("v1_coulomb_full_iq_1_rank0.dat"),
}
temporary = Path("RESPONSE_COMPLETE.json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
temporary.replace("RESPONSE_COMPLETE.json")
PY

rm -f .response-running
trap - EXIT
echo "RESPONSE_OK branch=$branch channels=$channels residual=$residual"
