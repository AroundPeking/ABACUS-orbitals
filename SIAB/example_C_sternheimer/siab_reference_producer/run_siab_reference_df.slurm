#!/bin/bash
#SBATCH --partition=9242
#SBATCH --job-name=c_siab_reference_df
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=378000M
#SBATCH --time=UNLIMITED
#SBATCH --exclusive

set -euo pipefail

root=${SIAB_REFERENCE_ROOT:?}
abacus=${ABACUS_EXE:?}
expected_abacus_sha=${ABACUS_SHA256:?}
abacus_source_commit=${ABACUS_SOURCE_COMMIT:?}
workflow_source_commit=${SIAB_REFERENCE_SOURCE_COMMIT:?}

test "${SLURM_JOB_PARTITION:?}" = 9242
test "${SLURM_NNODES:?}" -eq 10
test "${SLURM_NTASKS:?}" -eq 10
test "${SLURM_NTASKS_PER_NODE:?}" -eq 1
test "${SLURM_CPUS_PER_TASK:?}" -eq 96
test -x "$abacus"
test -s "$root/PREPARATION_MANIFEST.json"
test ! -e "$root/SIAB_REFERENCE_COMPLETE.json"
test "$abacus_source_commit" = 8cf890e8c09cc4d09bf8aca246158f5fca27d7f1
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"

grep -Eq '^out_sternheimer_librpa[[:space:]]+0$' "$root/INPUT"
grep -Eq '^out_sternheimer_siab[[:space:]]+1$' "$root/INPUT"
grep -Eq '^sternheimer_nfreq[[:space:]]+16$' "$root/INPUT"
grep -Eq '^sternheimer_frequency_mpi[[:space:]]+1$' "$root/INPUT"
grep -Eq '^sternheimer_channel_mpi[[:space:]]+1$' "$root/INPUT"
grep -Eq '^sternheimer_mpi_layout[[:space:]]+global_equation$' "$root/INPUT"
grep -Eq '^sternheimer_fd_order[[:space:]]+8$' "$root/INPUT"
grep -Eq '^sternheimer_siab_lmax[[:space:]]+4$' "$root/INPUT"
grep -Eq '^sternheimer_siab_coulomb_threshold[[:space:]]+1e-10$' "$root/INPUT"
grep -Eq '^exx_pca_threshold[[:space:]]+10$' "$root/INPUT"
grep -Eq '^ABFS_ORBITAL$' "$root/STRU"

module load oneapi/2024.2
export LD_LIBRARY_PATH=/data/home/df_iopcas_ghj/app/elpa-build/lib:/data/app/libxc/6.2.2-intel/lib:${LD_LIBRARY_PATH:-}
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
export I_MPI_FABRICS=shm:ofi
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export ABACUS_STERNHEIMER_CHANNEL_THREADS=96
export ABACUS_STERNHEIMER_FD_ST_FREQ_RANK_SHIFT=1
export ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-6
export ABACUS_STERNHEIMER_FD_ST_MAX_ITER=300
export ABACUS_STERNHEIMER_FD_ST_SOLVER_MAX_ITER=300
export ABACUS_STERNHEIMER_DELTA_A_BLOCK=reference_value_gradient

cd "$root"
{
  echo "status=running"
  echo "job_id=${SLURM_JOB_ID:?}"
  echo "nodes=$SLURM_NNODES"
  echo "mpi_ranks=$SLURM_NTASKS"
  echo "omp_threads=$OMP_NUM_THREADS"
  echo "abacus_sha256=$expected_abacus_sha"
  echo "abacus_source_commit=$abacus_source_commit"
  echo "workflow_source_commit=$workflow_source_commit"
  sha256sum INPUT STRU KPT fixed_frequency_grid_nfreq16.dat ./*.upf ./*.orb ./*.abfs SOURCE_EIG_OCC.txt
} > SIAB_REFERENCE_PROVENANCE_DF.txt

ulimit -s unlimited
/usr/bin/time -v -o abacus_df.time \
  mpirun -ppn 1 -np "$SLURM_NTASKS" "$abacus" > abacus_df.out 2>&1

status=OUT.C_SIAB_REFERENCE/STERNHEIMER_SIAB_STATUS.dat
matrix=OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat
test -s "$status"
test -s "$matrix"
test -s STERNHEIMER_SIAB_MEMORY.dat
test -s STERNHEIMER_ABFS_CHANNELS.dat
test -s STERNHEIMER_SIAB_COULOMB_WHITENING.dat
grep -Eq '^status[[:space:]]+success$' "$status"
grep -Eq '^all_converged[[:space:]]+yes$' "$status"
grep -Eq '^nfreq[[:space:]]+16$' "$status"

python3 - "$root" "$expected_abacus_sha" "$abacus_source_commit" "$workflow_source_commit" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
files = (
    root / "OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat",
    root / "OUT.C_SIAB_REFERENCE/STERNHEIMER_SIAB_STATUS.dat",
    root / "STERNHEIMER_SIAB_MEMORY.dat",
    root / "STERNHEIMER_ABFS_CHANNELS.dat",
    root / "STERNHEIMER_SIAB_COULOMB_WHITENING.dat",
)
payload = {
    "status": "SIAB_REFERENCE_COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "abacus_sha256": sys.argv[2],
    "abacus_source_commit": sys.argv[3],
    "workflow_source_commit": sys.argv[4],
    "files": {
        str(path.relative_to(root)): {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    },
}
(root / "SIAB_REFERENCE_COMPLETE.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
)
PY
