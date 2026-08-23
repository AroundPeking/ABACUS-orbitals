#!/bin/bash
#SBATCH --partition=640
#SBATCH --job-name=c_siab_abfs_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180000M
#SBATCH --time=02:00:00
#SBATCH --exclusive

set -euo pipefail

root=${SIAB_REFERENCE_ROOT:?}
abacus=${ABACUS_EXE:?}
expected_abacus_sha=${ABACUS_SHA256:?}
runtime_env=${ABACUS_ENV_SCRIPT:?}
diag=$root/abfs_diag

test ! -e "$diag"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
mkdir "$diag"
cp "$root"/INPUT "$root"/STRU "$root"/KPT "$root"/*.upf "$root"/*.orb \
  "$root"/*.abfs "$root"/fixed_frequency_grid_nfreq16.dat "$diag"/
cp -R "$root/OUT.C_SIAB_REFERENCE" "$diag/OUT.C_SIAB_REFERENCE"

source "$runtime_env"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_STACKSIZE=2G
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1

cd "$diag"
ulimit -s unlimited
/usr/bin/time -v -o abacus.time mpirun -np 1 -ppn 1 "$abacus" > abacus.out 2>&1
status=OUT.C_SIAB_REFERENCE/STERNHEIMER_SIAB_STATUS.dat
test -s "$status"
test -s STERNHEIMER_ABFS_CHANNELS.dat
test -s STERNHEIMER_SIAB_COULOMB_WHITENING.dat
test -s STERNHEIMER_SIAB_MEMORY.dat
grep -Eq '^status[[:space:]]+abfs_diag_only$' "$status"
grep -Eq '^nfreq[[:space:]]+16$' "$status"
touch ABFS_DIAG_COMPLETE
