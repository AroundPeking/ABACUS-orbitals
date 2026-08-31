#!/bin/bash
#SBATCH --job-name=c_sparse_qstar_rpa
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=c-sparse-qstar-rpa-%j.out
#SBATCH --error=c-sparse-qstar-rpa-%j.err

set -euo pipefail
trap 'status=$?; echo "C_SPARSE_QSTAR_RPA_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

source_root=${SOURCE_ROOT:?set SOURCE_ROOT to the prepared sparse-q reader root}
run_root=${RUN_ROOT:?set RUN_ROOT to a new replay directory}
librpa_debug=${LIBRPA_DEBUG:-f}
source_commit=${SIAB_SOURCE_COMMIT:?set SIAB_SOURCE_COMMIT to the immutable deployment commit}
librpa=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel/chi0_main.exe
expected_librpa_sha=971c839ca3748782b9a6a8ee2f8e9ffd647b66f10448c611e1dba89bb44348cf

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test -s "$source_root/librpa.in"
test -d "$source_root/reader_v1"
test ! -e "$run_root"
test -x "$librpa"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
grep -Eq '^[[:space:]]*n_bands_chi0[[:space:]]*=[[:space:]]*-1[[:space:]]*$' "$source_root/librpa.in"
test "$librpa_debug" = f || test "$librpa_debug" = t

mkdir -p "$run_root"
cp "$source_root/librpa.in" "$run_root/librpa.in"
ln -s "$source_root/reader_v1" "$run_root/reader_v1"
sed -i -E "s|^[[:space:]]*input_dir[[:space:]]*=.*|input_dir = $run_root/reader_v1/|" "$run_root/librpa.in"
sed -i -E "s|^[[:space:]]*debug[[:space:]]*=.*|debug = $librpa_debug|" "$run_root/librpa.in"
cd "$run_root"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=diamond_sparse_eight_qstar_ordinary_allband_sos_replay
  echo source_root=$source_root
  echo run_root=$run_root
  echo librpa_commit=d4810f73
  echo librpa_sha256=$expected_librpa_sha
  echo n_bands_chi0=-1
  echo librpa_debug=$librpa_debug
  echo source_commit=$source_commit
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$source_root/librpa.in" "$librpa"
} > provenance.txt

/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2> librpa.err

grep -q 'libRPA finished successfully' librpa.out
grep -Eq 'Total EcRPA:' librpa.out
test -s librpa_para_nprocs_1_myid_0.out
{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out librpa_para_nprocs_1_myid_0.out
} >> provenance.txt
printf 'success\n' > STATUS

echo C_SPARSE_QSTAR_RPA_OK root=$run_root
