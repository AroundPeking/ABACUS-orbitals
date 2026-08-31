#!/bin/bash
#SBATCH --job-name=c_candidate_freqdiag
#SBATCH --partition=48cp2,p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --output=c-candidate-freqdiag-%j.out
#SBATCH --error=c-candidate-freqdiag-%j.err

set -euo pipefail
trap 'status=$?; echo "C_CANDIDATE_FREQDIAG_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

source_root=${SOURCE_ROOT:?set SOURCE_ROOT to the completed sparse-q reader root}
run_root=${RUN_ROOT:?set RUN_ROOT to a new diagnostic directory}
expected_ecrpa=${EXPECTED_ECRPA:?set EXPECTED_ECRPA to the pinned d4810f73 sparse-q value}
source_commit=${SIAB_SOURCE_COMMIT:?set SIAB_SOURCE_COMMIT to the immutable deployment commit}
librpa=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-freqdiag-87cf66a7/build-df-intel/chi0_main.exe
librpa_commit=87cf66a768afaa45714a5aed5431b53ac7ea0eb3
parent_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
expected_librpa_sha=de334e398cccd3343da3003bd3c654474faabc467e72ba321ff3d215e4e300e4

case "${SLURM_JOB_PARTITION:?}" in
  p1|48cp2) ;;
  *) echo "unsupported partition: $SLURM_JOB_PARTITION" >&2; exit 2 ;;
esac
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test -s "$source_root/librpa.in"
test -s "$source_root/INPUT_SHA256.txt"
test -d "$source_root/reader_v1"
test ! -e "$run_root"
test -x "$librpa"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
grep -Eq '^[[:space:]]*n_bands_chi0[[:space:]]*=[[:space:]]*-1[[:space:]]*$' "$source_root/librpa.in"

mkdir -p "$run_root"
cp "$source_root/librpa.in" "$run_root/librpa.in"
cp "$source_root/INPUT_SHA256.txt" "$run_root/SOURCE_INPUT_SHA256.txt"
ln -s "$source_root/reader_v1" "$run_root/reader_v1"
sed -i -E "s|^[[:space:]]*input_dir[[:space:]]*=.*|input_dir = $run_root/reader_v1/|" "$run_root/librpa.in"
sed -i -E 's|^[[:space:]]*debug[[:space:]]*=.*|debug = t|' "$run_root/librpa.in"
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
  echo purpose=diagnostic_only_candidate_qstar_frequency_decomposition
  echo source_root=$source_root
  echo run_root=$run_root
  echo source_commit=$source_commit
  echo librpa_commit=$librpa_commit
  echo parent_physics_commit=$parent_commit
  echo librpa_sha256=$expected_librpa_sha
  echo expected_ecrpa=$expected_ecrpa
  echo n_bands_chi0=-1
  echo partition=$SLURM_JOB_PARTITION
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$source_root/librpa.in" "$source_root/INPUT_SHA256.txt" "$librpa"
} > provenance.txt

/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2> librpa.err

grep -q 'libRPA finished successfully' librpa.out
grep -Eq 'Total EcRPA:' librpa.out
test -s librpa_para_nprocs_1_myid_0.out
test "$(grep -c '^RPA normal split ifreq=' librpa.out)" -eq 384
test "$(grep -c '^RPA freqdiag ifreq=' librpa.out)" -eq 384
grep '^RPA normal split ifreq=' librpa.out > normal_split.dat
grep '^RPA freqdiag ifreq=' librpa.out > freqdiag.dat
awk '{raw=$6; sub(/^raw=\(/,"",raw); sub(/\)$/,"",raw); split(raw,value,","); if ((value[1]+0)!=0 || (value[2]+0)!=0) print}' freqdiag.dat > freqdiag_nonzero.dat
test "$(wc -l < freqdiag_nonzero.dat)" -eq 48
actual_ecrpa=$(awk '/Total EcRPA:/{value=$NF} END{print value}' librpa.out)
awk -v actual="$actual_ecrpa" -v expected="$expected_ecrpa" 'BEGIN {delta=actual-expected; if (delta<0) delta=-delta; exit !(delta<=5e-10)}'

{
  echo status=success
  echo actual_ecrpa=$actual_ecrpa
  echo normal_split_records=$(wc -l < normal_split.dat)
  echo freqdiag_records=$(wc -l < freqdiag.dat)
  echo nonzero_qstar_freqdiag_records=$(wc -l < freqdiag_nonzero.dat)
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in SOURCE_INPUT_SHA256.txt librpa.out normal_split.dat freqdiag.dat freqdiag_nonzero.dat librpa_para_nprocs_1_myid_0.out
} >> provenance.txt
printf 'success\n' > STATUS

echo C_CANDIDATE_FREQDIAG_OK root=$run_root
