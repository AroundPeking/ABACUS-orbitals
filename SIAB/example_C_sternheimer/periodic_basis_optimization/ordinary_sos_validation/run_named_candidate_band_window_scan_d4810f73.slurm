#!/bin/bash
#SBATCH --job-name=c_basis_bscan
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-basis-band-scan-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-basis-band-scan-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NAMED_BAND_SCAN_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

run_root=${RUN_ROOT:?missing completed all-band SOS run root}
diag_root=${DIAG_ROOT:?missing immutable diagnostic output root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
script=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_named_candidate_band_window_scan.py
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
windows=10:20,14:28,18:36,22:44,26:52,30:60,34:68
time_path=${diag_root}.time

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$diag_root"
test ! -e "$time_path"
test -s "$run_root/atom-sos/RESULT_SUMMARY.txt"
test -s "$run_root/solid-sos/RESULT_SUMMARY.txt"
grep -qx 'status success' "$run_root/atom-sos/RESULT_SUMMARY.txt"
grep -qx 'status success' "$run_root/solid-sos/RESULT_SUMMARY.txt"
test -s "$script" && test -x "$librpa"
test -s "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

/usr/bin/time -v -o "$time_path" \
  /data/home/df_iopcas_ghj/app/miniconda3/bin/python "$script" \
    --run-root "$run_root" \
    --output-root "$diag_root" \
    --librpa "$librpa" \
    --windows "$windows" \
    --nfreq 6

test "$(cat "$diag_root/STATUS")" = success
mv "$time_path" "$diag_root/time.txt"
{
  echo status=success
  echo purpose=matched_atom_solid_band_window_diagnosis
  echo source_all_band_run_root=$run_root
  echo windows=$windows
  echo nfreq=6
  echo minimax_grid=fixed_to_each_side_all_band_window
  echo librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
  echo librpa_sha256=$expected_librpa_sha
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  echo mkl_threads=$MKL_NUM_THREADS
  sha256sum "$0" "$script" "$diag_root/RESULT.json" "$diag_root/RESULT.tsv" "$diag_root/time.txt" "$build_provenance" "$librpa"
} > "$diag_root/provenance.txt"
echo "C_NAMED_BAND_SCAN_OK output=$diag_root"
