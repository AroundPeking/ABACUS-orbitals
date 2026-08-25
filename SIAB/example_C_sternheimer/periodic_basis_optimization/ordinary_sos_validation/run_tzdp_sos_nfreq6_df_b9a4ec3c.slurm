#!/bin/bash
#SBATCH --job-name=c24_tzdp_sos6
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-b9a4ec3c/logs/c24-tzdp-sos6-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-b9a4ec3c/logs/c24-tzdp-sos6-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_TZDP_SOS6_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

code=/data/home/df_iopcas_ghj/app/siab/periodic-c-b9a4ec3c
source=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/c-diamond-g24-fd8-nfreq6-q1-9ee4e4e0b-kpar8
template=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/librpa_tzdp_nfreq6.in
librpa=/data/home/df_iopcas_ghj/app/oml-pinned/LibRPA-v0.7.0-dd169fa/build-df-oml/chi0_main.exe
expected_librpa_sha=7af5e426933283a2aa08b8771692bc770506ce3b6f00ccf8545108cb8e58d3bc
expected_abacus_sha=67beeb675fd80ff21160d874352b2e5b075562b9db07df83a1a9c57675539b3a
work=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/sos-tzdp-nfreq6-q1producer-dd169fa

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$librpa"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -s "$template"
test -s "$source/RUN_PROVENANCE.txt"
grep -qx 'status=success' "$source/RUN_PROVENANCE.txt"
grep -q "${expected_abacus_sha}[[:space:]].*abacus_3p" "$source/RUN_PROVENANCE.txt"
for path in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out \
  KS_eigenvector_0.dat stru_out; do
  test -s "$source/$path"
done
test "$(find "$source" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' | wc -l)" -eq 48
test "$(find "$source" -maxdepth 1 -type f -name 'v1_coulomb_full_iq_*_rank*.dat' | wc -l)" -eq 24

mkdir -p "$work"
sed "s|__INPUT_DIR__|$source|" "$template" > "$work/librpa.in"
find "$source" -maxdepth 1 -type f \
  \( -name 'band_out' -o -name 'basis*_out' -o -name 'bz_sampling_out' \
     -o -name 'KS_eigenvector_*.dat' -o -name 'stru_out' \
     -o -name 'v1_Cs_data_*.txt' -o -name 'v1_coulomb_full_iq_*_rank*.dat' \) \
  -print0 | sort -z | xargs -0 sha256sum > "$work/INPUT_SHA256.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_original_tzdp_all_band_sos_body_baseline
  echo siab_code_commit=b9a4ec3c79536f66d82ce85d8642d46423e58ed7
  echo producer_job=3113577
  echo producer_abacus_commit=9ee4e4e0b72dd1c9441a688fb57712f714e33e5b
  echo producer_abacus_sha=$expected_abacus_sha
  echo librpa_commit=dd169fa11fa920d580d4f39dc11e218a7f17f7b5
  echo librpa_sha=$expected_librpa_sha
  echo orbital_basis=C_TZDP_10au_3s3p2d
  echo auxiliary_basis=C_10au_3s3p2d1f1g_1e-4
  echo nfreq=6
  echo kmesh=4,4,4
  echo qpoints=8
  echo full_coulomb=yes
  echo analytic_headwing=no_body_screen
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  echo mkl_threads=$MKL_NUM_THREADS
  echo mkl_dynamic=$MKL_DYNAMIC
  echo omp_max_active_levels=$OMP_MAX_ACTIVE_LEVELS
  sha256sum "$0" "$template" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time \
  mpirun -np 1 "$librpa" > librpa.out 2>&1

grep -q '^RPA correlation energy (Hartree)' librpa.out
grep -q '^| Total EcRPA:' librpa.out
python3 - librpa.out > RESULT_SUMMARY.txt <<'PY'
import math
import re
import sys

lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
contributions = []
total = None
inside = False
for line in lines:
    if line.strip() == "RPA correlation energy (Hartree)":
        inside = True
        continue
    if not inside:
        continue
    total_match = re.match(r"^\| Total EcRPA:\s+([-+0-9.eE]+)\s*$", line)
    if total_match:
        total = float(total_match.group(1))
        break
    contribution = re.search(r": \(([-+0-9.eE]+),([-+0-9.eE]+)\)\s*$", line)
    if contribution:
        contributions.append((float(contribution.group(1)), float(contribution.group(2))))

assert total is not None and math.isfinite(total)
assert len(contributions) == 8, len(contributions)
maximum_imaginary = max(abs(value[1]) for value in contributions)
assert maximum_imaginary <= 1.0e-10, maximum_imaginary
assert abs(total) < 2.0, total
print("status success")
print("scope body_only_no_analytic_headwing")
print("qpoint_count 8")
print("maximum_q_imaginary_ha {:.16e}".format(maximum_imaginary))
print("ecrpa_ha {:.12f}".format(total))
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt
} >> provenance.txt

echo "C24_TZDP_SOS6_OK work=$work"
