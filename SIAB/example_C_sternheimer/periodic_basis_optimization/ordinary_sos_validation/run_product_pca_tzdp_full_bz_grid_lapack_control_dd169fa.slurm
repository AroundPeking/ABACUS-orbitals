#!/bin/bash
#SBATCH --job-name=c24_tzdp_grid_lapack
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-tzdp-grid-lapack-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-tzdp-grid-lapack-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_TZDP_GRID_LAPACK_CONTROL_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source_work=$campaign/runs/product-pca-20260825/ordinary-sos-original-tzdp-grid-full-bz-body-nfreq6-ad29464fd
template=$source_work/librpa.in
librpa=/data/home/df_iopcas_ghj/app/oml-pinned/LibRPA-v0.7.0-dd169fa/build-df-oml/chi0_main.exe
expected_librpa_sha=7af5e426933283a2aa08b8771692bc770506ce3b6f00ccf8545108cb8e58d3bc
work=$campaign/runs/product-pca-20260825/ordinary-sos-original-tzdp-grid-full-bz-lapack-nfreq6-dd169fa

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$librpa"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -s "$template"
test -d "$source_work/reader_v1"
grep -Eq '^use_scalapack_ecrpa[[:space:]]*=[[:space:]]*t$' "$template"
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_grid_iq_$' "$template"

mkdir -p "$work"
sed 's/^use_scalapack_ecrpa[[:space:]]*=.*$/use_scalapack_ecrpa = f/' \
  "$template" > "$work/librpa.in"
grep -Eq '^use_scalapack_ecrpa[[:space:]]*=[[:space:]]*f$' "$work/librpa.in"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_original_tzdp_grid_coulomb_lapack_logdet_control
  echo source_work=$source_work
  echo abacus_commit=ad29464fdc9125114cc052840415647a1b5a6ef1
  echo librpa_commit=dd169fa11fa920d580d4f39dc11e218a7f17f7b5
  echo orbital_basis=C_TZDP_10au_3s3p2d
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo nfreq=6
  echo kmesh=4,4,4
  echo symmetry_rpa=false
  echo full_q_count=64
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo determinant_route=single_rank_lapack
  echo analytic_headwing=no_body_screen
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$template" "$work/librpa.in" "$source_work/INPUT_SHA256.txt" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1

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
assert len(contributions) == 64, len(contributions)
maximum_imaginary = max(abs(value[1]) for value in contributions)
assert maximum_imaginary <= 1.0e-10, maximum_imaginary
assert abs(total - (-0.496700837)) <= 1.0e-6, total
print("status success")
print("scope body_only_no_analytic_headwing")
print("coulomb_kernel exact_rhs_full_periodic_poisson")
print("determinant_route single_rank_lapack")
print("qpoint_count 64")
print("maximum_q_imaginary_ha {:.16e}".format(maximum_imaginary))
print("ecrpa_ha {:.12f}".format(total))
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt
} >> provenance.txt

echo "C24_TZDP_GRID_LAPACK_CONTROL_OK work=$work"
