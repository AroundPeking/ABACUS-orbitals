#!/bin/bash
#SBATCH --job-name=c24_tzdp_e132_gridrpa
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-tzdp-e132-gridrpa-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-tzdp-e132-gridrpa-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_TZDP_PATCHED_LOGDET_RPA_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source_work=$campaign/runs/product-pca-20260825/ordinary-sos-original-tzdp-grid-full-bz-body-nfreq6-ad29464fd
template=$source_work/librpa.in
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-test1/build-df-test
librpa=$librpa_root/chi0_main.exe
build_provenance=$librpa_root/PROVENANCE-e132bb9a.txt
librpa_commit=e132bb9a2acbbc558011329150b90e457d707d46
work=$campaign/runs/product-pca-20260825/ordinary-sos-original-tzdp-grid-full-bz-patched-logdet-nfreq6-e132bb9a

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$librpa"
test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(awk -v exe="$librpa" '$2 == exe {print $1}' "$build_provenance")
test -n "$expected_librpa_sha"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -s "$template"
test -s "$source_work/INPUT_SHA256.txt"
test -d "$source_work/reader_v1"
grep -Eq '^use_scalapack_ecrpa[[:space:]]*=[[:space:]]*t$' "$template"
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_grid_iq_$' "$template"
grep -Eq '^use_symmetry_rpa[[:space:]]*=[[:space:]]*f$' "$template"
grep -Eq '^replace_w_head[[:space:]]*=[[:space:]]*f$' "$template"
test "$(find -L "$source_work/reader_v1" -maxdepth 1 -type f -name 'v1_coulomb_grid_iq_*_rank*.dat' -size +0c | wc -l)" -eq 64

mkdir -p "$work"
cp "$template" "$work/librpa.in"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_original_tzdp_grid_coulomb_scalapack_logdet_fix_control
  echo source_work=$source_work
  echo abacus_commit=ad29464fdc9125114cc052840415647a1b5a6ef1
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo orbital_basis=C_TZDP_10au_3s3p2d
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo nfreq=6
  echo kmesh=4,4,4
  echo symmetry_rpa=false
  echo full_q_count=64
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo determinant_route=scalapack_pivot_parity_fixed
  echo exact_bvk_snap=yes
  echo analytic_headwing=no_body_screen
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$template" "$source_work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
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
print("determinant_route scalapack_pivot_parity_fixed")
print("exact_bvk_snap yes")
print("qpoint_count 64")
print("maximum_q_imaginary_ha {:.16e}".format(maximum_imaginary))
print("q_real_sum_ha {:.12f}".format(sum(value[0] for value in contributions)))
print("ecrpa_ha {:.12f}".format(total))
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt
} >> provenance.txt

echo "C24_TZDP_PATCHED_LOGDET_RPA_OK work=$work"
