#!/bin/bash
#SBATCH --job-name=c24_tzdp_64q_lriv
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-tzdp-64q-lriv-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-tzdp-64q-lriv-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_TZDP_FULL_BZ_LRI_CONTROL_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
grid_root=$campaign/runs/product-pca-20260825/grid-coulomb-original-tzdp-full-bz-ad29464fd
grid_work=$campaign/runs/product-pca-20260825/ordinary-sos-original-tzdp-grid-full-bz-body-nfreq6-ad29464fd
template=$campaign/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/librpa_tzdp_grid_full_bz_nfreq6.in
librpa=/data/home/df_iopcas_ghj/app/oml-pinned/LibRPA-v0.7.0-dd169fa/build-df-oml/chi0_main.exe
expected_librpa_sha=7af5e426933283a2aa08b8771692bc770506ce3b6f00ccf8545108cb8e58d3bc
work=$campaign/runs/product-pca-20260825/ordinary-sos-original-tzdp-lri-full-bz-body-nfreq6-dd169fa
reader=$work/reader_v1

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$librpa"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -s "$template"
test -d "$grid_work/reader_v1"

mkdir -p "$reader"
q1=$grid_root/q1
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out; do
  test -s "$q1/$name"
  ln -s "$q1/$name" "$reader/$name"
done
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(find "$q1" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c -print0 | sort -z)

for ordinal in $(seq 1 64); do
  matrix=$q1/v1_coulomb_full_iq_${ordinal}_rank0.dat
  test -s "$matrix"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done

full_coulomb_file_count=$(find -L "$reader" -maxdepth 1 -type f -name 'v1_coulomb_full_iq_*_rank*.dat' -size +0c | wc -l)
cs_file_count=$(find -L "$reader" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)
test "$full_coulomb_file_count" -eq 64
test "$cs_file_count" -eq 1

sed -e "s|__INPUT_DIR__|$reader|" \
    -e 's|^prefix_coul_full[[:space:]]*=.*$|prefix_coul_full = v1_coulomb_full_iq_|' \
    "$template" > "$work/librpa.in"
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_full_iq_$' "$work/librpa.in"
find -L "$reader" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work/INPUT_SHA256.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_original_tzdp_all_band_sos_lri_full_coulomb_control
  echo abacus_commit=ad29464fdc9125114cc052840415647a1b5a6ef1
  echo librpa_commit=dd169fa11fa920d580d4f39dc11e218a7f17f7b5
  echo orbital_basis=C_TZDP_10au_3s3p2d
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo nfreq=6
  echo kmesh=4,4,4
  echo symmetry_rpa=false
  echo full_q_count=64
  echo coulomb_kernel=abacus_lri_full_ewald
  echo analytic_headwing=no_body_screen
  echo full_coulomb_file_count=$full_coulomb_file_count
  echo cs_file_count=$cs_file_count
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$template" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$librpa"
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
assert abs(total) < 2.0, total
print("status success")
print("scope body_only_no_analytic_headwing")
print("coulomb_kernel abacus_lri_full_ewald")
print("qpoint_count 64")
print("maximum_q_imaginary_ha {:.16e}".format(maximum_imaginary))
print("ecrpa_ha {:.12f}".format(total))
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt
} >> provenance.txt

echo "C24_TZDP_FULL_BZ_LRI_CONTROL_OK work=$work"
