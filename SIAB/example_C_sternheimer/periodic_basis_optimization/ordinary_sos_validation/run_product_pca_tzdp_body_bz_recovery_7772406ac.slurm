#!/bin/bash
#SBATCH --job-name=c24_pca_tzdp_bzfix
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-tzdp-bzfix-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-tzdp-bzfix-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_PRODUCT_PCA_TZDP_BZ_RECOVERY_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

code=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1
dataset=$source/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
template=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/librpa_tzdp_nfreq6.in
repair_tool=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/repair_bz_sampling_identity.py
librpa=/data/home/df_iopcas_ghj/app/oml-pinned/LibRPA-v0.7.0-dd169fa/build-df-oml/chi0_main.exe
expected_librpa_sha=7af5e426933283a2aa08b8771692bc770506ce3b6f00ccf8545108cb8e58d3bc
expected_abacus_sha=5a7ab8f2fc14424c8b47c5a9cbe22d1784016140cc9fa302f7e032662eed58b2
expected_source_bz_sha=ceffcd167a01be0c0a02f35f0cccfd5c1ae720685ef7caead82bde17c7295827
bz_writer_fix_commit=7772406ac92925d410f38bf69142a98b5c23a62b
bz_writer_fix_build_job=3120556
bz_writer_fix_binary_sha=c6939a4ac6c990be0753dfb88c3239a680f8e8ef8bd2f593677d62b9f4bd1f61
work=$code/runs/product-pca-20260825/ordinary-sos-original-tzdp-body-nfreq6-bz-recovery-7772406ac
reader=$work/reader_v1

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$librpa"
test -x "$repair_tool"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test "$(sha256sum "$source/bz_sampling_out" | awk '{print $1}')" = "$expected_source_bz_sha"
test -s "$template"
grep -Eq '^version_coul_reader[[:space:]]*=[[:space:]]*1$' "$template"
grep -Eq '^version_lri_reader[[:space:]]*=[[:space:]]*1$' "$template"
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_full_iq_$' "$template"
grep -Eq '^prefix_lri_coeff[[:space:]]*=[[:space:]]*v1_Cs_data_$' "$template"
grep -Eq '^use_symmetry_rpa[[:space:]]*=[[:space:]]*t$' "$template"
! grep -Eq '^use_abacus_(exx|rpa|gw)_symmetry[[:space:]]*=' "$template"
test -s "$source/RUN_PROVENANCE.txt"
grep -qx 'status=success' "$source/RUN_PROVENANCE.txt"
grep -qx 'auxiliary_basis_source product_pca' "$dataset/manifest.dat"
grep -qx 'frequency_count 6' "$dataset/manifest.dat"
grep -qx 'kernel full_coulomb' "$dataset/manifest.dat"
grep -q "${expected_abacus_sha}[[:space:]].*abacus_3p" "$source/RUN_PROVENANCE.txt"
for path in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out \
  KS_eigenvector_0.dat stru_out; do
  test -s "$source/$path"
done
test "$(find "$source" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' | wc -l)" -eq 48
test "$(find "$source" -maxdepth 1 -type f -name 'v1_coulomb_full_iq_*_rank*.dat' | wc -l)" -eq 24

mkdir -p "$reader"
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(
  find "$source" -maxdepth 1 -type f \
    \( -name 'band_out' -o -name 'basis*_out' \
       -o -name 'KS_eigenvector_*.dat' -o -name 'stru_out' \
       -o -name 'v1_Cs_data_*.txt' -o -name 'v1_coulomb_full_iq_*_rank*.dat' \) \
    -print0 | sort -z
)
"$repair_tool" "$source/bz_sampling_out" "$reader/bz_sampling_out" \
  --report "$work/BZ_REPAIR.json" > "$work/BZ_REPAIR.stdout"
python3 - "$work/BZ_REPAIR.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="ascii"))
assert report["status"] == "success"
assert report["mesh"] == [4, 4, 4]
assert report["irreducible_count"] == 8
assert report["original_unique_reader_labels"] == 6
assert report["repaired_unique_reader_labels"] == 8
assert report["nonlabel_fields_unchanged"] is True
PY

sed "s|__INPUT_DIR__|$reader|" "$template" > "$work/librpa.in"
find -L "$reader" -maxdepth 1 -type f \
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
  echo purpose=periodic_c_original_tzdp_product_pca_all_band_sos_body_bz_metadata_recovery
  echo producer_job=${PRODUCER_JOB_ID:?missing producer job id}
  echo producer_abacus_commit=711af860c125b9757c344a1961b63524c550cfe4
  echo producer_abacus_sha=$expected_abacus_sha
  echo bz_writer_fix_commit=$bz_writer_fix_commit
  echo bz_writer_fix_build_job=$bz_writer_fix_build_job
  echo bz_writer_fix_binary_sha=$bz_writer_fix_binary_sha
  echo bz_recovery_scope=reader_labels_only_nonlabel_fields_unchanged
  echo source_bz_sha=$expected_source_bz_sha
  echo librpa_commit=dd169fa11fa920d580d4f39dc11e218a7f17f7b5
  echo librpa_sha=$expected_librpa_sha
  echo orbital_basis=C_TZDP_10au_3s3p2d
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo nfreq=6
  echo kmesh=4,4,4
  echo qpoints=8
  echo full_coulomb=yes
  echo analytic_headwing=no_body_screen
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  echo mkl_threads=$MKL_NUM_THREADS
  sha256sum "$0" "$template" "$repair_tool" "$work/librpa.in" \
    "$work/BZ_REPAIR.json" "$work/INPUT_SHA256.txt" "$librpa" \
    "$dataset/manifest.dat" "$dataset/status.dat"
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
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt BZ_REPAIR.json
} >> provenance.txt

echo "C24_PRODUCT_PCA_TZDP_BZ_RECOVERY_OK work=$work"
