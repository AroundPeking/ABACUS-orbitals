#!/bin/bash
#SBATCH --job-name=c24_pca_sos_g
#SBATCH --partition=p1
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=2,3
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-sos-g-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-sos-g-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C24_PRODUCT_PCA_CANDIDATE_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

code=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
abacus_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825
abacus=$abacus_root/artifacts/job3117942/abacus_3p
abacus_commit=711af860c125b9757c344a1961b63524c550cfe4
expected_abacus_sha=5a7ab8f2fc14424c8b47c5a9cbe22d1784016140cc9fa302f7e032662eed58b2
librpa=/data/home/df_iopcas_ghj/app/oml-pinned/LibRPA-v0.7.0-dd169fa/build-df-oml/chi0_main.exe
expected_librpa_sha=7af5e426933283a2aa08b8771692bc770506ce3b6f00ccf8545108cb8e58d3bc
librpa_template=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/librpa_tzdp_nfreq6.in
ng=${SLURM_ARRAY_TASK_ID:?missing g multiplicity}

case "$ng" in
  2) nbands=94 ;;
  3) nbands=112 ;;
  *) echo "unsupported g multiplicity: $ng" >&2; exit 2 ;;
esac

export_dir=$code/runs/product-pca-20260825/export-joint-${ng}g
orbital=$export_dir/C_gga_10au_100Ry_3s3p2d1f${ng}g_product_pca.orb
work=$code/runs/product-pca-20260825/ordinary-sos-joint-${ng}g-body-nfreq6
suffix=C_DIAMOND_PRODUCT_PCA_JOINT_${ng}G_SOS

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 8
test "$SLURM_NTASKS_PER_NODE" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$abacus"
test -x "$librpa"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -s "$orbital"
grep -qx 'status=success' "$export_dir/provenance.txt"
test -s "$librpa_template"

mkdir -p "$work"
cp -a "$abacus_root/gate-template-product-pca/." "$work/"
cp "$orbital" "$work/"
cd "$work"

set_input_key() {
  local key=$1 value=$2
  if awk -v key="$key" '$1 == key {found=1} END {exit !found}' INPUT; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]+.*|${key} ${value}|" INPUT
  else
    printf '%-36s %s\n' "$key" "$value" >> INPUT
  fi
}

sed -i -E 's|^[[:space:]]*2[[:space:]]+2[[:space:]]+2[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]*$|4 4 4 0 0 0|' KPT
sed -i -E "s|^C_gga_10au_100Ry_3s3p2d\.orb$|$(basename "$orbital")|" STRU
set_input_key suffix "$suffix"
set_input_key nbands "$nbands"
set_input_key ecutwfc 45
set_input_key nx 24
set_input_key ny 24
set_input_key nz 24
set_input_key kpar 8
set_input_key rpa 1
set_input_key out_librpa_reader_version 1
set_input_key out_sternheimer_librpa 0
set_input_key out_sternheimer_basis_opt 0
set_input_key exx_pca_threshold 1e-4
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 2

! grep -q '^ABFS_ORBITAL' STRU
test -z "$(find . -maxdepth 1 -type f -name '*.abfs' -print -quit)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export OMP_STACKSIZE=2G
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_product_pca_joint_candidate_all_band_sos_body
  echo candidate=joint_${ng}g
  echo abacus_commit=$abacus_commit
  echo abacus_build_job=3117942
  echo librpa_commit=dd169fa11fa920d580d4f39dc11e218a7f17f7b5
  echo optimizer_job=3118689_${ng}
  echo heldout_job=3118691
  echo export_job=${EXPORT_JOB_ID:-unknown}_${ng}
  echo nbands=$nbands
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo grid=24,24,24
  echo kmesh=4,4,4
  echo nfreq=6
  echo full_coulomb=yes
  echo analytic_headwing=no_body_screen
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$librpa" "$librpa_template" "$orbital" \
    "$export_dir/provenance.txt" INPUT STRU KPT C_ONCV_PBE-1.0.upf
} > provenance.txt

/usr/bin/time -v -o abacus.time \
  mpirun -np "$SLURM_NTASKS" -ppn "$SLURM_NTASKS_PER_NODE" \
  "$abacus" > abacus.out 2>&1

test -s "OUT.${suffix}/running_scf.log"
grep -q '!FINAL_ETOT_IS' "OUT.${suffix}/running_scf.log"
for path in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out \
  KS_eigenvector_0.dat stru_out; do
  test -s "$path"
done
cs_count=$(find . -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)
coulomb_count=$(find . -maxdepth 1 -type f -name 'v1_coulomb_full_iq_*_rank*.dat' -size +0c | wc -l)
test "$cs_count" -ge 8
test "$coulomb_count" -ge 8
test ! -e "OUT.${suffix}/STERNHEIMER_BASIS_OPT_V1"

sed "s|__INPUT_DIR__|$work|" "$librpa_template" > librpa.in
find . -maxdepth 1 -type f \
  \( -name 'band_out' -o -name 'basis*_out' -o -name 'bz_sampling_out' \
     -o -name 'KS_eigenvector_*.dat' -o -name 'stru_out' \
     -o -name 'v1_Cs_data_*.txt' -o -name 'v1_coulomb_full_iq_*_rank*.dat' \) \
  -print0 | sort -z | xargs -0 sha256sum > INPUT_SHA256.txt

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
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
  echo reader_cs_file_count=$cs_count
  echo reader_coulomb_file_count=$coulomb_count
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/abacus_wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/abacus_max_rss_kb=/p' abacus.time
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/librpa_wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/librpa_max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt
} >> provenance.txt

echo "C24_PRODUCT_PCA_CANDIDATE_SOS_OK g=$ng work=$work"
