#!/bin/bash
#SBATCH -J h2_4s3p3d_sos
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH -t 1-00:00:00
#SBATCH --array=0-5
#SBATCH -o sos.%A_%a.out
#SBATCH -e sos.%A_%a.err

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

case "${SLURM_ARRAY_TASK_ID:?}" in
  0)
    lane=regenerated_4s3p3d
    case_name=H2
    orbital=H_gga_8au_100Ry_4s3p3d.orb
    orbital_sha=860f3a23b2d81486a62b48c80992a5610194899e049fe44dacb67b1ef983a7b9
    fixed_abs=0
    nbands=56
    nspin=1
    nelec=2
    output_dir=OUT.H2_SIAB_JOINT_4S3P3D_BOX20A_PCA1e4_RMESH5
    ;;
  1)
    lane=regenerated_4s3p3d
    case_name=H
    orbital=H_gga_8au_100Ry_4s3p3d.orb
    orbital_sha=860f3a23b2d81486a62b48c80992a5610194899e049fe44dacb67b1ef983a7b9
    fixed_abs=0
    nbands=28
    nspin=2
    nelec=1
    output_dir=OUT.H_SIAB_JOINT_4S3P3D_BOX20A_NFREQ16_PCA1e4_RMESH5
    ;;
  2)
    lane=fixed_4s3p
    case_name=H2
    orbital=H_gga_8au_100Ry_4s3p.orb
    orbital_sha=b394bb7329754e38341050ca4beb3b242b78e4be50c418b8764c98226bc8f033
    fixed_abs=1
    nbands=26
    nspin=1
    nelec=2
    output_dir=OUT.H2_SIAB_JOINT_4S3P_BOX20A_FIXED_ABS_RMESH5
    ;;
  3)
    lane=fixed_4s3p
    case_name=H
    orbital=H_gga_8au_100Ry_4s3p.orb
    orbital_sha=b394bb7329754e38341050ca4beb3b242b78e4be50c418b8764c98226bc8f033
    fixed_abs=1
    nbands=13
    nspin=2
    nelec=1
    output_dir=OUT.H_SIAB_JOINT_4S3P_BOX20A_NFREQ16_FIXED_ABS_RMESH5
    ;;
  4)
    lane=fixed_4s3p3d
    case_name=H2
    orbital=H_gga_8au_100Ry_4s3p3d.orb
    orbital_sha=860f3a23b2d81486a62b48c80992a5610194899e049fe44dacb67b1ef983a7b9
    fixed_abs=1
    nbands=56
    nspin=1
    nelec=2
    output_dir=OUT.H2_SIAB_JOINT_4S3P3D_BOX20A_FIXED_ABS_RMESH5
    ;;
  5)
    lane=fixed_4s3p3d
    case_name=H
    orbital=H_gga_8au_100Ry_4s3p3d.orb
    orbital_sha=860f3a23b2d81486a62b48c80992a5610194899e049fe44dacb67b1ef983a7b9
    fixed_abs=1
    nbands=28
    nspin=2
    nelec=1
    output_dir=OUT.H_SIAB_JOINT_4S3P3D_BOX20A_NFREQ16_FIXED_ABS_RMESH5
    ;;
  *) echo "invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

script_dir=${SLURM_SUBMIT_DIR:?}
case_dir="$script_dir/cases/$lane/$case_name"
pseudo=H_ONCV_PBE-1.0.upf
abs=H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs
abacus=/public/home/ghj/app/src/abacus-sternheimer-siab-task4-80a606f57a26/build-task4-o3-debug-info/abacus-80a606f57a26
librpa=/public/home/ghj/app/src/LibRPA-sternheimer-rpa-20260707/build-sternheimer-fast/chi0_main.exe

check_hash() {
  local expected=$1
  local path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

check_hash "$orbital_sha" "$script_dir/$orbital"
check_hash 3f5deaaf57ff09d87da2c2c186eb1654d174fe385c9edc265036b3c8869286ef \
  "$script_dir/$pseudo"
check_hash 2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d \
  "$abacus"
check_hash defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c \
  "$librpa"
if (( fixed_abs )); then
  check_hash d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f \
    "$script_dir/$abs"
fi

test -s "$script_dir/SOURCE_COMMIT"
test -d "$case_dir"
test ! -e "$case_dir/$output_dir"
test ! -e "$case_dir/basis_out"
test ! -e "$case_dir/basis_wfc_out"
test ! -e "$case_dir/basis_aux_out"
test ! -e "$case_dir/sos_full_nfreq16_dump"

cp "$script_dir/$orbital" "$case_dir/$orbital"
cp "$script_dir/$pseudo" "$case_dir/$pseudo"
if (( fixed_abs )); then
  cp "$script_dir/$abs" "$case_dir/$abs"
fi
cd "$case_dir"

export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export MKL_DYNAMIC=FALSE

echo "date_start=$(date +%F_%T)"
echo "lane=$lane"
echo "case=$case_name"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "source_commit=$(cat "$script_dir/SOURCE_COMMIT")"
echo "abacus_source_commit=80a606f57a2610bc2532468661b687b01f58074c"
echo "omp_threads=$OMP_NUM_THREADS"
sha256sum INPUT STRU KPT librpa.in "$orbital" "$pseudo"
if (( fixed_abs )); then
  sha256sum "$abs"
fi

/usr/bin/time -v mpirun -np 1 -ppn 1 "$abacus" \
  > "abacus.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

grep -q '!FINAL_ETOT_IS' "$output_dir/running_scf.log"
grep -q 'rpa_lcao_exx(Ha):' \
  "abacus.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
test -s band_out
test -s basis_wfc_out
test -s basis_aux_out
test -s v1_coulomb_full_iq_1_rank0.dat
test -s v1_Cs_data_0.txt
test -s KS_eigenvector_0.dat

awk -v expected_bands="$nbands" -v expected_spins="$nspin" \
    -v expected_electrons="$nelec" '
  NR == 2 && $1 != expected_spins { exit 2 }
  NR == 3 && $1 != expected_bands { exit 3 }
  NR >= 7 && NF >= 4 {
    rows += 1
    occ = $2 + 0.0
    occ_sum += occ
    if (occ < -1.0e-8 || occ > 2.0 + 1.0e-8) { exit 4 }
    nearest = int(occ + 0.5)
    diff = occ - nearest
    if (diff < 0) { diff = -diff }
    if (diff > 1.0e-8) { exit 5 }
  }
  END {
    if (rows != expected_bands * expected_spins) { exit 6 }
    diff = occ_sum - expected_electrons
    if (diff < 0) { diff = -diff }
    if (diff > 1.0e-8) { exit 7 }
  }
' band_out

/usr/bin/time -v mpirun -np 1 -ppn 1 "$librpa" \
  > "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

grep -q 'libRPA finished successfully' \
  "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
grep -q 'Total EcRPA:' \
  "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
sha256sum band_out basis_wfc_out basis_aux_out \
  v1_coulomb_full_iq_1_rank0.dat v1_Cs_data_0.txt KS_eigenvector_0.dat
echo "date_end=$(date +%F_%T)"
