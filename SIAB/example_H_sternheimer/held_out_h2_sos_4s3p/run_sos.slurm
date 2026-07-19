#!/bin/bash
#SBATCH -J h2_4s3p_sos
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH -t 1-00:00:00
#SBATCH --array=0-1
#SBATCH -o sos.%A_%a.out
#SBATCH -e sos.%A_%a.err

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

case "${SLURM_ARRAY_TASK_ID:?}" in
  0)
    case_name=H2
    output_dir=OUT.H2_SIAB_JOINT_4S3P_BOX20A_PCA1e4_RMESH5
    ;;
  1)
    case_name=H
    output_dir=OUT.H_SIAB_JOINT_4S3P_BOX20A_NFREQ16_PCA1e4_RMESH5
    ;;
  *) echo "invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

script_dir=${SLURM_SUBMIT_DIR:?}
case_dir="$script_dir/cases/$case_name"
orbital="$script_dir/H_gga_8au_100Ry_4s3p.orb"
pseudo="$script_dir/H_ONCV_PBE-1.0.upf"
abacus=/public/home/ghj/app/src/abacus-sternheimer-siab-task4-80a606f57a26/build-task4-o3-debug-info/abacus-80a606f57a26
librpa=/public/home/ghj/app/src/LibRPA-sternheimer-rpa-20260707/build-sternheimer-fast/chi0_main.exe

check_hash() {
  local expected=$1
  local path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

check_hash b394bb7329754e38341050ca4beb3b242b78e4be50c418b8764c98226bc8f033 \
  "$orbital"
check_hash 3f5deaaf57ff09d87da2c2c186eb1654d174fe385c9edc265036b3c8869286ef \
  "$pseudo"
check_hash 2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d \
  "$abacus"
check_hash defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c \
  "$librpa"
test -d "$case_dir"
test ! -e "$case_dir/$output_dir"
test ! -e "$case_dir/basis_out"
test ! -e "$case_dir/sos_full_nfreq16_dump"

cp "$orbital" "$case_dir/H_gga_8au_100Ry_4s3p.orb"
cp "$pseudo" "$case_dir/H_ONCV_PBE-1.0.upf"
cd "$case_dir"

export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export MKL_DYNAMIC=FALSE

echo "date_start=$(date +%F_%T)"
echo "case=$case_name"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "omp_threads=$OMP_NUM_THREADS"
sha256sum INPUT STRU KPT librpa.in H_gga_8au_100Ry_4s3p.orb \
  H_ONCV_PBE-1.0.upf

/usr/bin/time -v mpirun -np 1 -ppn 1 "$abacus" \
  > "abacus.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

grep -q '!FINAL_ETOT_IS' "$output_dir/running_scf.log"
grep -q 'rpa_lcao_exx(Ha):' \
  "abacus.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
test -s basis_wfc_out
test -s basis_aux_out
test -s v1_coulomb_full_iq_1_rank0.dat
test -s v1_Cs_data_0.txt
test -s KS_eigenvector_0.dat

/usr/bin/time -v mpirun -np 1 -ppn 1 "$librpa" \
  > "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

grep -q 'libRPA finished successfully' \
  "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
grep -q 'Total EcRPA:' \
  "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
sha256sum basis_wfc_out basis_aux_out v1_coulomb_full_iq_1_rank0.dat \
  v1_Cs_data_0.txt KS_eigenvector_0.dat
echo "date_end=$(date +%F_%T)"
