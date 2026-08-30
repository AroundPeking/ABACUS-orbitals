#!/bin/bash
#SBATCH --job-name=c_nof_thr_sos
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-thr-sos-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-thr-sos-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NO_F_THRESHOLD_Q12_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

source_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-7a1fa80f
campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
run_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-threshold-q12-control
base=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_no_g_fixed_nu_q12_sos_d4810f73.slurm
patched=$run_root/patched-sos.sh

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test -s "$base"
test ! -e "$patched"

sed -E \
  -e 's|joint-atom-solid-no-g-fixed-nu-q12-control|joint-atom-solid-no-f-threshold-q12-control|g' \
  -e 's/grep -Eq '\''\^rpa_pca_fixed_nu\[\[:space:\]\]\+2,2,1,0\$'\'' "\$qdir\/INPUT"/! awk '\''$1 == "rpa_pca_fixed_nu" {found=1} END {exit !found}'\'' "$qdir\/INPUT"/' \
  -e 's/purpose=no_g_fixed_nu_q1_q2_sparse_sos_control/purpose=no_f_threshold_q1_q2_sparse_sos_control/' \
  -e 's/changed_variable=product_pca_fixed_radial_profiles/changed_variable=remove_f_orbital_with_threshold_only_product_pca/' \
  -e 's/echo rpa_pca_fixed_nu=2,2,1,0/echo pca_rule=exx_pca_threshold_1e-4/' \
  -e 's/scope no_g_fixed_nu_q1_q2_sparse_sos_control/scope no_f_threshold_q1_q2_sparse_sos_control/' \
  -e 's/print\("rpa_pca_fixed_nu 2,2,1,0"\)/print("pca_rule exx_pca_threshold_1e-4")/' \
  "$base" > "$patched"

grep -q 'joint-atom-solid-no-f-threshold-q12-control' "$patched"
grep -q 'purpose=no_f_threshold_q1_q2_sparse_sos_control' "$patched"
grep -q 'changed_variable=remove_f_orbital_with_threshold_only_product_pca' "$patched"
grep -q 'pca_rule=exx_pca_threshold_1e-4' "$patched"
! grep -q 'rpa_pca_fixed_nu=2,2,1,0' "$patched"

env Q12_ARRAY_JOB_ID="${Q12_ARRAY_JOB_ID:?missing q1/q2 array job id}" bash "$patched"

echo "C_NO_F_THRESHOLD_Q12_SOS_OK work=$run_root/solid-sos"
