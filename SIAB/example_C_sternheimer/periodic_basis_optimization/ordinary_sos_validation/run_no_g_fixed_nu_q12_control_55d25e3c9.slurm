#!/bin/bash
#SBATCH --job-name=c_nog_fixnu_q
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=1-2
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nog-fixnu-q-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nog-fixnu-q-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C_NO_G_FIXED_NU_Q12_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-ce2801f3
candidate_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-g-candidate-570a6d79
run_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-g-fixed-nu-q12-control
base=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_threshold_candidate_solid_qstar_55d25e3c9.slurm
task=${SLURM_ARRAY_TASK_ID:?missing array task}
patched=$run_root/patched-q${task}.sh

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test "$task" -ge 1 && test "$task" -le 2
test -s "$base"
test -s "$candidate_root/TRUNCATION.json"
mkdir -p "$run_root"
test ! -e "$patched"

sed -E \
  -e 's/q_indices=\(1 2 3 6 7 8 11 28\)/q_indices=(1 2)/' \
  -e 's/test "\$task" -ge 1 && test "\$task" -le 8/test "$task" -ge 1 \&\& test "$task" -le 2/' \
  -e 's/remove_input_key rpa_pca_fixed_nu/set_input_key rpa_pca_fixed_nu 2,2,1,0/' \
  -e 's/! awk '\''\$1 == "rpa_pca_fixed_nu" \{found=1\} END \{exit !found\}'\'' INPUT/grep -Eq '\''^rpa_pca_fixed_nu[[:space:]]+2,2,1,0$'\'' INPUT/' \
  -e 's/purpose=threshold_only_product_pca_c_solid_eight_qstar/purpose=no_g_fixed_nu_product_pca_q1_q2_control/' \
  -e 's/auxiliary_basis_rule=product_pca_threshold_only/auxiliary_basis_rule=product_pca_fixed_nu_2_2_1_0/' \
  "$base" > "$patched"

grep -q 'q_indices=(1 2)' "$patched"
grep -q 'set_input_key rpa_pca_fixed_nu 2,2,1,0' "$patched"
grep -q 'purpose=no_g_fixed_nu_product_pca_q1_q2_control' "$patched"
grep -q 'auxiliary_basis_rule=product_pca_fixed_nu_2_2_1_0' "$patched"

env CANDIDATE_ROOT="$candidate_root" RUN_ROOT="$run_root" bash "$patched"

q_index=$task
work=$run_root/solid-grid/q${q_index}
grep -qx 'success' "$work/STATUS"
grep -Eq '^rpa_pca_fixed_nu[[:space:]]+2,2,1,0$' "$work/INPUT"
grep -q 'Response-aware RPA product PCA fixed radial profiles: 2,2,1,0' "$work/abacus.out"
{
  echo fixed_nu_control_wrapper_status=success
  echo fixed_nu_control_base_source_commit="$(cat "$source_root/DEPLOYED_COMMIT")"
  sha256sum "$0" "$base" "$patched"
} >> "$work/provenance.txt"

echo "C_NO_G_FIXED_NU_Q12_OK q=$q_index work=$work"
