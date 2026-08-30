#!/bin/bash
#SBATCH --job-name=c_nof_thr_q
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=1-2
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-thr-q-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-thr-q-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C_NO_F_THRESHOLD_Q12_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-ce2801f3
candidate_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-candidate-570a6d79
run_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-threshold-q12-control
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
  -e 's/payload\["output_nu"\] == \[3, 3, 2, 1\]/payload["output_nu"] == [3, 3, 2]/' \
  -e 's/payload\["output_nao"\] == 29/payload["output_nao"] == 22/' \
  -e 's/set_input_key nbands 58/set_input_key nbands 44/' \
  -e 's/purpose=threshold_only_product_pca_c_solid_eight_qstar/purpose=no_f_threshold_product_pca_q1_q2_control/' \
  -e 's/orbital_layout=3s3p2d1f/orbital_layout=3s3p2d/' \
  "$base" > "$patched"

grep -q 'q_indices=(1 2)' "$patched"
grep -q 'payload\["output_nu"\] == \[3, 3, 2\]' "$patched"
grep -q 'payload\["output_nao"\] == 22' "$patched"
grep -q 'set_input_key nbands 44' "$patched"
grep -q 'purpose=no_f_threshold_product_pca_q1_q2_control' "$patched"
grep -q 'orbital_layout=3s3p2d' "$patched"

env CANDIDATE_ROOT="$candidate_root" RUN_ROOT="$run_root" bash "$patched"

q_index=$task
work=$run_root/solid-grid/q${q_index}
grep -qx 'success' "$work/STATUS"
! awk '$1 == "rpa_pca_fixed_nu" {found=1} END {exit !found}' "$work/INPUT"
grep -Eq '^exx_pca_threshold[[:space:]]+1e-4$' "$work/INPUT"
{
  echo no_f_threshold_control_wrapper_status=success
  sha256sum "$0" "$base" "$patched"
} >> "$work/provenance.txt"

echo "C_NO_F_THRESHOLD_Q12_OK q=$q_index work=$work"
