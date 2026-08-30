#!/bin/bash
#SBATCH --job-name=c_nof_qsos
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-qsos-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-qsos-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NO_F_QSTAR_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-ce2801f3
candidate_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-candidate-570a6d79
run_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-threshold-q12-control
base=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_threshold_candidate_solid_qstar_sos_d4810f73.slurm
patched=$run_root/patched-qstar-sos.sh

test -s "$base"
test ! -e "$patched"
sed -E \
  -e 's|work=\$run_root/solid-sos|work=$run_root/solid-qstar-sos|' \
  -e 's/payload\["output_nu"\] == \[3, 3, 2, 1\] and payload\["output_nao"\] == 29/payload["output_nu"] == [3, 3, 2] and payload["output_nao"] == 22/' \
  -e 's/purpose=threshold_candidate_c_solid_sparse_eight_qstar_all_band_sos/purpose=no_f_threshold_c_solid_sparse_eight_qstar_all_band_sos/' \
  "$base" > "$patched"

grep -q 'payload\["output_nu"\] == \[3, 3, 2\].*output_nao.*22' "$patched"
grep -q 'purpose=no_f_threshold_c_solid_sparse_eight_qstar_all_band_sos' "$patched"
grep -q 'work=\$run_root/solid-qstar-sos' "$patched"
env CANDIDATE_ROOT="$candidate_root" RUN_ROOT="$run_root" SIAB_SOURCE_ROOT="$source_root" QSTAR_ARRAY_JOB_ID="${QSTAR_ARRAY_JOB_ID:?missing remainder array job id}" bash "$patched"

echo "C_NO_F_QSTAR_SOS_OK"
