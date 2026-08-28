#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
q1=$root/run_selected_candidate_response_pca_matched_delta_q1_55d25e3c9.slurm
reader=$root/run_selected_candidate_response_pca_matched_delta_reader_d4810f73.slurm
release=$root/release_selected_candidate_response_pca_matched_delta_after_q1.sh

test -s "$q1"
test -s "$reader"
test -s "$release"

grep -q '^#SBATCH --array=1$' "$q1"
grep -q '^#SBATCH --nodes=48$' "$q1"
grep -q '^#SBATCH --ntasks=48$' "$q1"
grep -q '^#SBATCH --cpus-per-task=40$' "$q1"
grep -q '^abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5$' "$q1"
grep -q '^expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9$' "$q1"
grep -q '^fixed_nu=2,2,1,0,0$' "$q1"
grep -q '^expected_naux=522$' "$q1"
grep -q 'set_input_key rpa_pca_fixed_nu "$fixed_nu"' "$q1"
grep -q 'set_input_key sternheimer_delta 1' "$q1"
grep -q 'set_input_key sternheimer_fd_order 8' "$q1"
grep -q 'set_input_key sternheimer_frequency_grid_file "$frequency_name"' "$q1"
grep -q 'grep -qx "sternheimer_q_index $iq" "$report"' "$q1"
grep -q 'if test "$iq" = 1; then grid_q=$grid_q1; else grid_q=$grid_rest/q${iq}; fi' "$q1"
grep -q 'full_periodic_poisson' "$q1"
grep -q 'POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt' "$q1"
grep -q 'assert header\["naux"\] == expected_naux == channels' "$q1"
! grep -q 'ad29464fd' "$q1"
! grep -q 'grid-coulomb-selected-fixed-prefix-full-bz' "$q1"

grep -q '^expected_naux=522$' "$reader"
grep -q 'POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt' "$reader"
grep -q 'for ordinal in $(seq 1 64)' "$reader"
grep -q 'assert dimensions == {expected_naux}' "$reader"
grep -q 'replace_w_head = false' "$reader"
grep -q 'scope body_only_no_analytic_headwing' "$reader"
! grep -q 'ad29464fd' "$reader"

grep -q 'runtime_gate_hours=18' "$release"
grep -q 'all_converged yes' "$release"
grep -q 'canonical_q_indices' "$release"
grep -q 'POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt' "$release"
! grep -q 'SELECTED_SOS_JOB_ID' "$release"

echo "C_RESPONSE_PCA_MATCHED_DELTA_CONTRACT_OK"
