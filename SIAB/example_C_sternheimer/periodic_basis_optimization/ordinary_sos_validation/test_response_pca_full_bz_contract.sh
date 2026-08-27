#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
producer=$root/run_selected_candidate_response_pca_full_bz_55d25e3c9.slurm
reader=$root/run_selected_candidate_response_pca_reader_d4810f73.slurm

test -s "$producer"
test -s "$reader"

grep -q '^#SBATCH --array=2-64%8$' "$producer"
grep -q '^abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5$' "$producer"
grep -q '^fixed_nu=2,2,1,0,0$' "$producer"
grep -q '^expected_naux=522$' "$producer"
grep -q 'set_input_key rpa_pca_fixed_nu "$fixed_nu"' "$producer"
grep -q 'test "$ordinal" -ge 2' "$producer"
grep -q 'test "$ordinal" -le 64' "$producer"
grep -q 'assert naux == expected_naux' "$producer"
! grep -q 'unified-basis-opt-grid-coulomb-ad29464fd' "$producer"

grep -q '^q1_job=3137632$' "$reader"
grep -q '^expected_naux=522$' "$reader"
grep -q 'for ordinal in $(seq 1 64)' "$reader"
grep -q 'test "$grid_coulomb_file_count" -eq 64' "$reader"
grep -q 'assert dimensions == {expected_naux}' "$reader"
grep -Eq "prefix_coul_full.*v1_coulomb_grid_iq_" "$reader"
grep -Eq "replace_w_head.*f" "$reader"
! grep -q 'grid-coulomb-selected-fixed-prefix-full-bz-ad29464fd' "$reader"

echo "C_RESPONSE_PCA_FULL_BZ_CONTRACT_OK"
