#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
consumer=$root/run_product_pca_tzdp_full_bz_grid_body_ad29464fd.slurm
input=$root/librpa_tzdp_grid_full_bz_nfreq6.in

test -s "$consumer"
test -s "$input"

grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_grid_iq_$' "$input"
grep -Eq '^use_symmetry_rpa[[:space:]]*=[[:space:]]*f$' "$input"
grep -Eq '^replace_w_head[[:space:]]*=[[:space:]]*f$' "$input"
grep -Fq 'grid-coulomb-original-tzdp-full-bz-ad29464fd' "$consumer"
grep -Fq 'test "$grid_coulomb_file_count" -eq 64' "$consumer"
grep -Fq 'assert counts == [64, 64]' "$consumer"
grep -Fq 'assert len(contributions) == 64' "$consumer"
grep -Fq 'maximum_imaginary <= 1.0e-10' "$consumer"
grep -Fq 'qpoint_count 64' "$consumer"
grep -Fq '7af5e426933283a2aa08b8771692bc770506ce3b6f00ccf8545108cb8e58d3bc' "$consumer"
grep -Fq 'grid_coulomb_producer_job=${GRID_COULOMB_ARRAY_JOB_ID' "$consumer"

echo "C_PRODUCT_PCA_TZDP_FULL_BZ_RPA_CONTRACT_OK"
