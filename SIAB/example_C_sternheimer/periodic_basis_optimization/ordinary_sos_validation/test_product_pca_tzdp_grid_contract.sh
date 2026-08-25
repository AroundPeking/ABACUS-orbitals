#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
producer=$root/run_product_pca_tzdp_grid_coulomb_allq_ad29464fd.slurm
consumer=$root/run_product_pca_tzdp_grid_body_ad29464fd.slurm
input=$root/librpa_tzdp_grid_nfreq6.in

test -s "$producer"
test -s "$consumer"
test -s "$input"

grep -Eq '^#SBATCH --array=1-8$' "$producer"
grep -Fq 'full_q_indices=(1 22 43 6 27 23 11 55)' "$producer"
grep -Fq 'ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1' "$producer"
grep -Fq 'ad29464fdc9125114cc052840415647a1b5a6ef1' "$producer"
grep -Fq 'grid_coulomb_v1_librpa_iq' "$producer"
grep -Fq 'grid_coulomb_v1_full_q_index' "$producer"

grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_grid_iq_$' "$input"
grep -Eq '^replace_w_head[[:space:]]*=[[:space:]]*f$' "$input"
grep -Fq 'grid_coulomb_file_count' "$consumer"
grep -Fq 'expected_full_q_indices=(1 22 43 6 27 23 11 55)' "$consumer"
grep -Fq 'abs(total) < 2.0' "$consumer"

echo "C_PRODUCT_PCA_TZDP_GRID_COULOMB_CONTRACT_OK"
