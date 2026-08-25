#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
producer=$root/run_product_pca_tzdp_full_bz_grid_coulomb_ad29464fd.slurm

test -s "$producer"

grep -Eq '^#SBATCH --array=1-64%8$' "$producer"
grep -Fq 'set_input_key symmetry -1' "$producer"
grep -Fq 'set_input_key nbands 44' "$producer"
grep -Fq 'set_input_key sternheimer_q_index "$full_q_index"' "$producer"
grep -Fq 'full_q_index=$ordinal' "$producer"
grep -Fq 'ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1' "$producer"
grep -Fq 'ad29464fdc9125114cc052840415647a1b5a6ef1' "$producer"
grep -Fq 'assert nibz == 64' "$producer"
grep -Fq 'assert nfull == 64' "$producer"
grep -Fq 'assert iq == expected_iq' "$producer"
grep -Fq 'assert naux == 236' "$producer"
grep -Fq 'grid_coulomb_v1_librpa_iq $ordinal' "$producer"
grep -Fq 'grid_coulomb_v1_full_q_index $full_q_index' "$producer"

echo "C_PRODUCT_PCA_TZDP_FULL_BZ_GRID_COULOMB_CONTRACT_OK"
