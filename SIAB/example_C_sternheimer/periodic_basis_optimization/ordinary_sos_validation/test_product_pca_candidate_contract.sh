#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
script=$root/run_product_pca_candidate_body_nfreq6_efc24c2c.slurm

grep -q '^abacus_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-bz-identity-7772406ac-20260825$' "$script"
grep -q '^abacus=.*artifacts/job3120556/abacus_3p$' "$script"
grep -q '^abacus_commit=7772406ac92925d410f38bf69142a98b5c23a62b$' "$script"
grep -q '^expected_abacus_sha=c6939a4ac6c990be0753dfb88c3239a680f8e8ef8bd2f593677d62b9f4bd1f61$' "$script"
grep -q 'echo abacus_build_job=3120556' "$script"
grep -q 'python3 - bz_sampling_out > BZ_SAMPLING_IDENTITY.txt' "$script"
grep -q 'assert labels == expected' "$script"

echo "C_PRODUCT_PCA_CANDIDATE_CONTRACT_OK"
