#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
script=$root/run_product_pca_tzdp_body_bz_recovery_7772406ac.slurm

grep -q '^#SBATCH --partition=p1$' "$script"
grep -q '^#SBATCH --cpus-per-task=40$' "$script"
grep -q '^#SBATCH --mem=190000$' "$script"
grep -q '^expected_source_bz_sha=ceffcd167a01be0c0a02f35f0cccfd5c1ae720685ef7caead82bde17c7295827$' "$script"
grep -q '^bz_writer_fix_commit=7772406ac92925d410f38bf69142a98b5c23a62b$' "$script"
grep -q '^bz_writer_fix_build_job=3120556$' "$script"
grep -q '^bz_writer_fix_binary_sha=c6939a4ac6c990be0753dfb88c3239a680f8e8ef8bd2f593677d62b9f4bd1f61$' "$script"
grep -q 'repair_bz_sampling_identity.py' "$script"
grep -q 'original_unique_reader_labels.*== 6' "$script"
grep -q 'repaired_unique_reader_labels.*== 8' "$script"
grep -q 'nonlabel_fields_unchanged.*is True' "$script"
grep -q 'ln -s.*reader' "$script"
grep -q 'sed.*__INPUT_DIR__.*reader' "$script"
! grep -q 'cp .*v1_Cs_data' "$script"

echo "C_PRODUCT_PCA_TZDP_BZ_RECOVERY_CONTRACT_OK"
