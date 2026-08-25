#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
consumer=$root/run_product_pca_tzdp_full_bz_patched_logdet_e132bb9a.slurm
lapack_control=$root/run_product_pca_tzdp_full_bz_patched_lapack_control_e132bb9a.slurm

test -s "$consumer"
test -s "$lapack_control"
grep -Fq 'e132bb9a2acbbc558011329150b90e457d707d46' "$consumer"
grep -Fq 'LibRPA-rpa-logdet-bz-exact-test1/build-df-test' "$consumer"
grep -Fq 'librpa=$librpa_root/chi0_main.exe' "$consumer"
grep -Fq 'ordinary-sos-original-tzdp-grid-full-bz-body-nfreq6-ad29464fd' "$consumer"
grep -Fq "use_scalapack_ecrpa[[:space:]]*=[[:space:]]*t" "$consumer"
grep -Fq 'assert len(contributions) == 64' "$consumer"
grep -Fq 'maximum_imaginary <= 1.0e-10' "$consumer"
grep -Fq 'abs(total - (-0.496700837)) <= 1.0e-6' "$consumer"
grep -Fq 'determinant_route=scalapack_pivot_parity_fixed' "$consumer"

grep -Fq 'e132bb9a2acbbc558011329150b90e457d707d46' "$lapack_control"
grep -Fq 'LibRPA-rpa-logdet-bz-exact-test1/build-df-test' "$lapack_control"
grep -Fq "use_scalapack_ecrpa[[:space:]]*=[[:space:]]*f" "$lapack_control"
grep -Fq 'determinant_route=single_rank_lapack_exact_bvk' "$lapack_control"
grep -Fq 'reference_scalapack_total=-0.501460253' "$lapack_control"

echo C_PRODUCT_PCA_TZDP_FULL_BZ_PATCHED_LOGDET_CONTRACT_OK
