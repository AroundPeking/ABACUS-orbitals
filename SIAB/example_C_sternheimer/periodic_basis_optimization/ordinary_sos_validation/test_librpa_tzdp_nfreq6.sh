#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
input=$root/librpa_tzdp_nfreq6.in

grep -Eq '^version_coul_reader[[:space:]]*=[[:space:]]*1$' "$input"
grep -Eq '^version_lri_reader[[:space:]]*=[[:space:]]*1$' "$input"
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_full_iq_$' "$input"
grep -Eq '^prefix_lri_coeff[[:space:]]*=[[:space:]]*v1_Cs_data_$' "$input"
grep -Eq '^use_symmetry_exx[[:space:]]*=[[:space:]]*t$' "$input"
grep -Eq '^use_symmetry_rpa[[:space:]]*=[[:space:]]*t$' "$input"
grep -Eq '^use_symmetry_gw[[:space:]]*=[[:space:]]*f$' "$input"
! grep -Eq '^use_abacus_(exx|rpa|gw)_symmetry[[:space:]]*=' "$input"

echo "C_PRODUCT_PCA_LIBRPA_V1_TEMPLATE_OK"
