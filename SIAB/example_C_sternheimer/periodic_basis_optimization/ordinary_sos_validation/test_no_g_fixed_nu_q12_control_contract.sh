#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
producer=$root/run_no_g_fixed_nu_q12_control_55d25e3c9.slurm
sos=$root/run_no_g_fixed_nu_q12_sos_d4810f73.slurm

for path in "$producer" "$sos"; do test -s "$path"; done
grep -qx '#SBATCH --array=1-2' "$producer"
grep -q 'q_indices=(1 2)' "$producer"
grep -q 'rpa_pca_fixed_nu 2,2,1,0' "$producer"
grep -q 'fixed radial profiles: 2,2,1,0' "$producer"
grep -q 'changed_variable=product_pca_fixed_radial_profiles' "$sos"
grep -q 'n_bands_chi0 = -1' "$sos"
grep -q 'libRPA finished successfully' "$sos"
grep -q 'gamma_contribution_ha' "$sos"
grep -q 'q2_contribution_ha' "$sos"

echo NO_G_FIXED_NU_Q12_CONTROL_CONTRACT_OK
