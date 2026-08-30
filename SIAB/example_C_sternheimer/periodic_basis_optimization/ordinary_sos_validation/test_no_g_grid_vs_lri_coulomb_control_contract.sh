#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
runner=$root/run_no_g_grid_vs_lri_coulomb_control_d4810f73.slurm

test -s "$runner"
grep -qx '#SBATCH --partition=p1' "$runner"
grep -qx '#SBATCH --cpus-per-task=40' "$runner"
grep -qx '#SBATCH --mem=190000' "$runner"
grep -q 'changed_variable=coulomb_matrix_source' "$runner"
grep -q 'prefix_coul_full = v1_coulomb_full_iq_' "$runner"
grep -q 'comparison_kernel=exact_rhs_full_periodic_poisson' "$runner"
grep -q 'n_bands_chi0.*-1' "$runner"
grep -q 'libRPA finished successfully' "$runner"
grep -q 'gamma_contribution_ha' "$runner"

echo NO_G_GRID_VS_LRI_COULOMB_CONTROL_CONTRACT_OK
