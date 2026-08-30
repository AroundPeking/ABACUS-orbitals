#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
runner=$root/run_no_g_qstar_sqrt_coulomb_scan_d4810f73.slurm

test -s "$runner"
grep -qx '#SBATCH --partition=p1' "$runner"
grep -qx '#SBATCH --cpus-per-task=40' "$runner"
grep -qx '#SBATCH --mem=190000' "$runner"
grep -qx '#SBATCH --array=1-3' "$runner"
grep -q 'thresholds=("1.1e-5" "2e-5" "1e-4")' "$runner"
grep -q 'joint-atom-solid-no-g-threshold-qstar-sos-ce2801f3/solid-sos' "$runner"
grep -q 'changed_variable=sqrt_coulomb_threshold' "$runner"
grep -q 'n_bands_chi0.*-1' "$runner"
grep -q 'libRPA finished successfully' "$runner"
grep -q 'gamma_contribution_ha' "$runner"

echo NO_G_QSTAR_SQRT_COULOMB_SCAN_CONTRACT_OK
