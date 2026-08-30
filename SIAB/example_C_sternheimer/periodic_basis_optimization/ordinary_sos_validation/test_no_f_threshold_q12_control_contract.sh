#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
producer=$root/run_no_f_threshold_q12_control_55d25e3c9.slurm
sos=$root/run_no_f_threshold_q12_sos_d4810f73.slurm

for path in "$producer" "$sos"; do
  test -s "$path"
  bash -n "$path"
done
grep -qx '#SBATCH --array=1-2' "$producer"
grep -q 'output_nu.*3, 3, 2' "$producer"
grep -q 'output_nao.*22' "$producer"
grep -q 'set_input_key nbands 44' "$producer"
grep -q 'remove_f_orbital_with_threshold_only_product_pca' "$sos"
grep -q 'pca_rule=exx_pca_threshold_1e-4' "$sos"
grep -q 'run_no_g_fixed_nu_q12_sos_d4810f73.slurm' "$sos"

echo NO_F_THRESHOLD_Q12_CONTROL_CONTRACT_OK
