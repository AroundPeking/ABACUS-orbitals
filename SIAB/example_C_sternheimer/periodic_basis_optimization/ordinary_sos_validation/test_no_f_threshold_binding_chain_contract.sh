#!/bin/bash

set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
scripts=(
  "$root/run_no_f_threshold_qstar_remainder_55d25e3c9.slurm"
  "$root/run_no_f_threshold_atom_sos_55d25e3c9_d4810f73.slurm"
  "$root/run_no_f_threshold_qstar_sos_d4810f73.slurm"
)
for script in "${scripts[@]}"; do
  test -s "$script"
  bash -n "$script"
done
grep -qx '#SBATCH --array=1-6%6' "${scripts[0]}"
grep -q 'q_indices=(3 6 7 8 11 28)' "${scripts[0]}"
grep -q 'set_input_key nbands 44' "${scripts[0]}"
grep -q 'set_input_key nbands 22' "${scripts[1]}"
grep -q '3\*1 19\*0 1\*1 21\*0' "${scripts[1]}"
grep -q 'sparse_eight_qstar_all_band_sos' "${scripts[2]}"
echo NO_F_THRESHOLD_BINDING_CHAIN_CONTRACT_OK
