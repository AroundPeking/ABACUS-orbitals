#!/bin/bash

set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
runner=$root/run_no_f_threshold_atom_restart_sos_55d25e3c9_d4810f73.slurm
test -s "$runner"
bash -n "$runner"
grep -q 'restart_source=.*OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA' "$runner"
grep -q 'set_input_key init_wfc file' "$runner"
grep -q 'set_input_key init_chg file' "$runner"
grep -q 'set_input_key mixing_beta 0.1' "$runner"
grep -q 'set_input_key mixing_beta_mag 0.1' "$runner"
grep -q 'set_input_key mixing_gg0 0.0' "$runner"
grep -q 'set_input_key mixing_gg0_mag 0.0' "$runner"
grep -q '#SCF IS CONVERGED#' "$runner"
echo NO_F_THRESHOLD_ATOM_RESTART_CONTRACT_OK
