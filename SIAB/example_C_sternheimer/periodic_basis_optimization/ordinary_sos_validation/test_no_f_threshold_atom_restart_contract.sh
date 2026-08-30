#!/bin/bash

set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
runner=$root/run_no_f_threshold_atom_restart_sos_55d25e3c9_d4810f73.slurm
test -s "$runner"
bash -n "$runner"
grep -q 'restart_source=.*OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA' "$runner"
grep -q 'OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA/"\$name"' "$runner"
grep -q 'restart_input_snapshot/"\$name"' "$runner"
grep -q 'set_input_key init_wfc file' "$runner"
grep -q 'set_input_key init_chg file' "$runner"
! grep -Eq '^[[:space:]]*set_input_key mixing_' "$runner"
grep -q 'restart_change=reset_broyden_history_only' "$runner"
grep -q 'mixing_beta=0.3' "$runner"
grep -q 'CANDIDATE_ROOT=.*RUN_ROOT=.*SIAB_SOURCE_ROOT=' "$runner"
grep -q '#SCF IS CONVERGED#' "$runner"
grep -q 'Read NAO wave functions from' "$runner"
grep -q 'Read in electron density:' "$runner"
echo NO_F_THRESHOLD_ATOM_RESTART_CONTRACT_OK
