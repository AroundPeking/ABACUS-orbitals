#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
submitter=$root/submit_relaxed_dzp_pbe_qstar_chain.sh
pbe=$root/run_relaxed_dzp_pbe_gate_55d25e3c9.slurm
test -s "$submitter"
test -s "$pbe"
grep -q 'CANDIDATE.json' "$submitter"
grep -q 'run_relaxed_dzp_pbe_gate_55d25e3c9.slurm' "$submitter"
grep -q 'afterok:$pbe_job' "$submitter"
grep -q 'run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm' "$submitter"
grep -q 'run_threshold_candidate_solid_qstar_55d25e3c9.slurm' "$submitter"
grep -q 'run_threshold_candidate_solid_qstar_sos_d4810f73.slurm' "$submitter"
grep -q 'run_threshold_candidate_qstar_binding_collect.slurm' "$submitter"
grep -q 'test ! -e "$receipt"' "$submitter"
grep -q 'sbatch --test-only' "$submitter"
grep -q 'read_periodic_candidate_manifest.py' "$pbe"
grep -q 'read_periodic_candidate_manifest.py' "$submitter"
echo RELAXED_DZP_PBE_QSTAR_CHAIN_CONTRACT_OK
