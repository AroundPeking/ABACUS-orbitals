#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
atom=$root/run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm
solid_q=$root/run_threshold_candidate_solid_qstar_55d25e3c9.slurm
solid_sos=$root/run_threshold_candidate_solid_qstar_sos_d4810f73.slurm
collector=$root/run_threshold_candidate_qstar_binding_collect.slurm
submitter=$root/submit_threshold_candidate_qstar_chain.sh

for path in "$atom" "$solid_q" "$solid_sos" "$collector" "$submitter"; do
  test -s "$path"
done

grep -q 'read_periodic_candidate_manifest.py' "$atom"
grep -q 'set_input_key nbands "$nbands"' "$atom"
grep -q 'remove_input_key rpa_pca_fixed_nu' "$atom"
grep -q 'set_input_key exx_pca_threshold 1e-4' "$atom"
grep -q 'n_bands_chi0 = -1' "$atom"
grep -q 'libRPA finished successfully' "$atom"

grep -qx '#SBATCH --array=1-8%8' "$solid_q"
grep -q 'q_indices=(1 2 3 6 7 8 11 28)' "$solid_q"
grep -q 'read_periodic_candidate_manifest.py' "$solid_q"
grep -q 'set_input_key nbands "$nbands"' "$solid_q"
grep -q 'remove_input_key rpa_pca_fixed_nu' "$solid_q"
grep -q 'set_input_key exx_pca_threshold 1e-4' "$solid_q"
grep -q 'exact_rhs_full_periodic_poisson' "$solid_q"

grep -q 'sparse_qstar_coulomb.py' "$solid_sos"
grep -q 'sparse_qstar_sos_gate.py' "$solid_sos"
grep -q 'n_bands_chi0 = -1' "$solid_sos"
grep -q 'libRPA finished successfully' "$solid_sos"
grep -q 'qstar_reconstruction_ha' "$solid_sos"

grep -q '0.1' "$collector"
grep -q 'afterok' "$submitter"
grep -q 'test -z.*squeue' "$submitter"

echo THRESHOLD_CANDIDATE_QSTAR_CHAIN_CONTRACT_OK
