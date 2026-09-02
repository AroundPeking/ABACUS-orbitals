#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
atom=$root/run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm
solid_q=$root/run_threshold_candidate_solid_qstar_55d25e3c9.slurm
solid_sos=$root/run_threshold_candidate_solid_qstar_sos_d4810f73.slurm
collector=$root/run_threshold_candidate_qstar_binding_collect.slurm
submitter=$root/submit_threshold_candidate_qstar_chain.sh
recovery=$root/run_threshold_candidate_solid_qstar_recovery.slurm

for path in "$atom" "$solid_q" "$solid_sos" "$collector" "$submitter" "$recovery"; do
  test -s "$path"
done

for job in "$atom" "$solid_q" "$solid_sos"; do
  grep -Fq 'case "${SLURM_JOB_PARTITION:?}" in' "$job"
  ! grep -Fq 'test "${SLURM_JOB_PARTITION:?}" = p1' "$job"
done

grep -Fq 'p1|48cp2|normal)' "$solid_q"

grep -q 'read_periodic_candidate_manifest.py' "$atom"
grep -q 'set_input_key nbands "$nbands"' "$atom"
grep -q 'remove_input_key rpa_pca_fixed_nu' "$atom"
grep -q 'set_input_key exx_pca_threshold 1e-4' "$atom"
grep -q 'n_bands_chi0 = -1' "$atom"
grep -q 'libRPA finished successfully' "$atom"
grep -q 'ATOM_SOS_ROOT' "$atom"
grep -q 'ATOM_RESTART_SOURCE' "$atom"
grep -q 'restart_input_snapshot' "$atom"
grep -q 'init_wfc file' "$atom"
grep -q 'init_chg file' "$atom"
grep -q 'restart_loaded yes' "$atom"

grep -qx '#SBATCH --array=1-8%8' "$solid_q"
grep -q 'q_indices=(1 2 3 6 7 8 11 28)' "$solid_q"
grep -q 'read_periodic_candidate_manifest.py' "$solid_q"
grep -q 'set_input_key nbands "$nbands"' "$solid_q"
grep -q 'remove_input_key rpa_pca_fixed_nu' "$solid_q"
grep -q 'set_input_key exx_pca_threshold 1e-4' "$solid_q"
grep -q 'exact_rhs_full_periodic_poisson' "$solid_q"
grep -Fq 'source=${C_SOLID_SOURCE_ROOT:-' "$solid_q"
grep -Fq 'abacus=${C_BASIS_ABACUS:-' "$solid_q"
grep -Fq 'python=${C_BASIS_PYTHON:-' "$solid_q"
grep -Fq 'expected_cpus=${C_BASIS_CPUS_PER_TASK:-40}' "$solid_q"
grep -Fq 'C_BASIS_ENV_SCRIPT' "$solid_q"

grep -q 'sparse_qstar_coulomb.py' "$solid_sos"
grep -q 'sparse_qstar_sos_gate.py' "$solid_sos"
grep -q 'n_bands_chi0 = -1' "$solid_sos"
grep -q 'libRPA finished successfully' "$solid_sos"
grep -q 'qstar_reconstruction_ha' "$solid_sos"
grep -q 'SIAB_SOURCE_COMMIT' "$solid_sos"
! grep -q 'source_root/.git/HEAD' "$solid_sos"

grep -q 'FAILED_SOLID_SOS_ROOT' "$recovery"
grep -q 'FAILED_SOLID_SOS_JOB_ID' "$recovery"
grep -q 'libRPA finished successfully' "$recovery"
grep -q 'qstar_reconstruction_ha' "$recovery"

grep -q '0.1' "$collector"
grep -q 'ATOM_SOS_ROOT' "$collector"
grep -q 'SOLID_SOS_ROOT' "$collector"
grep -q 'afterok' "$submitter"
grep -q 'test -z.*squeue' "$submitter"
grep -q 'SIAB_SOURCE_COMMIT' "$submitter"
grep -q 'controlled_lowest_f' "$root/read_periodic_candidate_manifest.py"

echo THRESHOLD_CANDIDATE_QSTAR_CHAIN_CONTRACT_OK
