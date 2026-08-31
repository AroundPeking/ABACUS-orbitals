#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
script=$root/run_candidate_qstar_freqdiag_replay_df.slurm

test -s "$script"
grep -qx '#SBATCH --partition=48cp2,p1' "$script"
grep -qx '#SBATCH --nodes=1' "$script"
grep -qx '#SBATCH --ntasks-per-node=1' "$script"
grep -qx '#SBATCH --cpus-per-task=40' "$script"
grep -qx '#SBATCH --mem=190000' "$script"
grep -qx '#SBATCH --time=01:00:00' "$script"
grep -Fq 'case "${SLURM_JOB_PARTITION:?}" in' "$script"
grep -Fq 'p1|48cp2)' "$script"
grep -q 'source_root=${SOURCE_ROOT:?' "$script"
grep -q 'run_root=${RUN_ROOT:?' "$script"
grep -q 'expected_ecrpa=${EXPECTED_ECRPA:?' "$script"
grep -q 'source_commit=${SIAB_SOURCE_COMMIT:?' "$script"
grep -q 'parent_commit=d4810f73aab20c36e69b1c353c945b77f40931c9' "$script"
grep -q 'expected_librpa_sha=de334e398cccd3343da3003bd3c654474faabc467e72ba321ff3d215e4e300e4' "$script"
grep -q 'n_bands_chi0[[:space:]]*=[[:space:]]*-1' "$script"
grep -q "grep -c '^RPA normal split ifreq='" "$script"
grep -q "grep -c '^RPA freqdiag ifreq='" "$script"
grep -q 'nonzero_qstar_freqdiag_records' "$script"
grep -q 'delta<=5e-10' "$script"
grep -q 'status=success' "$script"

echo CANDIDATE_QSTAR_FREQDIAG_REPLAY_CONTRACT_OK
