#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
script=$root/run_candidate_two_q_tail_gate_df.slurm

test -s "$script"
grep -Fq '#SBATCH --partition=p1,48cp2,normal' "$script"
grep -qx '#SBATCH --nodes=1' "$script"
grep -qx '#SBATCH --ntasks-per-node=1' "$script"
grep -qx '#SBATCH --cpus-per-task=40' "$script"
grep -qx '#SBATCH --mem=190000' "$script"
grep -Fq 'q_indices=(2 6)' "$script"
grep -Fq 'expected_librpa_sha=de334e398cccd3343da3003bd3c654474faabc467e72ba321ff3d215e4e300e4' "$script"
grep -Fq 'parent_commit=d4810f73aab20c36e69b1c353c945b77f40931c9' "$script"
grep -Fq 'n_bands_chi0 = -1' "$script"
grep -Fq 'debug = t' "$script"
grep -Fq -- '--qstar-indices 2,6' "$script"
grep -Fq "grep -c '^RPA normal split ifreq='" "$script"
grep -Fq "grep -c '^RPA freqdiag ifreq='" "$script"
grep -Fq 'nonzero_record_count=12' "$script"
grep -Fq '0.010207674744' "$script"
grep -Fq '0.021127738990' "$script"
grep -Fq '0.010130257316' "$script"
grep -Fq '0.021125217918' "$script"
grep -Fq 'printf '\''rejected\n'\'' > STATUS' "$script"
grep -Fq 'printf '\''success\n'\'' > STATUS' "$script"
grep -Fq 'python=${C_BASIS_PYTHON:-' "$script"
grep -Fq 'librpa=${C_BASIS_LIBRPA:-' "$script"
grep -Fq 'expected_cpus=${C_BASIS_CPUS_PER_TASK:-40}' "$script"
grep -Fq 'C_BASIS_ENV_SCRIPT' "$script"
grep -Fq 'p1|48cp2|normal)' "$script"
! grep -Fq 'q_indices=(1 2 3 6 7 8 11 28)' "$script"

echo CANDIDATE_TWO_Q_TAIL_GATE_CONTRACT_OK
