#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
job=$root/run_controlled_g_candidate_gate_df.slurm

test -s "$job"
grep -Fq '#SBATCH --partition=48cp2,p1' "$job"
grep -Fq 'prepare_controlled_g_candidate.py' "$job"
grep -Fq 'G_SECOND_PRIMITIVE_AMPLITUDE' "$job"
grep -Fq -- '--second-primitive-amplitude' "$job"
grep -Fq 'controlled-g:' "$job"
grep -Fq '3,3,2,0,1' "$job"
grep -Fq 'selected_iq 43' "$job"
grep -Fq 'capture >= 0.9998982409775239' "$job"
grep -Fq 'condition_ratio < 3.0' "$job"
grep -Fq 'virtual_ratio < 1.5' "$job"
grep -Fq 'not an SOS energy gate' "$job"
! grep -Fq 'sbatch ' "$job"

echo CONTROLLED_G_CANDIDATE_GATE_CONTRACT_OK
