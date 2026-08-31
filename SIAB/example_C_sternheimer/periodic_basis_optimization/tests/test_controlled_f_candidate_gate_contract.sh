#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
job=$root/run_controlled_f_candidate_gate_df.slurm

test -s "$job"
grep -Fq '#SBATCH --partition=48cp2,p1' "$job"
grep -Fq 'prepare_controlled_f_candidate.py' "$job"
grep -Fq 'SECOND_PRIMITIVE_AMPLITUDE' "$job"
grep -Fq -- '--second-primitive-amplitude "$second_primitive_amplitude"' "$job"
grep -Fq 'controlled-f:' "$job"
grep -Fq '3,3,2,1,0' "$job"
grep -Fq 'selected_iq 43' "$job"
grep -Fq 'capture >= 0.9998982409775239' "$job"
grep -Fq 'condition_ratio < 3.0' "$job"
grep -Fq 'virtual_ratio < 1.5' "$job"
grep -Fq 'not an SOS energy gate' "$job"
! grep -Fq 'sbatch ' "$job"

echo CONTROLLED_F_CANDIDATE_GATE_CONTRACT_OK
