#!/bin/bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
script=$root/galerkin_binding_workflow/run_c_candidate_bank_q3_df.slurm

test -s "$script"
grep -qx '#SBATCH --partition=p1,48cp2' "$script"
grep -qx '#SBATCH --ntasks=1' "$script"
grep -qx '#SBATCH --cpus-per-task=40' "$script"
grep -qx '#SBATCH --mem=190000' "$script"
grep -Fq 'test ! -e "$output"' "$script"
grep -Fq 'selected_iq 43' "$script"
grep -Fq 'COMPARISON_RESULT.json' "$script"
grep -Fq 'SELECTION_RESULT.json' "$script"
grep -Fq 'STATUS.json' "$script"
grep -Fq 'PROVENANCE.json' "$script"
grep -Fq 'sha256sum' "$script"
! grep -Fq 'nvidia-smi' "$script"
