#!/bin/bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
script=$root/galerkin_binding_workflow/run_c_candidate_bank_df.slurm

test -s "$script"
grep -q '^#SBATCH --partition=p1,48cp2$' "$script"
grep -q '^#SBATCH --ntasks=1$' "$script"
grep -q '^#SBATCH --cpus-per-task=40$' "$script"
grep -q '^#SBATCH --mem=190000$' "$script"
grep -q 'test ! -e "$output"' "$script"
grep -q 'CANDIDATE_BANK.json' "$script"
grep -q 'STATUS.json' "$script"
grep -q 'PROVENANCE.json' "$script"
grep -q 'sha256sum' "$script"
grep -q 'nvidia-smi' "$script" && exit 1 || true
