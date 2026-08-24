#!/bin/bash
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${TARGET:?TARGET is required}"
: "${SEED:?SEED is required}"
: "${PYTHON_EXE:?PYTHON_EXE is required}"

SCRIPT="$REPO_ROOT/SIAB/example_C_sternheimer/atomic_basis_optimization/run_f_shell_optimization_df.slurm"
RECEIPT="$CAMPAIGN_ROOT/F_SHELL_OPTIMIZATION_DF_JOB_ID.txt"
RESULT="$CAMPAIGN_ROOT/runs/f_shell_optimization/F_SHELL_OPTIMIZATION_RESULT.json"

for path in "$SCRIPT" "$TARGET" "$SEED"; do
  test -e "$path" && test ! -L "$path"
done
test -x "$PYTHON_EXE"
test ! -e "$RECEIPT"
test ! -e "$RESULT"
mkdir -p "$CAMPAIGN_ROOT/runs"

EXPECTED_SOURCE_COMMIT=$(cd "$REPO_ROOT" && git rev-parse HEAD)
test -n "$EXPECTED_SOURCE_COMMIT"
EXPORTS="ALL,REPO_ROOT=$REPO_ROOT,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,TARGET=$TARGET,SEED=$SEED,PYTHON_EXE=$PYTHON_EXE,EXPECTED_SOURCE_COMMIT=$EXPECTED_SOURCE_COMMIT"
cd "$CAMPAIGN_ROOT"
sbatch --test-only --export="$EXPORTS" "$SCRIPT"
job_id=$(sbatch --parsable --export="$EXPORTS" "$SCRIPT")
printf '%s\n' "$job_id" > "$RECEIPT"
printf 'submitted C f-shell response optimization job %s\n' "$job_id"
