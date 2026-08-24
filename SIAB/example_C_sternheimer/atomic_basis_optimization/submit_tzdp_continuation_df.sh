#!/bin/bash
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${TARGET:?TARGET is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${PYTHON_EXE:?PYTHON_EXE is required}"

SCRIPT="$REPO_ROOT/SIAB/example_C_sternheimer/atomic_basis_optimization/run_tzdp_continuation_df.slurm"
RECEIPT="$CAMPAIGN_ROOT/TZDP_CONTINUATION_DF_JOB_ID.txt"
RESULT="$CAMPAIGN_ROOT/runs/tzdp_continuation/TZDP_CONTINUATION_RESULT.json"
SPECTRUM="$CAMPAIGN_ROOT/runs/tzdp_continuation/RESIDUAL_SPECTRUM.json"

for path in "$SCRIPT" "$TARGET" "$CHECKPOINT" "$PYTHON_EXE"; do
  test -e "$path" && test ! -L "$path"
done
test -x "$PYTHON_EXE"
test ! -e "$RECEIPT"
test ! -e "$RESULT"
test ! -e "$SPECTRUM"
mkdir -p "$CAMPAIGN_ROOT/runs"

EXPECTED_SOURCE_COMMIT=$(cd "$REPO_ROOT" && git rev-parse HEAD)
test -n "$EXPECTED_SOURCE_COMMIT"
EXPORTS="ALL,REPO_ROOT=$REPO_ROOT,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,TARGET=$TARGET,CHECKPOINT=$CHECKPOINT,PYTHON_EXE=$PYTHON_EXE,EXPECTED_SOURCE_COMMIT=$EXPECTED_SOURCE_COMMIT"
sbatch --test-only --export="$EXPORTS" "$SCRIPT"
job_id=$(sbatch --parsable --export="$EXPORTS" "$SCRIPT")
printf '%s\n' "$job_id" > "$RECEIPT"
printf 'submitted C TZDP Sternheimer continuation job %s\n' "$job_id"
