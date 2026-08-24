#!/bin/bash
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${TARGET:?TARGET is required}"
: "${PYTHON_EXE:?PYTHON_EXE is required}"
: "${MIGRATION_RECORD:?MIGRATION_RECORD is required}"

SCRIPT="$REPO_ROOT/SIAB/example_C_sternheimer/atomic_optimization_gate/run_atomic_gradient_gate_df.slurm"
RECEIPT="$CAMPAIGN_ROOT/ATOMIC_GRADIENT_GATE_DF_JOB_ID.txt"
RESULT="$CAMPAIGN_ROOT/runs/atomic_gradient_gate_df/ATOMIC_GRADIENT_GATE_RESULT.json"

test -f "$SCRIPT" && test ! -L "$SCRIPT"
test -f "$MIGRATION_RECORD" && test ! -L "$MIGRATION_RECORD"
grep -qx "server66_job_id=410776" "$MIGRATION_RECORD"
grep -qx "server66_state=CANCELLED" "$MIGRATION_RECORD"
grep -qx "server66_elapsed=00:00:00" "$MIGRATION_RECORD"
test ! -e "$RECEIPT"
test ! -e "$RESULT"
mkdir -p "$CAMPAIGN_ROOT/runs"

EXPECTED_SOURCE_COMMIT=$(cd "$REPO_ROOT" && git rev-parse HEAD)
test -n "$EXPECTED_SOURCE_COMMIT"
EXPORTS="ALL,REPO_ROOT=$REPO_ROOT,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,TARGET=$TARGET,PYTHON_EXE=$PYTHON_EXE,EXPECTED_SOURCE_COMMIT=$EXPECTED_SOURCE_COMMIT"
sbatch --test-only --export="$EXPORTS" "$SCRIPT"
job_id=$(sbatch --parsable --export="$EXPORTS" "$SCRIPT")
printf '%s\n' "$job_id" > "$RECEIPT"
printf 'submitted migrated C atomic gradient gate job %s\n' "$job_id"
