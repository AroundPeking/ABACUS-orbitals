#!/bin/bash
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${TARGET:?TARGET is required}"
: "${PYTHON_EXE:?PYTHON_EXE is required}"

SCRIPT="$REPO_ROOT/SIAB/example_C_sternheimer/atomic_optimization_gate/run_atomic_gradient_gate_server66.slurm"
RECEIPT="$CAMPAIGN_ROOT/ATOMIC_GRADIENT_GATE_JOB_ID.txt"
RESULT="$CAMPAIGN_ROOT/runs/atomic_gradient_gate/ATOMIC_GRADIENT_GATE_RESULT.json"

test -f "$SCRIPT" && test ! -L "$SCRIPT"
test ! -e "$RECEIPT"
test ! -e "$RESULT"
mkdir -p "$CAMPAIGN_ROOT/runs"

EXPORTS="ALL,REPO_ROOT=$REPO_ROOT,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,TARGET=$TARGET,PYTHON_EXE=$PYTHON_EXE"
sbatch --test-only --export="$EXPORTS" "$SCRIPT"
job_id=$(sbatch --parsable --export="$EXPORTS" "$SCRIPT")
printf '%s\n' "$job_id" > "$RECEIPT"
printf 'submitted C atomic gradient gate job %s\n' "$job_id"
