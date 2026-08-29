#!/bin/bash
set -euo pipefail

: "${REPO_ROOT:?}"
: "${SOURCE_ROOT:?}"
: "${RUN_ROOT:?}"
: "${ABACUS_EXE:?}"
: "${ABACUS_SHA256:?}"
: "${EXPECTED_TARGET_SHA256:?}"
: "${PYTHON_EXE:?}"

script="$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/run_c_atom_source_only_df.slurm"
run_root=$RUN_ROOT
receipt="${run_root}.JOB_ID.txt"
test -f "$script" && test ! -L "$script"
test -d "$SOURCE_ROOT" && test ! -L "$SOURCE_ROOT"
test -x "$ABACUS_EXE"
test -x "$PYTHON_EXE"
test ! -e "$receipt"
test ! -e "$run_root"

source_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
test -n "$source_commit"
exports="ALL,REPO_ROOT=$REPO_ROOT,SOURCE_ROOT=$SOURCE_ROOT,RUN_ROOT=$run_root,ABACUS_EXE=$ABACUS_EXE,ABACUS_SHA256=$ABACUS_SHA256,EXPECTED_SOURCE_COMMIT=$source_commit,EXPECTED_TARGET_SHA256=$EXPECTED_TARGET_SHA256,PYTHON_EXE=$PYTHON_EXE"

sbatch --test-only --export="$exports" "$script"
job_id=$(sbatch --parsable --export="$exports" "$script")
printf '%s\n' "$job_id" > "$receipt"
printf 'submitted C atom SIAB source-only job %s\n' "$job_id"

