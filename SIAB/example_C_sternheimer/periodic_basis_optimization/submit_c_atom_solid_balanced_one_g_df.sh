#!/bin/bash
set -euo pipefail

: "${REPO_ROOT:?}"
: "${CAMPAIGN_ROOT:?}"
: "${ATOM_PAIR_ROOT:?}"
: "${RUN_ROOT:?}"

mode=${1:?usage: submit_c_atom_solid_balanced_one_g_df.sh pilot|production}
case "$mode" in
  pilot|production) ;;
  *) echo "unsupported run mode: $mode" >&2; exit 2 ;;
esac

script=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/run_c_atom_solid_balanced_one_g_df.slurm
run_root=$RUN_ROOT
receipt=${run_root}.JOB_ID.txt
test -f "$script" && test ! -L "$script"
test -d "$CAMPAIGN_ROOT" && test ! -L "$CAMPAIGN_ROOT"
test -d "$ATOM_PAIR_ROOT" && test ! -L "$ATOM_PAIR_ROOT"
test ! -e "$receipt"
test ! -e "$run_root"

source_commit=$(/usr/bin/git -C "$REPO_ROOT" rev-parse HEAD)
test -n "$source_commit"
exports="ALL,REPO_ROOT=$REPO_ROOT,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,ATOM_PAIR_ROOT=$ATOM_PAIR_ROOT,RUN_ROOT=$run_root,RUN_MODE=$mode,EXPECTED_SIAB_COMMIT=$source_commit"

sbatch --test-only --export="$exports" "$script"
job_id=$(sbatch --parsable --export="$exports" "$script")
printf '%s\n' "$job_id" > "$receipt"
printf 'submitted C atom-solid balanced %s job %s\n' "$mode" "$job_id"
