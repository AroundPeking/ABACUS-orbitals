#!/bin/bash
set -euo pipefail

: "${REPO_ROOT:?}"
: "${CAMPAIGN_ROOT:?}"
: "${ATOM_PAIR_ROOT:?}"
: "${RUN_ROOT:?}"

mode=${1:?usage: submit_c_atom_solid_balanced_one_g_df.sh pilot|production [one_g|no_f|relaxed_dzp]}
case "$mode" in
  pilot|production) ;;
  *) echo "unsupported run mode: $mode" >&2; exit 2 ;;
esac
profile=${2:-one_g}
case "$profile" in
  one_g|no_f|relaxed_dzp) ;;
  *) echo "unsupported candidate profile: $profile" >&2; exit 2 ;;
esac

script=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/run_c_atom_solid_balanced_one_g_df.slurm
run_root=$RUN_ROOT
receipt=${run_root}.JOB_ID.txt
test -f "$script" && test ! -L "$script"
test -d "$CAMPAIGN_ROOT" && test ! -L "$CAMPAIGN_ROOT"
test -d "$ATOM_PAIR_ROOT" && test ! -L "$ATOM_PAIR_ROOT"
test ! -e "$receipt"
test ! -e "$run_root"

if source_commit=$(/usr/bin/git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null); then
  :
else
  source_commit=$(cat "$REPO_ROOT/.git/HEAD")
fi
if ! printf '%s\n' "$source_commit" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "source commit must be a 40-character lowercase SHA" >&2
  exit 2
fi
exports="ALL,REPO_ROOT=$REPO_ROOT,CAMPAIGN_ROOT=$CAMPAIGN_ROOT,ATOM_PAIR_ROOT=$ATOM_PAIR_ROOT,RUN_ROOT=$run_root,RUN_MODE=$mode,CANDIDATE_PROFILE=$profile,EXPECTED_SIAB_COMMIT=$source_commit"

sbatch --test-only --export="$exports" "$script"
job_id=$(sbatch --parsable --export="$exports" "$script")
printf '%s\n' "$job_id" > "$receipt"
printf 'submitted C atom-solid balanced %s %s job %s\n' "$mode" "$profile" "$job_id"
