#!/bin/bash
# Validate and index cached occupied response targets only.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
set -euo pipefail

run=${1:?run directory required}
source=${2:?immutable source required}
target_root=${3:?complete target root required}
source_summary=${4:?source summary required}
source_summary_sha256=${5:?source summary SHA256 required}

cd "$run"
mkdir execution-once.lock
export C_EXECUTION_HOST=df_iopcas_ghj PYTHONDONTWRITEBYTECODE=1
source_commit=$(/data/home/df_iopcas_ghj/app/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' \
    "$source/SOURCE_MANIFEST.json")
status=failed
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$status" > JOB_STATUS
    printf 'status=%s\njob_id=%s\nexit_code=%s\n' \
        "$status" "$SLURM_JOB_ID" "$rc" > JOB_PROVENANCE
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

/usr/bin/time -v /data/home/df_iopcas_ghj/app/miniconda3/bin/python -u \
    "$source/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/collect_c_jy_response_targets.py" \
    --source "$source" --source-commit "$source_commit" \
    --target-root "$target_root" --source-summary "$source_summary" \
    --source-summary-sha256 "$source_summary_sha256" --output "$run/result" \
    > collect.out 2> collect.err
status=success
