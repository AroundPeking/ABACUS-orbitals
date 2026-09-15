#!/bin/bash
# Cached covariance eigenspectra only; no electronic-structure calculation.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64000M
#SBATCH --time=04:00:00
set -euo pipefail

run=${1:?run directory required}
source=${2:?immutable source required}
target_root=${3:?occupied target root required}
collection=${4:?collection summary required}
collection_sha256=${5:?collection summary SHA256 required}
profiles=${6:-all}

cd "$run"
mkdir execution-once.lock
export C_EXECUTION_HOST=df_iopcas_ghj PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE

source_commit=$(/data/home/df_iopcas_ghj/app/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' \
    "$source/SOURCE_MANIFEST.json")
status=failed
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$status" > JOB_STATUS
    printf 'status=%s\njob_id=%s\nexit_code=%s\nprofiles=%s\n' \
        "$status" "$SLURM_JOB_ID" "$rc" "$profiles" > JOB_PROVENANCE
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

/usr/bin/time -v /data/home/df_iopcas_ghj/app/miniconda3/bin/python -u \
    "$source/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_jy_rank_capacity_bound.py" \
    --source "$source" --source-commit "$source_commit" \
    --target-root "$target_root" \
    --collection-summary "$collection" \
    --collection-summary-sha256 "$collection_sha256" \
    --output "$run/result" --profiles "$profiles" \
    > rank-bound.out 2> rank-bound.err
status=success
