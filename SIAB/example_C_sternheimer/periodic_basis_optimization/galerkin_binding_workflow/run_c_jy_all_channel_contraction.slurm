#!/bin/bash
# Cached jY response contraction only; no ABACUS, LibRPA, or Delta-ST.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=190000M
#SBATCH --time=12:00:00
set -euo pipefail

run=${1:?run directory required}
source=${2:?immutable source required}
target_root=${3:?occupied target root required}
collection=${4:?collection summary required}
collection_sha256=${5:?collection summary SHA256 required}
initial_root=${6:?initial rank ladder required}
profile=${7:?one fixed profile required}

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
    printf 'status=%s\njob_id=%s\nexit_code=%s\nprofile=%s\n' \
        "$status" "$SLURM_JOB_ID" "$rc" "$profile" > JOB_PROVENANCE
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

/usr/bin/time -v /data/home/df_iopcas_ghj/app/miniconda3/bin/python -u \
    "$source/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_jy_all_channel_contraction.py" \
    --source "$source" --source-commit "$source_commit" \
    --target-root "$target_root" \
    --collection-summary "$collection" \
    --collection-summary-sha256 "$collection_sha256" \
    --initial-root "$initial_root" --output "$run/result" \
    --profiles "$profile" --torch-threads "$SLURM_CPUS_PER_TASK" \
    --max-steps "${MAX_STEPS:-12}" \
    --max-evaluations "${MAX_EVALUATIONS:-32}" \
    --radius "${TRUST_RADIUS:-0.05}" \
    --occupied-weight "${OCCUPIED_WEIGHT:-1.0}" \
    --occupied-capture-floor "${OCCUPIED_CAPTURE_FLOOR:-0.999999}" \
    > contraction.out 2> contraction.err
status=success
