#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=190000M
#SBATCH --time=04:00:00
set -euo pipefail
run=${1:?registered run directory required}
workflow=${2:?immutable workflow required}
cd "$run"
mkdir execution-once.lock
export C_EXECUTION_HOST=df_iopcas_ghj PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
status=failed
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$status" > JOB_STATUS
    printf 'status=%s\njob_id=%s\nexit_code=%s\n' "$status" "$SLURM_JOB_ID" "$rc" > PROVENANCE
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date
printf '%s\n' "$PWD" "$SLURM_JOB_NODELIST" "$SLURM_CPUS_PER_TASK"
/usr/bin/time -v /data/home/df_iopcas_ghj/app/miniconda3/bin/python -u \
    "$workflow/run_c_jy_joined_ladder.py" --contract "$run/CONTRACT.json" \
    --output "$run/result" > ladder.out 2> ladder.err
status=success
