#!/bin/bash
# Cached full-q candidate evaluation only; no ABACUS, LibRPA, or Delta-ST.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=190000M
#SBATCH --time=08:00:00
set -euo pipefail

run=${1:?registered q run directory required}
workflow=${2:?immutable workflow required}
cd "$run"
mkdir execution-once.lock
export C_EXECUTION_HOST=df_iopcas_ghj PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE

sed "s/__RUNTIME_JOB_ID__/$SLURM_JOB_ID/g" CONTRACT.json > RUNTIME_CONTRACT.json
q_slot=$(/data/home/df_iopcas_ghj/app/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["q_slot"])' \
    RUNTIME_CONTRACT.json)
if [ "$q_slot" -eq 0 ]; then
    runner=run_c_jy_angular_ladder.py
else
    runner=run_c_jy_joined_ladder.py
fi

status=failed
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$status" > JOB_STATUS
    printf 'status=%s\njob_id=%s\nexit_code=%s\nq_slot=%s\n' \
        "$status" "$SLURM_JOB_ID" "$rc" "$q_slot" > JOB_PROVENANCE
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

/usr/bin/time -v /data/home/df_iopcas_ghj/app/miniconda3/bin/python -u \
    "$workflow/$runner" --contract "$run/RUNTIME_CONTRACT.json" \
    --output "$run/result" > evaluation.out 2> evaluation.err
status=success
