#!/bin/bash
# Deterministic JSON collection only; no ABACUS, LibRPA, or Delta-ST.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:20:00
set -euo pipefail

run=${1:?registered collector run directory required}
workflow=${2:?immutable workflow required}
cd "$run"
mkdir execution-once.lock
export C_EXECUTION_HOST=df_iopcas_ghj PYTHONDONTWRITEBYTECODE=1
sed "s/__RUNTIME_JOB_ID__/$SLURM_JOB_ID/g" CONTRACT.json > RUNTIME_CONTRACT.json
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

/data/home/df_iopcas_ghj/app/miniconda3/bin/python -u \
    "$workflow/collect_c_jy_compressed_rank_ladder.py" \
    --contract "$run/RUNTIME_CONTRACT.json" --output "$run/result" \
    > collector.out 2> collector.err
status=success
