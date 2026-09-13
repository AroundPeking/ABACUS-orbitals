#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32000M
#SBATCH --time=00:30:00
set -euo pipefail
run=${1:?operator run required}
out=${2:?isolated output directory required}
source_dir=${3:?immutable workflow source required}
mkdir "$out"
cd "$out"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
status=failed
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$status" > JOB_STATUS
    printf 'status=%s\njob_id=%s\nexit_code=%s\nsource=%s\n' \
        "$status" "$SLURM_JOB_ID" "$rc" "$source_dir" > PROVENANCE
    sha256sum "$source_dir/audit_c_jy_operator_restart.py" > SOURCE_SHA256
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
/usr/bin/time -v /data/home/df_iopcas_ghj/app/miniconda3/bin/python \
    "$source_dir/audit_c_jy_operator_restart.py" --run "$run" --output "$out/result" \
    > audit.out 2> audit.err
sha256sum result/RESULT.json > RESULT_SHA256
status=success
