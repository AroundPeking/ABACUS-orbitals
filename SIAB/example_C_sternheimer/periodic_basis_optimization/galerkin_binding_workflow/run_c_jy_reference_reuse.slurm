#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180000M
#SBATCH --time=00:30:00
set -euo pipefail
bundle=${1:?bundle required}
reference=${2:?reference required}
out=${3:?isolated output required}
source_dir=${4:?immutable workflow source required}
commit=${5:?source commit required}
mkdir "$out"
cd "$out"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
# This hash/I/O audit uses only tiny dense contractions; avoid nested BLAS overhead.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=failed
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$status" > JOB_STATUS
    printf 'status=%s\njob_id=%s\nexit_code=%s\nsource=%s\n' \
        "$status" "$SLURM_JOB_ID" "$rc" "$commit" > PROVENANCE
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
/usr/bin/time -v /data/home/df_iopcas_ghj/app/miniconda3/bin/python \
    "$source_dir/audit_c_jy_reference_reuse.py" --bundle "$bundle" --reference "$reference" \
    --output "$out/result" --source-commit "$commit" > audit.out 2> audit.err
sha256sum result/RESULT.json > RESULT_SHA256
status=success
