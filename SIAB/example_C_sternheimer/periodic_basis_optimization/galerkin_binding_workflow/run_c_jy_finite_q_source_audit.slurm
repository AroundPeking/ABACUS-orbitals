#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=190000M
#SBATCH --time=00:45:00
set -euo pipefail
run=${1:?source run required}
bundle=${2:?bundle required}
out=${3:?isolated audit output required}
source_dir=${4:?immutable audit workflow required}
commit=${5:?source commit required}
reader_source=${6:?frozen cache reader required}
mkdir "$out"
cd "$out"
# Hashing and tiny gauge contractions dominate; avoid nested BLAS overhead.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
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
    "$source_dir/audit_c_jy_finite_q_source.py" --run "$run" --bundle "$bundle" \
    --output "$out/result" --source-commit "$commit" --reader-source "$reader_source" \
    > audit.out 2> audit.err
sha256sum result/RESULT.json result/GAUGE_MAPS.npz > RESULT_SHA256
status=success
