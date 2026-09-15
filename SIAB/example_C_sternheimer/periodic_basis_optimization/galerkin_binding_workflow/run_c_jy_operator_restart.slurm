#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --mem=180000M
#SBATCH --time=02:00:00
set -euo pipefail
module load oneapi/2024.2
root=${1:?prepared isolated run root required}
cd "$root"
mkdir execution-once.lock
test "$SLURM_NTASKS" = 8
test "$SLURM_CPUS_PER_TASK" = 6
test "$SLURM_JOB_NUM_NODES" = 1
for name in $(compgen -v ABACUS_STERNHEIMER_); do
    unset "$name"
done
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
binary=$($python -c 'import json; print(json.load(open("CONTRACT.json"))["binary"])')
expected=$($python -c 'import json; print(json.load(open("CONTRACT.json"))["binary_sha256"])')
test "$(sha256sum "$binary" | awk '{print $1}')" = "$expected"
sha256sum -c INPUT_SHA256SUMS
frozen=$($python -c 'import json; print(int("frozen_auxiliary" in json.load(open("CONTRACT.json"))))')
if [ "$frozen" = 1 ]; then
    export ABACUS_STERNHEIMER_FROZEN_AUXILIARY_DIR="$root/frozen-auxiliary"
    ABACUS_STERNHEIMER_FROZEN_AUXILIARY_METRIC_SHA256=$($python -c 'import json; print(json.load(open("CONTRACT.json"))["frozen_auxiliary"]["metric_sha256"])')
    ABACUS_STERNHEIMER_FROZEN_AUXILIARY_WHITENING_SHA256=$($python -c 'import json; print(json.load(open("CONTRACT.json"))["frozen_auxiliary"]["whitening_sha256"])')
    export ABACUS_STERNHEIMER_FROZEN_AUXILIARY_METRIC_SHA256 ABACUS_STERNHEIMER_FROZEN_AUXILIARY_WHITENING_SHA256
fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
export I_MPI_FABRICS=shm:ofi
status=failed
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$status" > STATUS
    printf 'status=%s\njob_id=%s\nexit_code=%s\nresponse_solved=no\ncompatibility_admitted=no\n' \
        "$status" "$SLURM_JOB_ID" "$rc" > PROVENANCE
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
/usr/bin/time -v mpirun -np "$SLURM_NTASKS" "$binary" > abacus.out 2> abacus.err
sha256sum -c INPUT_SHA256SUMS
suffix=$($python -c 'import json; print(json.load(open("CONTRACT.json"))["suffix"])')
dataset=OUT.$suffix/STERNHEIMER_BASIS_OPERATORS_V1
grep -Fx 'status success' "$dataset/status.dat"
grep -Fx 'response_solved no' "$dataset/status.dat"
grep -Fx 'solved_equations 0' "$dataset/status.dat"
grep -Fx 'ABACUS_STERNHEIMER_BASIS_OPERATORS_MANIFEST_V1' "$dataset/manifest.dat"
grep -Fx 'entry_count 258' "$dataset/manifest.dat"
expected_primitive_count=$($python -c 'import json; c=json.load(open("CONTRACT.json")); print(c.get("primitive_expansion", {}).get("expanded_primitive_count", ""))')
if [ -n "$expected_primitive_count" ]; then
    grep -Fx "primitive_count $expected_primitive_count" "$dataset/manifest.dat"
fi
if [ "$frozen" = 1 ]; then
    grep -Fx 'auxiliary_transform_origin frozen_reference' "$dataset/manifest.dat"
    grep -Fx "frozen_auxiliary_metric_sha256 $ABACUS_STERNHEIMER_FROZEN_AUXILIARY_METRIC_SHA256" "$dataset/manifest.dat"
    grep -Fx "frozen_auxiliary_whitening_sha256 $ABACUS_STERNHEIMER_FROZEN_AUXILIARY_WHITENING_SHA256" "$dataset/manifest.dat"
    rank=$($python -c 'import json; print(json.load(open("CONTRACT.json"))["frozen_auxiliary"]["whitened_auxiliary_rank"])')
    grep -Fx "whitened_auxiliary_rank $rank" "$dataset/manifest.dat"
fi
test "$(find "$dataset" -maxdepth 1 -name '*.bin' | wc -l)" -eq 258
test ! -d "OUT.$suffix/STERNHEIMER_BASIS_OPT_V1"
grep -F 'Read electron density from file:' "OUT.$suffix/running_nscf.log"
status=success
