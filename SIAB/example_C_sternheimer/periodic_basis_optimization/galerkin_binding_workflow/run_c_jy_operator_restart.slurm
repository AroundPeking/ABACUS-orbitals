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
test "$(find "$dataset" -maxdepth 1 -name '*.bin' | wc -l)" -eq 258
test ! -d "OUT.$suffix/STERNHEIMER_BASIS_OPT_V1"
grep -F 'Read electron density from file:' "OUT.$suffix/running_nscf.log"
status=success
