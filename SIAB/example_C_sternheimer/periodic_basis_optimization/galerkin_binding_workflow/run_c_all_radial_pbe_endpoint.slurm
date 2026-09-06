#!/bin/bash
#SBATCH --job-name=c_free_pbe
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=24000M
#SBATCH --time=01:00:00
#SBATCH --no-requeue

source ~/.bashrc >/dev/null 2>&1 || true
set -euo pipefail
export LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1
: "${C_ENDPOINT_STAGE:?}"
: "${C_OPTIMIZER_RESULT:?}"
: "${C_OPTIMIZER_RESULT_SHA256:?}"
: "${C_PBE_CHECKER_SHA256:?}"
: "${C_PBE_CHECKER_COMMIT:?}"
root=/work1/ghj/c-solid-fd8-q13-standard-20260903
stage=$C_ENDPOINT_STAGE
py=$root/optimizer-env-torch1121cpu/bin/python
abacus=$root/artifacts/abacus-711af860c-intel2021/abacus_3p
checker=$stage/check_c_optimized_pbe.py
result=$C_OPTIMIZER_RESULT
original=$root/runs/c-solid-fd8-q13-standard-711af860c-ab8f1192/q1
mkdir "$stage/EXECUTION_LOCK"
cd "$stage"
status=failed
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; printf "%s\n" "$status" > "$stage/STATUS"; exit "$rc"' EXIT
"$py" - "$checker" "$C_PBE_CHECKER_SHA256" "$abacus" "$result" "$C_OPTIMIZER_RESULT_SHA256" <<'PY'
import hashlib, pathlib, sys
args = sys.argv[1:]
for path, expected in ((args[0], args[1]),
                       (args[2], 'ab8f1192ad7f7246218422c80218c869d8053fa0d44f6ad728b9323eb4ee5ab3'),
                       (args[3], args[4])):
    assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == expected, path
PY
"$py" "$checker" prepare --optimizer-result "$result" \
 --optimizer-result-sha256 "$C_OPTIMIZER_RESULT_SHA256" \
 --original-q1 "$original" \
 --baseline-log-sha256 d139b11e3e99d82c8d1306aa2af83af58a1d481c1be20b2eea8f82b9b4283393 \
 --output "$stage/pbe" > "$stage/preparation.log"
preparation_sha=$($py -c 'import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$stage/pbe/PBE_ENDPOINT_PREPARED.json")
"$py" - "$stage" <<'PY'
import hashlib, json, os, pathlib, sys
stage = pathlib.Path(sys.argv[1])
with (stage/'EXECUTION_PROVENANCE.json').open('x') as stream:
    json.dump(dict(status='launching', server='df_dcu', scheduler_job=os.environ['SLURM_JOB_ID'],
        profile='frozen-c-reference-711af860c-reproduction', version_verdict='feature-branch-exception',
        exception_reason='Same binary and inputs as accepted original C PBE reference',
        abacus_commit='711af860c125b9757c344a1961b63524c550cfe4',
        abacus_sha256='ab8f1192ad7f7246218422c80218c869d8053fa0d44f6ad728b9323eb4ee5ab3',
        checker_commit=os.environ['C_PBE_CHECKER_COMMIT'], checker_sha256=os.environ['C_PBE_CHECKER_SHA256'],
        optimizer_result_sha256=os.environ['C_OPTIMIZER_RESULT_SHA256'],
        reader_contract='not-used-pure-PBE', sternheimer='disabled', librpa='not-used',
        physical_release_gate='hold', scheduler_gate='pending_external_validation',
        mpi_ranks=4, omp_threads=7,
        runner_sha256=hashlib.sha256((stage/'run.slurm').read_bytes()).hexdigest()), stream, indent=2)
PY
source /public/home/ghj/app/src/env_60_245_intel2021.sh
export LD_LIBRARY_PATH=/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021/lib:$root/deps/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE OMP_STACKSIZE=2G
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp I_MPI_FABRICS=shm:ofi
cd "$stage/pbe"
/usr/bin/time -v -o "$stage/abacus.time" \
 mpirun -np 4 -ppn 4 "$abacus" > abacus.out 2>&1
"$py" "$checker" collect --prepared-dir "$stage/pbe" \
 --preparation-sha256 "$preparation_sha" > "$stage/collection.log"
status=success
