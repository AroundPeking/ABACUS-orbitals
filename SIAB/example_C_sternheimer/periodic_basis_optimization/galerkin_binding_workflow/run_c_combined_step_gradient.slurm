#!/bin/bash
#SBATCH --job-name=c_ec_refresh
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=102400M
#SBATCH --time=01:00:00
#SBATCH --no-requeue

source ~/.bashrc >/dev/null 2>&1 || true
set -euo pipefail
export LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1
: "${C_EC_REFRESH_STAGE:?}"
: "${C_EC_REFRESH_SOURCE:?}"
: "${C_EC_REFRESH_SOURCE_COMMIT:?}"
: "${C_EC_REFRESH_DEPLOYMENT_SHA256:?}"
: "${C_EC_REFRESH_ACCEPTED_STAGE:?}"
: "${C_EC_REFRESH_RESULT_SHA256:?}"
: "${C_EC_REFRESH_ACCEPTANCE_SHA256:?}"
stage=$(cd -- "$C_EC_REFRESH_STAGE" && pwd -P)
accepted_stage=$(cd -- "$C_EC_REFRESH_ACCEPTED_STAGE" && pwd -P)
case "$stage/" in "$accepted_stage/"*) exit 2;; esac
case "$accepted_stage/" in "$stage/"*) exit 2;; esac
for name in STATUS PROVENANCE.json result; do
    if [ -e "$stage/$name" ] || [ -L "$stage/$name" ]; then exit 2; fi
done
root=/work1/ghj/c-solid-fd8-q13-standard-20260903
mkdir "$stage/EXECUTION_LOCK"
status=failed
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; if [ "$rc" -ne 0 ] || [ "$status" != success ]; then status=failed; [ "$rc" -ne 0 ] || rc=1; fi; printf "%s\n" "$status" > "$stage/STATUS"; exit "$rc"' EXIT
cd "$stage"
source /public/home/ghj/app/src/env_60_245_intel2021.sh
export LD_LIBRARY_PATH=/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021/lib:$root/deps/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28 OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE OMP_STACKSIZE=2G
workflow=$C_EC_REFRESH_SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
/usr/bin/time -v -o "$stage/ec-gradient.time" \
 "$root/optimizer-env-torch1121cpu/bin/python" "$workflow/refresh_c_combined_step_gradient.py" \
 --accepted-stage "$C_EC_REFRESH_ACCEPTED_STAGE" \
 --result-sha256 "$C_EC_REFRESH_RESULT_SHA256" \
 --acceptance-sha256 "$C_EC_REFRESH_ACCEPTANCE_SHA256" \
 --stage "$stage" --source-commit "$C_EC_REFRESH_SOURCE_COMMIT" \
 --deployment-sha256 "$C_EC_REFRESH_DEPLOYMENT_SHA256"
status=success
