#!/bin/bash
#SBATCH --job-name=c_ec_constraints
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
: "${C_CONSTRAINT_STAGE:?}"
: "${C_CONSTRAINT_SOURCE:?}"
: "${C_CONSTRAINT_COMMIT:?}"
: "${C_CONSTRAINT_DEPLOYMENT:?}"
: "${C_CONSTRAINT_GRADIENT_STAGE:?}"
: "${C_CONSTRAINT_GRADIENT_RESULT:?}"
: "${C_CONSTRAINT_GRADIENT_ACCEPTANCE:?}"
stage=$(cd -- "$C_CONSTRAINT_STAGE" && pwd -P)
previous=$(cd -- "$C_CONSTRAINT_GRADIENT_STAGE" && pwd -P)
case "$stage/" in "$previous/"*) exit 2;; esac
case "$previous/" in "$stage/"*) exit 2;; esac
for name in STATUS PROVENANCE.json result; do
    if [ -e "$stage/$name" ] || [ -L "$stage/$name" ]; then exit 2; fi
done
mkdir "$stage/EXECUTION_LOCK"
status=failed
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; if [ "$rc" -ne 0 ] || [ "$status" != success ]; then status=failed; [ "$rc" -ne 0 ] || rc=1; fi; printf "%s\n" "$status" > "$stage/STATUS"; exit "$rc"' EXIT
cd "$stage"
root=/work1/ghj/c-solid-fd8-q13-standard-20260903
source /public/home/ghj/app/src/env_60_245_intel2021.sh
export LD_LIBRARY_PATH=/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021/lib:$root/deps/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28 OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE OMP_STACKSIZE=2G
workflow=$C_CONSTRAINT_SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
/usr/bin/time -v -o "$stage/constraint-screen.time" \
 "$root/optimizer-env-torch1121cpu/bin/python" "$workflow/screen_c_ec_step_constraints.py" \
 --stage "$stage" --source-commit "$C_CONSTRAINT_COMMIT" --deployment-sha256 "$C_CONSTRAINT_DEPLOYMENT" \
 --gradient-stage "$C_CONSTRAINT_GRADIENT_STAGE" --gradient-result-sha256 "$C_CONSTRAINT_GRADIENT_RESULT" \
 --gradient-acceptance-sha256 "$C_CONSTRAINT_GRADIENT_ACCEPTANCE"
status=success
