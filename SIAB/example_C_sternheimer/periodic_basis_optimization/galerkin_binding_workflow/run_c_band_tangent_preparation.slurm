#!/bin/bash
#SBATCH --job-name=c_band_tangent
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
: "${C_BAND_STAGE:?}"
: "${C_BAND_SOURCE:?}"
: "${C_BAND_COMMIT:?}"
: "${C_BAND_DEPLOYMENT:?}"
: "${C_BAND_SCREEN_STAGE:?}"
: "${C_BAND_SCREEN_RESULT:?}"
: "${C_BAND_SCREEN_ACCEPTANCE:?}"
: "${C_BAND_SCREEN_DEPLOYMENT:?}"
: "${C_BAND_SCREEN_COMMIT:?}"
: "${C_BAND_GRADIENT_RESULT:?}"
: "${C_BAND_GRADIENT_ACCEPTANCE:?}"
stage=$(cd -- "$C_BAND_STAGE" && pwd -P)
previous=$(cd -- "$C_BAND_SCREEN_STAGE" && pwd -P)
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
workflow=$C_BAND_SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
/usr/bin/time -v -o "$stage/band-tangent-preparation.time" \
 "$root/optimizer-env-torch1121cpu/bin/python" "$workflow/prepare_c_band_tangent_probes.py" \
 --stage "$stage" --source-commit "$C_BAND_COMMIT" --deployment-sha256 "$C_BAND_DEPLOYMENT" \
 --screen-stage "$C_BAND_SCREEN_STAGE" --screen-result-sha256 "$C_BAND_SCREEN_RESULT" \
 --screen-acceptance-sha256 "$C_BAND_SCREEN_ACCEPTANCE" --screen-deployment-sha256 "$C_BAND_SCREEN_DEPLOYMENT" \
 --screen-source-commit "$C_BAND_SCREEN_COMMIT" --gradient-result-sha256 "$C_BAND_GRADIENT_RESULT" \
 --gradient-acceptance-sha256 "$C_BAND_GRADIENT_ACCEPTANCE"
status=success
