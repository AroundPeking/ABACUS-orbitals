#!/bin/bash
#SBATCH --job-name=c_band_TN_pbe
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=102400M
#SBATCH --time=01:00:00
#SBATCH --no-requeue

source ~/.bashrc >/dev/null 2>&1 || true
set -euo pipefail
export LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1
: "${C_BAND_PBE_STAGE:?}"
: "${C_BAND_PBE_SOURCE:?}"
: "${C_BAND_PBE_COMMIT:?}"
: "${C_BAND_PBE_DEPLOYMENT:?}"
: "${C_BAND_PBE_PARENT:?}"
: "${C_BAND_PBE_PARENT_RESULT:?}"
: "${C_BAND_PBE_PARENT_ACCEPTANCE:?}"
: "${C_BAND_PBE_PARENT_DEPLOYMENT:?}"
: "${C_BAND_PBE_PARENT_COMMIT:?}"
stage=$(cd -- "$C_BAND_PBE_STAGE" && pwd -P)
for name in STATUS PROVENANCE.json result; do
    if [ -e "$stage/$name" ] || [ -L "$stage/$name" ]; then exit 2; fi
done
root=/work1/ghj/c-solid-fd8-q13-standard-20260903
source /public/home/ghj/app/src/env_60_245_intel2021.sh
export LD_LIBRARY_PATH=/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021/lib:$root/deps/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE OMP_STACKSIZE=2G
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm-2.3.6/lib/libpmi2.so.0.0.0
workflow=$C_BAND_PBE_SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
args=(--stage "$stage" --source-commit "$C_BAND_PBE_COMMIT" --deployment-sha256 "$C_BAND_PBE_DEPLOYMENT"
 --preparation-stage "$C_BAND_PBE_PARENT" --preparation-result-sha256 "$C_BAND_PBE_PARENT_RESULT"
 --preparation-acceptance-sha256 "$C_BAND_PBE_PARENT_ACCEPTANCE"
 --preparation-deployment-sha256 "$C_BAND_PBE_PARENT_DEPLOYMENT" --preparation-source-commit "$C_BAND_PBE_PARENT_COMMIT")
"$root/optimizer-env-torch1121cpu/bin/python" "$workflow/run_c_band_tangent_pbe_calibration.py" "${args[@]}" --preflight-only
mkdir "$stage/EXECUTION_LOCK"
status=failed
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; if [ "$rc" -ne 0 ] || [ "$status" != success ]; then status=failed; [ "$rc" -ne 0 ] || rc=1; fi; printf "%s\n" "$status" > "$stage/STATUS"; exit "$rc"' EXIT
cd "$stage"
/usr/bin/time -v -o "$stage/calibration.time" \
 "$root/optimizer-env-torch1121cpu/bin/python" "$workflow/run_c_band_tangent_pbe_calibration.py" "${args[@]}"
status=success
