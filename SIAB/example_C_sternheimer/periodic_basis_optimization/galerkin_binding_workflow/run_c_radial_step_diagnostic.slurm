#!/bin/bash
#SBATCH --job-name=c_radial_cap_diagnostic
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
: "${C_DIAGNOSTIC_STAGE:?}"
: "${C_DIAGNOSTIC_SOURCE_COMMIT:?}"
: "${C_DIAGNOSTIC_DEPLOYMENT_SHA256:?}"
stage=$C_DIAGNOSTIC_STAGE
root=/work1/ghj/c-solid-fd8-q13-standard-20260903
source /public/home/ghj/app/src/env_60_245_intel2021.sh
export LD_LIBRARY_PATH=/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021/lib:$root/deps/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE OMP_STACKSIZE=2G
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi2.so
workflow=$stage/source/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
args=(--stage "$stage" --source-commit "$C_DIAGNOSTIC_SOURCE_COMMIT" --deployment-sha256 "$C_DIAGNOSTIC_DEPLOYMENT_SHA256")
"$root/optimizer-env-torch1121cpu/bin/python" "$workflow/diagnose_c_radial_step_cap.py" "${args[@]}" --preflight-only
mkdir "$stage/EXECUTION_LOCK"
cd "$stage"
status=failed
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; printf "%s\n" "$status" > "$stage/STATUS"; exit "$rc"' EXIT
/usr/bin/time -v -o "$stage/diagnostic.time" \
 "$root/optimizer-env-torch1121cpu/bin/python" "$workflow/diagnose_c_radial_step_cap.py" "${args[@]}"
status=success
