#!/bin/bash
#SBATCH --job-name=c_radial_expand
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=102400M
#SBATCH --time=01:00:00
#SBATCH --no-requeue

source ~/.bashrc >/dev/null 2>&1 || true
set -euo pipefail
export LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=28 MKL_NUM_THREADS=28 OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
: "${C_EXPANSION_STAGE:?}" "${C_EXPANSION_COMMIT:?}" "${C_EXPANSION_DEPLOYMENT_SHA:?}"
stage=$C_EXPANSION_STAGE
root=/work1/ghj/c-solid-fd8-q13-standard-20260903
workflow=$stage/source/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
args=(--stage "$stage" --source-commit "$C_EXPANSION_COMMIT" --deployment-sha256 "$C_EXPANSION_DEPLOYMENT_SHA")
status=failed
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; printf "%s\n" "$status" > "$stage/STATUS"; exit "$rc"' EXIT
cd "$stage"
"$root/optimizer-env-torch1121cpu/bin/python" "$workflow/screen_c_radial_expansion.py" "${args[@]}" --preflight-only
mkdir "$stage/EXECUTION_LOCK"
/usr/bin/time -v -o "$stage/screen.time" \
 "$root/optimizer-env-torch1121cpu/bin/python" "$workflow/screen_c_radial_expansion.py" "${args[@]}"
status=success
