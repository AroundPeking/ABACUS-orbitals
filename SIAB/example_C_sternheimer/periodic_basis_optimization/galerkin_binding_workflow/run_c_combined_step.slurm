#!/bin/bash
#SBATCH --job-name=c_pbe_tangent
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
: "${C_COMBINED_STAGE:?}"
: "${C_COMBINED_SOURCE:?}"
: "${C_COMBINED_DEPLOYMENT_SHA256:?}"
stage=$C_COMBINED_STAGE
root=/work1/ghj/c-solid-fd8-q13-standard-20260903
mkdir "$stage/EXECUTION_LOCK"
cd "$stage"
status=failed
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'rc=$?; printf "%s\n" "$status" > "$stage/STATUS"; exit "$rc"' EXIT
source /public/home/ghj/app/src/env_60_245_intel2021.sh
export LD_LIBRARY_PATH=/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021/lib:$root/deps/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE OMP_STACKSIZE=2G
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp I_MPI_FABRICS=shm:ofi
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi2.so
workflow=$C_COMBINED_SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
/usr/bin/time -v -o "$stage/combined.time" \
 "$root/optimizer-env-torch1121cpu/bin/python" "$workflow/run_c_combined_step.py" --stage "$stage"
status=success
