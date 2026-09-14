#!/bin/bash
#SBATCH --job-name=c_jy_freeze_f
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=190000M
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/c-solid-rpa-continuation-20260912/log-c-jy-frozen-spd-f-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/c-solid-rpa-continuation-20260912/log-c-jy-frozen-spd-f-%j.err

set -euo pipefail
date
echo "PWD=$PWD"
echo "JOB=$SLURM_JOB_ID NODELIST=$SLURM_JOB_NODELIST CPUS=$SLURM_CPUS_PER_TASK"
export C_EXECUTION_HOST=df_iopcas_ghj
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

BASE=/data/home/df_iopcas_ghj/app/siab/c-solid-rpa-continuation-20260912
: "${SOURCE:?SOURCE must be exported at submission}"
SOURCE_COMMIT="$(/data/home/df_iopcas_ghj/app/miniconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$SOURCE/SOURCE_MANIFEST.json")"
TARGET_ROOT="$BASE/jy-spdf-response-targets-c91e23a8"
BUNDLE="$BASE/bundle"
RANK_ROOT="$BASE/jy-spdf-radial-rank-ladder-30d4390e"
OUTPUT="$BASE/jy-spdf-frozen-spd-f-fit-28e6867d"

test ! -e "$OUTPUT/STATUS"
PYTHONPATH="$SOURCE/SIAB/opt_orb_pytorch_dpsi:$SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow" \
  /data/home/df_iopcas_ghj/app/miniconda3/bin/python \
  "$SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_jy_frozen_spd_f_fit.py" \
  --source "$SOURCE" \
  --source-commit "$SOURCE_COMMIT" \
  --target-root "$TARGET_ROOT" \
  --bundle "$BUNDLE" \
  --rank-root "$RANK_ROOT" \
  --output "$OUTPUT" \
  --collection-summary-sha256 a63d18ab16d173db6a37df27e38a39ba03ae7d34548339e1bc21aece03e0e162 \
  --max-steps 8 \
  --max-evaluations 20 \
  --radius 0.05
