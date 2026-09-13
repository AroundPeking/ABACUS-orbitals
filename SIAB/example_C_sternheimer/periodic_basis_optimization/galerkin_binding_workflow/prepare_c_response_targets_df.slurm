#!/bin/bash
# Cached-matrix targets only. Placement and the accepted baseline are manifest-bound.
set -euo pipefail
test "$C_EXECUTION_HOST" = df_iopcas_ghj
test "$SLURM_JOB_NUM_NODES" = 1
test "$SLURM_JOB_PARTITION" = "$C_EXPECTED_PARTITION"
test "$SLURM_CPUS_PER_TASK" = "$C_EXPECTED_THREADS"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
exec /usr/bin/time -v /data/home/df_iopcas_ghj/app/miniconda3/bin/python -u \
  "$C_MOTHER_SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_available_mother.py" \
  --bundle "$C_MOTHER_BUNDLE" --output "$C_MOTHER_OUTPUT" \
  --source-commit "$C_MOTHER_COMMIT" --q-slot "$SLURM_ARRAY_TASK_ID" \
  --relative-rank-tolerance 1e-10 --prepare-targets-from "$C_TARGET_BASELINE"
