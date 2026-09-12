#!/bin/bash
#SBATCH --job-name=c-rpa-lbfgs-compare
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000M
#SBATCH --time=03:00:00

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export C_EXECUTION_HOST=df_iopcas_ghj
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
ROOT="${C_COMPARISON_ROOT:?missing reserved output root}"
SOURCE="${C_COMPARISON_SOURCE:?missing immutable deployment}"
COMMIT="${C_COMPARISON_COMMIT:?missing source commit}"
BUNDLE=/data/home/df_iopcas_ghj/app/siab/c-solid-rpa-continuation-20260912/bundle
PYTHON=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
test -d "$ROOT"
test ! -e "$ROOT/result"
test ! -e "$ROOT/BATCH_STATUS"
printf 'running\n' > "$ROOT/BATCH_STATUS"
/usr/bin/time -v -o "$ROOT/TIMING.txt" "$PYTHON" -u \
  "$SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_optimizer_comparison.py" \
  --bundle "$BUNDLE" --output "$ROOT/result" --source-commit "$COMMIT"
"$PYTHON" - "$ROOT/result/RESULT.json" <<'PY'
import json, os, sys
r=json.load(open(sys.argv[1]))
assert r['status']=='success' and r['job_id']==os.environ['SLURM_JOB_ID']
assert r['physical_release_gate']=='hold'
assert set(r['arms'])=={'steepest','lbfgs'}
for a in r['arms'].values():
    assert a['status']=='success' and a['counts']['forward']<=r['max_forwards_per_arm']
PY
printf 'success\n' > "$ROOT/BATCH_STATUS"
