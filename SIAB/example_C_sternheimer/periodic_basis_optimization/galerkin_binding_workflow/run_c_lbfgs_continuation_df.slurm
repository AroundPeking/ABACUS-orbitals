#!/bin/bash
#SBATCH --job-name=c-rpa-lbfgs-continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?full-node CPU count required}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
export C_EXECUTION_HOST=df_iopcas_ghj
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
ROOT="${C_COMPARISON_ROOT:?reserved run required}"
SOURCE="${C_COMPARISON_SOURCE:?immutable source required}"
COMMIT="${C_COMPARISON_COMMIT:?source commit required}"
PARENT="${C_CONTINUATION_ROOT:?completed parent required}"
PARENT_SHA="${C_CONTINUATION_SHA:?parent result hash required}"
BUNDLE=/data/home/df_iopcas_ghj/app/siab/c-solid-rpa-continuation-20260912/bundle
PYTHON=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
test -d "$ROOT"
test ! -e "$ROOT/result"
test ! -e "$ROOT/BATCH_STATUS"
printf 'running\n' > "$ROOT/BATCH_STATUS"
/usr/bin/time -v -o "$ROOT/TIMING.txt" "$PYTHON" -u \
  "$SOURCE/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_optimizer_comparison.py" \
  --bundle "$BUNDLE" --output "$ROOT/result" --source-commit "$COMMIT" \
  --continuation-root "$PARENT" --continuation-sha256 "$PARENT_SHA" \
  --max-steps 60 --max-forwards 120
"$PYTHON" - "$ROOT/result/RESULT.json" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1]))
assert r['status']=='success' and r['job_id']==os.environ['SLURM_JOB_ID']
assert r['scope']=='lbfgs_continuation' and set(r['arms'])=={'lbfgs'}
assert r['physical_release_gate']=='hold' and r['continuation']
assert r['shared_initial_forward']==r['shared_initial_backward']==0
assert r['arms']['lbfgs']['counts']['forward']<=120
PY
printf 'success\n' > "$ROOT/BATCH_STATUS"
