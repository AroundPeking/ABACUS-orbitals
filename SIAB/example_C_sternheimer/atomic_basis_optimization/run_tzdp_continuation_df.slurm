#!/bin/bash
#SBATCH --job-name=c_siab_conv
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=UNLIMITED
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${TARGET:?TARGET is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${PYTHON_EXE:?PYTHON_EXE is required}"
: "${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"

CAMPAIGN_DIR="$REPO_ROOT/SIAB/example_C_sternheimer/atomic_basis_optimization"
CAMPAIGN_SCRIPT="$CAMPAIGN_DIR/continuation_campaign.py"
SPECTRUM_SCRIPT="$CAMPAIGN_DIR/analyze_residual_spectrum.py"
OPTIMIZER="$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi/main.py"
REFERENCE="$REPO_ROOT/SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt"
ORBITAL="$REPO_ROOT/SG15_v1.0/Orbitals_v2.0/C_TZDP/C_gga_10au_100Ry_3s3p2d.orb"
OUTPUT="$CAMPAIGN_ROOT/runs/tzdp_continuation"

SOURCE_HEAD_FILE="$REPO_ROOT/.git/HEAD"
test -f "$SOURCE_HEAD_FILE" && test ! -L "$SOURCE_HEAD_FILE"
IFS= read -r source_head < "$SOURCE_HEAD_FILE"
source_ref=${source_head#ref: }
test "$source_ref" = "refs/heads/codex/c-pbe-reference-gate"
SOURCE_REF_FILE="$REPO_ROOT/.git/$source_ref"
test -f "$SOURCE_REF_FILE" && test ! -L "$SOURCE_REF_FILE"
IFS= read -r source_commit < "$SOURCE_REF_FILE"
test "$source_commit" = "$EXPECTED_SOURCE_COMMIT"
for path in "$CAMPAIGN_SCRIPT" "$SPECTRUM_SCRIPT" "$OPTIMIZER" \
  "$REFERENCE" "$ORBITAL" "$TARGET" "$CHECKPOINT"; do
  test -f "$path" && test ! -L "$path" && test -s "$path"
done
test -x "$PYTHON_EXE"
test ! -e "$OUTPUT/TZDP_CONTINUATION_RESULT.json"
test ! -e "$OUTPUT/RESIDUAL_SPECTRUM.json"
if test -d "$OUTPUT" && test -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)"; then
  echo "refusing nonempty output directory: $OUTPUT" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_MAX_ACTIVE_LEVELS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export PYTHONPATH="$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$OUTPUT/.mplconfig"

"$PYTHON_EXE" "$CAMPAIGN_SCRIPT" prepare \
  --target "$TARGET" \
  --checkpoint "$CHECKPOINT" \
  --reference "$REFERENCE" \
  --orbital "$ORBITAL" \
  --output "$OUTPUT"

"$PYTHON_EXE" - <<'PY' > "$OUTPUT/PYTHON_ENVIRONMENT.txt"
import sys
import numpy
import scipy
import torch
print("python", sys.version.replace("\n", " "))
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("torch_threads", torch.get_num_threads())
PY

cd "$OUTPUT"
/usr/bin/time -v "$PYTHON_EXE" "$OPTIMIZER" > run.log 2>&1
"$PYTHON_EXE" "$CAMPAIGN_SCRIPT" audit --output "$OUTPUT"
if grep -q '"status": "TZDP_CONVERGED"' "$OUTPUT/TZDP_CONTINUATION_RESULT.json"; then
  "$PYTHON_EXE" "$SPECTRUM_SCRIPT" \
    --coefficients "$OUTPUT/ORBITAL_RESULTS.txt" \
    --atom-target "$TARGET" \
    --output "$OUTPUT/ORBITAL_RESULTS.next_shell_seed.txt" \
    --report "$OUTPUT/RESIDUAL_SPECTRUM.json" \
    > "$OUTPUT/residual_spectrum.log" 2>&1
fi
