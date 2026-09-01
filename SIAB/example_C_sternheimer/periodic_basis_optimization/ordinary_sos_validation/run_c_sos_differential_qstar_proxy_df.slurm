#!/bin/bash
#SBATCH --job-name=c-sos-diff-qproxy
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=00:10:00

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SIAB_COMMIT:?SIAB_COMMIT is required}"

test -s "$REPO_ROOT/SOURCE_COMMIT"
test "$(cat "$REPO_ROOT/SOURCE_COMMIT")" = "$SIAB_COMMIT"
test ! -e "$RUN_ROOT"

module load python/3.9.22
validation_root=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
analyzer=$validation_root/c_sos_differential_qstar_proxy.py
manifest=$validation_root/c_sos_differential_qstar_proxy_current.json

test -s "$analyzer"
test -s "$manifest"
python3 "$analyzer" \
  --manifest "$manifest" \
  --output-root "$RUN_ROOT"

test "$(cat "$RUN_ROOT/STATUS")" = success
python3 - "$RUN_ROOT/RESULT.json" <<'PY'
import json
import math
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
assert result["status"] == "success"
assert result["proxy_gate"] in {"pass", "fail"}
assert result["q_indices_one_based"] == [6, 7, 8]
assert result["calibration_count"] == 13
assert math.isfinite(result["loo_max_abs_error_ev_per_c"])
assert math.isfinite(result["loo_rank_concordance"])
PY

{
  echo "slurm_job_id ${SLURM_JOB_ID:-none}"
  echo "source_commit $SIAB_COMMIT"
  echo "host $(hostname)"
} >> "$RUN_ROOT/provenance.txt"

echo C_SOS_DIFFERENTIAL_QSTAR_PROXY_OK root="$RUN_ROOT"
