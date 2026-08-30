#!/bin/bash
#SBATCH --job-name=c-diff-trlog
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000M
#SBATCH --time=04:00:00

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SIAB_COMMIT:?SIAB_COMMIT is required}"

test -d "$REPO_ROOT/.git" || test -f "$REPO_ROOT/.git"
test "$(cat "$REPO_ROOT/.git/HEAD")" = "$SIAB_COMMIT"
test ! -e "$RUN_ROOT/result"
mkdir "$RUN_ROOT/result"

module load python/3.9.22
runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
python_exe=$(command -v python3)
export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages:$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-40}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

analyzer=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/analyze_atom_solid_differential_trace_log.py
data_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/runs/product-pca-20260825
solid_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q1=$solid_root/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
q2=$solid_root/c-diamond-g24-fd8-nfreq6-q2-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
q3=$solid_root/c-diamond-g24-fd8-nfreq6-q3-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q3_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
atomic_response=/data/home/df_iopcas_ghj/gw/260823/c-atom-siab-reference-fixedocc-p1-8cf890e8/OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat
atomic_source=$data_root/c-atom-source-only-9bd0795d/OUT.C_ATOM_SIAB_SOURCE/STERNHEIMER_SIAB_SOURCE_V1.dat
original=$data_root/balanced-atom-solid-no-f-production-1181149a/input/C_3s3p2d_joint_initial.txt
early=$data_root/balanced-atom-solid-no-f-pilot-1181149a/result/ORBITAL_RESULTS.txt
fixed500=$data_root/balanced-atom-solid-no-f-production-1181149a/result/ORBITAL_RESULTS.txt
relaxed=$data_root/balanced-atom-solid-relaxed-dzp-production-880e7150/result/ORBITAL_RESULTS.txt
status=$RUN_ROOT/result/STATUS.json
result=$RUN_ROOT/result/DIFFERENTIAL_TRACE_LOG.json
provenance=$RUN_ROOT/result/PROVENANCE.json

for path in "$analyzer" "$atomic_response" "$atomic_source" "$original" "$early" "$fixed500" "$relaxed"; do
  test -s "$path"
done
for path in "$q1" "$q2" "$q3"; do
  test -d "$path"
done

"$python_exe" - "$status" "$SIAB_COMMIT" <<'PY'
import json
import os
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "status": "running",
    "siab_commit": sys.argv[2],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

record_failure() {
  rc=$?
  if test "$rc" -ne 0; then
    "$python_exe" - "$status" "$SIAB_COMMIT" "$rc" <<'PY'
import json
import os
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "status": "failed",
    "siab_commit": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
  fi
}
trap record_failure EXIT

date -Ins
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "cpus=${SLURM_CPUS_PER_TASK:-40}"
"$python_exe" - <<'PY'
import torch
assert torch.__version__ == "1.9.1"
print("torch=" + torch.__version__)
PY

"$python_exe" -u "$analyzer" \
  --dataset "$q1" \
  --dataset "$q2" \
  --dataset "$q3" \
  --q-star-multiplicity 1:1 \
  --q-star-multiplicity 22:8 \
  --q-star-multiplicity 43:4 \
  --full-q-count 64 \
  --atomic-response "$atomic_response" \
  --atomic-source "$atomic_source" \
  --candidate "original:$original:3,3,2,0,0" \
  --candidate "early:$early:3,3,2,0,0" \
  --candidate "fixed500:$fixed500:3,3,2,0,0" \
  --candidate "relaxed:$relaxed:3,3,2,0,0" \
  --known-sos-binding-error-ev original:0.870347342670 \
  --known-sos-binding-error-ev fixed500:1.145074039398 \
  --known-sos-binding-error-ev relaxed:1.275306726142 \
  --occupied-capture-floor 0.9998 \
  --output "$result"

"$python_exe" - "$result" "$provenance" "$status" "$SIAB_COMMIT" "$analyzer" <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import torch

result_path, provenance_path, status_path, commit, analyzer_path = map(Path, sys.argv[1:6])
payload = json.loads(result_path.read_text(encoding="ascii"))
assert payload["format_version"] == 1
assert [item["label"] for item in payload["candidates"]] == [
    "original", "early", "fixed500", "relaxed"
]
for candidate in payload["candidates"]:
    assert math.isfinite(candidate["minimum_periodic_occupied_capture"])
    assert math.isfinite(candidate["atomic_maximum_overlap_condition"])
    for name in ("raw_q_weight", "star_partial", "star_normalized_extrapolation"):
        assert math.isfinite(
            candidate["proxies"][name]["atom_minus_half_solid_difference"]
        )

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

provenance_path.write_text(json.dumps({
    "status": "success",
    "siab_commit": str(commit),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "host": platform.node(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "omp_threads": os.environ.get("OMP_NUM_THREADS"),
    "analyzer": str(analyzer_path),
    "analyzer_sha256": sha256(analyzer_path),
    "result": str(result_path),
    "result_sha256": sha256(result_path),
}, indent=2, sort_keys=True) + "\n", encoding="ascii")
status_path.write_text(json.dumps({
    "status": "success",
    "siab_commit": str(commit),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "result_sha256": sha256(result_path),
}, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

trap - EXIT
date -Ins
