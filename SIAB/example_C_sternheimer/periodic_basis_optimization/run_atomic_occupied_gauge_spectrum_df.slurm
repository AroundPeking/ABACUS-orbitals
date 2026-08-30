#!/bin/bash
#SBATCH --job-name=c-atom-gauge
#SBATCH --partition=48cp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32000M
#SBATCH --time=02:00:00

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SIAB_COMMIT:?SIAB_COMMIT is required}"

test -d "$REPO_ROOT/.git"
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

data_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/runs/product-pca-20260825
response_root=/data/home/df_iopcas_ghj/gw/260823/c-atom-siab-reference-fixedocc-p1-8cf890e8/OUT.C_SIAB_REFERENCE
source_root=$data_root/c-atom-source-only-9bd0795d/OUT.C_ATOM_SIAB_SOURCE
response=$data_root/atom-projected-pi-lowfreq-spectrum-35442f7f/result/LOWEST_FREQUENCY_RESPONSE.dat
source=$source_root/STERNHEIMER_SIAB_SOURCE_V1.dat
analyzer=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/analyze_atomic_occupied_gauge_spectrum.py
report=$RUN_ROOT/result/ATOMIC_OCCUPIED_GAUGE_SPECTRUM.json
status=$RUN_ROOT/result/STATUS.json
provenance=$RUN_ROOT/result/PROVENANCE.json

for path in \
  "$response" "$source" "$analyzer" \
  "$response_root/wfs1_nao.txt" "$response_root/wfs2_nao.txt" \
  "$source_root/wfs1_nao.txt" "$source_root/wfs2_nao.txt"; do
  test -s "$path"
done

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
"$python_exe" - <<'PY'
import torch
assert torch.__version__ == "1.9.1"
print("torch=" + torch.__version__)
PY

"$python_exe" -u "$analyzer" \
  --response "$response" \
  --source "$source" \
  --response-wfs "$response_root/wfs1_nao.txt" \
  --response-wfs "$response_root/wfs2_nao.txt" \
  --source-wfs "$source_root/wfs1_nao.txt" \
  --source-wfs "$source_root/wfs2_nao.txt" \
  --output "$report"

"$python_exe" - "$report" "$provenance" "$status" "$SIAB_COMMIT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

report, provenance, status, commit = map(Path, sys.argv[1:5])
payload = json.loads(report.read_text(encoding="ascii"))
assert payload["status"] == "success"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

provenance.write_text(json.dumps({
    "status": "success",
    "siab_commit": str(commit),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "host": platform.node(),
    "report_sha256": sha256(report),
}, indent=2, sort_keys=True) + "\n", encoding="ascii")
status.write_text(json.dumps({
    "status": "success",
    "siab_commit": str(commit),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "uncorrected_i_minus_pi_positive": payload[
        "uncorrected_i_minus_pi_positive"
    ],
    "gauge_corrected_i_minus_pi_positive": payload[
        "gauge_corrected_i_minus_pi_positive"
    ],
}, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

trap - EXIT
date -Ins
