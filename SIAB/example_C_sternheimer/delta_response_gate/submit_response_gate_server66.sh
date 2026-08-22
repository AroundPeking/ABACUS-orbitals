#!/bin/bash
set -euo pipefail

root=${DELTA_GATE_ROOT:?}
source_dir=${DELTA_GATE_SOURCE:?}
test -s "$root/PREPARATION_MANIFEST.json"
test -s "$source_dir/run_response_branch_server66.slurm"
test -s "$source_dir/run_librpa_branch_server66.slurm"
test ! -e "$root/RESPONSE_JOB_ID.txt"
test ! -e "$root/LIBRPA_JOB_ID.txt"
test ! -e "$root/.submission-claim"

campaign_id=$(printf '%s' "$root" | sha256sum | cut -c1-10)
response_name=cdr_${campaign_id}
librpa_name=cdl_${campaign_id}
mkdir -p "$root/logs"

queue_state=$(squeue -h -o '%i|%j|%T' 2>&1) || {
  echo "squeue failed; refusing an unobservable submission" >&2
  exit 2
}
history_state=$(sacct -n -X -S 2026-08-01 -o JobIDRaw,JobName,State 2>&1) || {
  echo "sacct failed; refusing an unobservable submission" >&2
  exit 2
}
if printf '%s\n%s\n' "$queue_state" "$history_state" | \
    grep -Eq "(^|[|[:space:]])(${response_name}|${librpa_name})([|[:space:]]|$)"; then
  echo "matching response-gate job already exists" >&2
  exit 3
fi

mkdir "$root/.submission-claim"
trap 'rm -rf "$root/.submission-claim"' EXIT
printf '%s\n' "$queue_state" > "$root/.submission-claim/SQUEUE_BEFORE.txt"
printf '%s\n' "$history_state" > "$root/.submission-claim/SACCT_BEFORE.txt"

response_receipt=$(sbatch --parsable \
  --job-name="$response_name" \
  --array=0-1%2 \
  --output="$root/logs/${response_name}-%A_%a.out" \
  --error="$root/logs/${response_name}-%A_%a.err" \
  --export=ALL,DELTA_GATE_ROOT="$root",DELTA_GATE_SOURCE="$source_dir",DELTA_GATE_SOURCE_COMMIT="${DELTA_GATE_SOURCE_COMMIT:?}",ABACUS_EXE="${ABACUS_EXE:?}",ABACUS_SHA256="${ABACUS_SHA256:?}",ABACUS_ENV_SCRIPT="${ABACUS_ENV_SCRIPT:?}",PYTHON_EXE="${PYTHON_EXE:-python3}" \
  "$source_dir/run_response_branch_server66.slurm")
response_job=${response_receipt%%;*}
test -n "$response_job"

librpa_receipt=$(sbatch --parsable \
  --dependency=afterok:"$response_job" \
  --job-name="$librpa_name" \
  --array=0-1%2 \
  --output="$root/logs/${librpa_name}-%A_%a.out" \
  --error="$root/logs/${librpa_name}-%A_%a.err" \
  --export=ALL,DELTA_GATE_ROOT="$root",LIBRPA_EXE="${LIBRPA_EXE:?}",LIBRPA_SHA256="${LIBRPA_SHA256:?}",ABACUS_ENV_SCRIPT="${ABACUS_ENV_SCRIPT:?}",PYTHON_EXE="${PYTHON_EXE:-python3}" \
  "$source_dir/run_librpa_branch_server66.slurm")
librpa_job=${librpa_receipt%%;*}
test -n "$librpa_job"

printf '%s\n' "$response_job" > "$root/.submission-claim/RESPONSE_JOB_ID.txt"
printf '%s\n' "$librpa_job" > "$root/.submission-claim/LIBRPA_JOB_ID.txt"
python3 - "$root" "$response_name" "$librpa_name" "$response_job" "$librpa_job" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, response_name, librpa_name, response_job, librpa_job = sys.argv[1:]
payload = {
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "root": root,
    "response_job_name": response_name,
    "response_job_id": response_job,
    "response_array": "0-1%2",
    "librpa_job_name": librpa_name,
    "librpa_job_id": librpa_job,
    "librpa_dependency": f"afterok:{response_job}",
}
Path(root, ".submission-claim", "SUBMISSION_PROVENANCE.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
)
PY

mv "$root/.submission-claim/RESPONSE_JOB_ID.txt" "$root/RESPONSE_JOB_ID.txt"
mv "$root/.submission-claim/LIBRPA_JOB_ID.txt" "$root/LIBRPA_JOB_ID.txt"
cp "$root/.submission-claim/SUBMISSION_PROVENANCE.json" "$root/SUBMISSION_PROVENANCE.json"
trap - EXIT
echo "response_job_id=$response_job"
echo "librpa_job_id=$librpa_job"
