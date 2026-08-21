#!/usr/bin/env bash

set -euo pipefail

MODULE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RUNNER="$MODULE_DIR/run_pbe_branch.slurm"
SUBMITTER="$MODULE_DIR/submit_pbe_gate.sh"
PYTHON_EXE=${PYTHON_EXE:-python3}

fail() {
    printf '%s\n' "$*" >&2
    exit 2
}

require_environment() {
    local name=$1
    [[ -n ${!name:-} ]] || fail "required environment variable is missing: $name"
}

for required in GATE_ROOT ABACUS_ARTIFACT PSEUDO_SOURCE ORBITAL_SOURCE; do
    require_environment "$required"
done

PYTHON_COMMAND=$(command -v -- "$PYTHON_EXE") \
    || fail "PYTHON_EXE is not executable: $PYTHON_EXE"

resolve_regular() {
    "$PYTHON_COMMAND" - "$1" "$2" "$3" "${4:-0}" <<'PY'
import os
import stat
import sys
from pathlib import Path

raw, kind, executable, allow_symlink = sys.argv[1:]
path = Path(raw).expanduser().absolute()
try:
    metadata = path.lstat()
except OSError as error:
    raise SystemExit(f"cannot inspect {kind}: {path}: {error}")
if stat.S_ISLNK(metadata.st_mode) and allow_symlink != "1":
    raise SystemExit(f"{kind} must be a non-symlink regular file: {path}")
resolved = path.resolve(strict=True)
resolved_metadata = resolved.stat()
if not stat.S_ISREG(resolved_metadata.st_mode) or resolved_metadata.st_size == 0:
    raise SystemExit(f"{kind} must be a nonempty regular file: {resolved}")
if executable == "1" and not os.access(resolved, os.X_OK):
    raise SystemExit(f"{kind} must be executable: {resolved}")
print(resolved)
PY
}

resolve_gate_root() {
    "$PYTHON_COMMAND" - "$1" <<'PY'
import os
import stat
import sys
from pathlib import Path

requested = Path(sys.argv[1]).expanduser().absolute()
if requested.name in {"", ".", ".."}:
    raise SystemExit(f"invalid GATE_ROOT: {requested}")
if os.path.lexists(requested):
    metadata = requested.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit(f"GATE_ROOT must be a non-symlink directory: {requested}")
    resolved = requested.resolve(strict=True)
else:
    parent = requested.parent.resolve(strict=True)
    resolved = parent / requested.name
    try:
        resolved.mkdir(mode=0o700)
    except FileExistsError:
        metadata = resolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SystemExit(
                f"GATE_ROOT must be a non-symlink directory: {resolved}"
            )
resolved_metadata = resolved.lstat()
if stat.S_ISLNK(resolved_metadata.st_mode) or not stat.S_ISDIR(resolved_metadata.st_mode):
    raise SystemExit(f"GATE_ROOT must be a non-symlink directory: {resolved}")
print(resolved)
PY
}

PYTHON_REAL=$(resolve_regular "$PYTHON_COMMAND" PYTHON_EXE 1 1)
ABACUS_REAL=$(resolve_regular "$ABACUS_ARTIFACT" ABACUS_ARTIFACT 1)
PSEUDO_REAL=$(resolve_regular "$PSEUDO_SOURCE" PSEUDO_SOURCE 0)
ORBITAL_REAL=$(resolve_regular "$ORBITAL_SOURCE" ORBITAL_SOURCE 0)
RUNNER_REAL=$(resolve_regular "$RUNNER" run_pbe_branch.slurm 0)
SUBMITTER_REAL=$(resolve_regular "$SUBMITTER" submit_pbe_gate.sh 0)
GATE_ROOT_REAL=$(resolve_gate_root "$GATE_ROOT")

for value in "$GATE_ROOT_REAL" "$ABACUS_REAL" "$PSEUDO_REAL" "$ORBITAL_REAL" "$PYTHON_REAL"; do
    [[ $value != *','* && $value != *$'\n'* && $value != *$'\r'* ]] \
        || fail "resolved export paths must not contain commas or newlines: $value"
done

SOURCE_COMMIT=$(git -C "$MODULE_DIR" rev-parse --verify HEAD) \
    || fail "cannot determine source commit"
[[ $SOURCE_COMMIT =~ ^[0-9a-f]{40}$ ]] || fail "invalid source commit: $SOURCE_COMMIT"
JOB_NAME=$(
    "$PYTHON_REAL" - "$GATE_ROOT_REAL" <<'PY'
import hashlib
import sys

digest = hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest()[:12]
print(f"c_pbe_gate_{digest}")
PY
)

check_gate_is_unstarted() {
    "$PYTHON_REAL" - "$GATE_ROOT_REAL" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
root_markers = (
    "SUBMITTED_JOB_ID.txt",
    "SUBMISSION_PROVENANCE.json",
    "PBE_GATE_PASSED",
    "PBE_GATE_FAILED",
    "DELTA_ST_GATE_PASSED",
    "RUN_FAILED.json",
    "RESULT_SUMMARY.json",
    "RESULT_SUMMARY.txt",
)
for name in root_markers:
    path = root / name
    if os.path.lexists(path):
        raise SystemExit(f"gate root already contains submission/result evidence: {path}")
claim = root / ".submission-claim"
if os.path.lexists(claim):
    raise SystemExit(f"gate root already has an immutable submission claim: {claim}")
runs = root / "runs"
if os.path.lexists(runs):
    metadata = runs.lstat()
    if runs.is_symlink() or not runs.is_dir():
        raise SystemExit(f"invalid runs path in gate root: {runs}")
    for branch in ("fixed", "dir0", "dir1", "dir2"):
        path = runs / branch
        if os.path.lexists(path):
            raise SystemExit(f"gate root already contains formal branch evidence: {path}")
PY
}

validate_scheduler_output() {
    "$PYTHON_REAL" - "$1" "$2" <<'PY'
import sys
from pathlib import Path

for label, filename in (("squeue", sys.argv[1]), ("sacct", sys.argv[2])):
    text = Path(filename).read_text()
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) != 3 or not fields[0].strip() or not fields[2].strip():
            raise SystemExit(f"ambiguous {label} record at line {number}: {raw!r}")
PY
}

query_scheduler() {
    local scope=$1 selector=$2 output_dir=$3 squeue_file sacct_file
    mkdir -p "$output_dir"
    squeue_file="$output_dir/${scope}.squeue.txt"
    sacct_file="$output_dir/${scope}.sacct.txt"
    if [[ $scope == job-id ]]; then
        squeue --jobs "$selector" --noheader --format='%i|%j|%T' >"$squeue_file" \
            || fail "squeue query failed; scheduler state is unobservable"
        sacct -X --jobs "$selector" --noheader --parsable2 \
            --format=JobIDRaw,JobName,State >"$sacct_file" \
            || fail "sacct query failed; scheduler state is unobservable"
    else
        squeue --name "$selector" --noheader --format='%i|%j|%T' >"$squeue_file" \
            || fail "squeue query failed; scheduler state is unobservable"
        sacct -X --name "$selector" --starttime 1970-01-01 --noheader --parsable2 \
            --format=JobIDRaw,JobName,State >"$sacct_file" \
            || fail "sacct query failed; scheduler state is unobservable"
    fi
    validate_scheduler_output "$squeue_file" "$sacct_file" \
        || fail "scheduler query returned ambiguous records"
    if [[ -s $squeue_file || -s $sacct_file ]]; then
        printf 'existing scheduler work blocks this immutable gate root:\n' >&2
        sed -n '1,20p' "$squeue_file" "$sacct_file" >&2
        exit 2
    fi
}

QUERY_DIR=$(mktemp -d "${TMPDIR:-/tmp}/c-pbe-submit-query.XXXXXX")
trap 'rm -rf -- "$QUERY_DIR"' EXIT

if [[ -e $GATE_ROOT_REAL/SUBMITTED_JOB_ID.txt || -L $GATE_ROOT_REAL/SUBMITTED_JOB_ID.txt ]]; then
    JOB_ID=$(
        "$PYTHON_REAL" - "$GATE_ROOT_REAL/SUBMITTED_JOB_ID.txt" <<'PY'
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = path.lstat()
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(f"SUBMITTED_JOB_ID.txt must be a non-symlink regular file: {path}")
value = path.read_text().strip()
if not value.isdigit() or value.startswith("0"):
    raise SystemExit(f"stored job ID is malformed: {value!r}")
print(value)
PY
    ) || exit 2
    query_scheduler job-id "$JOB_ID" "$QUERY_DIR/existing-job"
    fail "gate root has already been submitted as job $JOB_ID"
fi

query_scheduler job-name "$JOB_NAME" "$QUERY_DIR/preclaim"
check_gate_is_unstarted

CLAIM="$GATE_ROOT_REAL/.submission-claim"
if ! mkdir "$CLAIM" 2>/dev/null; then
    query_scheduler job-name "$JOB_NAME" "$QUERY_DIR/concurrent-claim"
    fail "another submitter already owns the immutable submission claim: $CLAIM"
fi

write_json_noreplace() {
    local target=$1 status=$2 message=$3
    "$PYTHON_REAL" - "$target" "$status" "$message" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": sys.argv[2],
    "message": sys.argv[3],
    "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
temporary = target.parent / f".{target.name}.tmp.{os.getpid()}"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    temporary.unlink(missing_ok=True)
PY
}

write_json_noreplace "$CLAIM/SUBMISSION_CLAIM.json" CLAIMED \
    "exclusive submission claim acquired before sbatch"

query_scheduler job-name "$JOB_NAME" "$CLAIM/pre_submit_scheduler"

EXPORT_MAP="ALL,GATE_ROOT=$GATE_ROOT_REAL,ABACUS_ARTIFACT=$ABACUS_REAL,PSEUDO_ASSET=$PSEUDO_REAL,ORBITAL_ASSET=$ORBITAL_REAL,PYTHON_EXE=$PYTHON_REAL"
SBATCH_COMMAND=(
    sbatch
    --parsable
    "--job-name=$JOB_NAME"
    --array=0-3
    "--export=$EXPORT_MAP"
    "$RUNNER_REAL"
)
RECEIPT="$CLAIM/SBATCH_RECEIPT.txt"
SBATCH_STDERR="$CLAIM/SBATCH_STDERR.txt"

set +e
"${SBATCH_COMMAND[@]}" >"$RECEIPT" 2>"$SBATCH_STDERR"
SBATCH_STATUS=$?
set -e
"$PYTHON_REAL" - "$RECEIPT" "$SBATCH_STDERR" <<'PY'
import os
import sys
from pathlib import Path

for filename in sys.argv[1:]:
    path = Path(filename)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
directory = os.open(Path(sys.argv[1]).parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY

if (( SBATCH_STATUS != 0 )); then
    write_json_noreplace "$CLAIM/SUBMISSION_AMBIGUOUS.json" SUBMISSION_AMBIGUOUS \
        "sbatch exited $SBATCH_STATUS; receipt is retained and this gate root must not be resubmitted"
    fail "sbatch failed or became ambiguous after the durable claim; do not retry this gate root"
fi

JOB_ID=$(
    "$PYTHON_REAL" - "$RECEIPT" <<'PY'
import re
import sys
from pathlib import Path

receipt = Path(sys.argv[1]).read_text().strip()
match = re.fullmatch(r"([1-9][0-9]*)(?:;[^;\s]+)?", receipt)
if match is None:
    raise SystemExit(f"malformed sbatch receipt: {receipt!r}")
print(match.group(1))
PY
) || {
    write_json_noreplace "$CLAIM/SUBMISSION_AMBIGUOUS.json" SUBMISSION_AMBIGUOUS \
        "sbatch returned success but its durable receipt has no unique numeric job ID"
    fail "sbatch receipt is ambiguous; do not retry this gate root"
}

SUBMITTED_AT_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
PROVENANCE_TEMP="$CLAIM/SUBMISSION_PROVENANCE.json"
JOB_ID_TEMP="$CLAIM/SUBMITTED_JOB_ID.txt"

"$PYTHON_REAL" - \
    "$PROVENANCE_TEMP" "$JOB_ID_TEMP" "$GATE_ROOT_REAL" "$ABACUS_REAL" \
    "$PSEUDO_REAL" "$ORBITAL_REAL" "$PYTHON_REAL" "$RUNNER_REAL" \
    "$SUBMITTER_REAL" "$RECEIPT" "$SOURCE_COMMIT" "$JOB_ID" "$JOB_NAME" \
    "$SUBMITTED_AT_UTC" "${SBATCH_COMMAND[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    provenance_path,
    job_id_path,
    gate_root,
    abacus,
    pseudo,
    orbital,
    python,
    runner,
    submitter,
    receipt,
    source_commit,
    job_id,
    job_name,
    submitted_at,
    *command,
) = sys.argv[1:]

def file_record(filename):
    path = Path(filename)
    data = path.read_bytes()
    return {
        "path": str(path),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

payload = {
    "schema_version": 1,
    "status": "SUBMITTED",
    "job_id": job_id,
    "job_name": job_name,
    "submitted_at_utc": submitted_at,
    "source_commit": source_commit,
    "command": command,
    "resolved_paths": {
        "gate_root": gate_root,
        "abacus_artifact": abacus,
        "pseudo_source": pseudo,
        "orbital_source": orbital,
        "python_exe": python,
        "runner": runner,
        "submitter": submitter,
    },
    "runner_environment": {
        "GATE_ROOT": gate_root,
        "ABACUS_ARTIFACT": abacus,
        "PSEUDO_ASSET": pseudo,
        "ORBITAL_ASSET": orbital,
        "PYTHON_EXE": python,
    },
    "files": {
        "abacus": file_record(abacus),
        "pseudo": file_record(pseudo),
        "orbital": file_record(orbital),
        "python": file_record(python),
        "runner": file_record(runner),
        "submitter": file_record(submitter),
        "sbatch_receipt": file_record(receipt),
    },
}

def write_new(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())

write_new(provenance_path, (json.dumps(payload, sort_keys=True) + "\n").encode())
write_new(job_id_path, f"{job_id}\n".encode())
directory = os.open(Path(provenance_path).parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY

"$PYTHON_REAL" - "$PROVENANCE_TEMP" "$GATE_ROOT_REAL/SUBMISSION_PROVENANCE.json" \
    "$JOB_ID_TEMP" "$GATE_ROOT_REAL/SUBMITTED_JOB_ID.txt" <<'PY'
import os
import sys
from pathlib import Path

source_provenance, target_provenance, source_id, target_id = map(Path, sys.argv[1:])
os.link(source_provenance, target_provenance)
os.link(source_id, target_id)
directory = os.open(target_provenance.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY

printf 'submitted C PBE reference gate as Slurm array job %s\n' "$JOB_ID"
printf 'submission provenance: %s\n' "$GATE_ROOT_REAL/SUBMISSION_PROVENANCE.json"
