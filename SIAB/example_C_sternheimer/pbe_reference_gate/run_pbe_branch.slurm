#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=126500M
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

set -euo pipefail
set -E

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

MODULE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PREPARE="$MODULE_DIR/prepare_gate.py"
AUDIT="$MODULE_DIR/audit_gate.py"
RUNNER="$MODULE_DIR/run_pbe_branch.slurm"
PYTHON_EXE=${PYTHON_EXE:-python3}

require_environment() {
    local name=$1
    if [[ -z ${!name:-} ]]; then
        printf 'required environment variable is missing: %s\n' "$name" >&2
        return 2
    fi
}

for required in GATE_ROOT ABACUS_ARTIFACT PSEUDO_ASSET ORBITAL_ASSET; do
    require_environment "$required"
done

[[ ${SLURM_JOB_PARTITION:-} == normal ]] || { printf 'partition must be normal\n' >&2; exit 2; }
[[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-3]$ ]] || { printf 'array task must be 0-3\n' >&2; exit 2; }
[[ ${SLURM_ARRAY_TASK_COUNT:-} == 4 ]] || { printf 'array task count must be 4\n' >&2; exit 2; }
[[ ${SLURM_CPUS_PER_TASK:-} == 32 ]] || { printf 'cpus per task must be 32\n' >&2; exit 2; }
[[ ${SLURM_NTASKS:-} == 1 ]] || { printf 'ntasks must be 1\n' >&2; exit 2; }
[[ ${SLURM_JOB_NUM_NODES:-} == 1 ]] || { printf 'node count must be 1\n' >&2; exit 2; }
[[ ${SLURM_TASKS_PER_NODE:-} == 1 || ${SLURM_TASKS_PER_NODE:-} == '1(x1)' ]] || {
    printf 'tasks per node must be 1\n' >&2
    exit 2
}
if [[ -n ${SLURM_MEM_PER_NODE:-} ]]; then
    [[ $SLURM_MEM_PER_NODE == 126500 ]] || { printf 'memory per node must be 126500 MB\n' >&2; exit 2; }
fi

case "$SLURM_ARRAY_TASK_ID" in
    0) BRANCH=fixed ;;
    1) BRANCH=dir0 ;;
    2) BRANCH=dir1 ;;
    3) BRANCH=dir2 ;;
    *) printf 'invalid array task: %s\n' "$SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

resolve_regular() {
    "$PYTHON_EXE" - "$1" "$2" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().absolute()
kind = sys.argv[2]
metadata = path.lstat()
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(f"{kind} must be a non-symlink regular file: {path}")
resolved = path.resolve(strict=True)
if resolved != path:
    raise SystemExit(f"{kind} path must already be resolved: {path}")
if metadata.st_size == 0:
    raise SystemExit(f"{kind} must be nonempty: {path}")
if kind == "ABACUS" and not os.access(path, os.X_OK):
    raise SystemExit(f"ABACUS is not executable: {path}")
print(path)
PY
}

ABACUS_REAL=$(resolve_regular "$ABACUS_ARTIFACT" ABACUS)
PSEUDO_REAL=$(resolve_regular "$PSEUDO_ASSET" pseudo)
ORBITAL_REAL=$(resolve_regular "$ORBITAL_ASSET" orbital)
PYTHON_REAL=$(command -v "$PYTHON_EXE")
GATE_ROOT=$(
    "$PYTHON_REAL" - "$GATE_ROOT" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().absolute()
if os.path.lexists(path):
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit(f"GATE_ROOT must be a non-symlink directory: {path}")
print(path)
PY
)
mkdir -p "$GATE_ROOT"
PREPARE_GUARD="$GATE_ROOT/.task4-prepare.guard"
PREPARE_GUARD_HELD=0

guard_owner_matches_this_array() {
    local owner="$PREPARE_GUARD/owner"
    [[ -f $owner && ! -L $owner ]] || return 1
    grep -qx "array_job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-UNKNOWN}}" "$owner" \
        && grep -Eq '^array_task_id=[0-3]$' "$owner"
}

acquire_prepare_guard() {
    local attempt delay temporary_owner
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
        if mkdir "$PREPARE_GUARD" 2>/dev/null; then
            PREPARE_GUARD_HELD=1
            temporary_owner="$PREPARE_GUARD/.owner.$$"
            if ! {
                {
                    printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-UNKNOWN}}"
                    printf 'array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
                    printf 'hostname=%s\n' "$(hostname)"
                    printf 'pid=%s\n' "$$"
                } >"$temporary_owner" \
                    && mv "$temporary_owner" "$PREPARE_GUARD/owner"
            }; then
                rm -f "$temporary_owner"
                rmdir "$PREPARE_GUARD"
                PREPARE_GUARD_HELD=0
                return 1
            fi
            return 0
        fi
        [[ -d $PREPARE_GUARD && ! -L $PREPARE_GUARD ]] || {
            printf 'prepare guard is not a local directory: %s\n' "$PREPARE_GUARD" >&2
            return 1
        }
        if guard_owner_matches_this_array; then
            delay=$((attempt < 4 ? attempt : 4))
            sleep "$delay"
            continue
        fi
        if [[ ! -e $PREPARE_GUARD/owner && ! -L $PREPARE_GUARD/owner ]] \
            && (( attempt <= 2 )); then
            sleep "$attempt"
            continue
        fi
        printf 'prepare guard is stale or belongs to another job: %s\n' "$PREPARE_GUARD" >&2
        return 1
    done
    printf 'timed out waiting for active Task4 prepare guard\n' >&2
    return 1
}

release_prepare_guard() {
    if (( PREPARE_GUARD_HELD )); then
        rm -f "$PREPARE_GUARD/owner"
        rmdir "$PREPARE_GUARD"
        PREPARE_GUARD_HELD=0
    fi
}

active_prepare_lock() {
    local lock payload pid
    shopt -s nullglob
    for lock in "$GATE_ROOT"/runs/.*.prepare.lock; do
        [[ -f $lock && ! -L $lock ]] || continue
        payload=$(<"$lock")
        if [[ $payload =~ pid=([0-9]+) ]]; then
            pid=${BASH_REMATCH[1]}
            if kill -0 "$pid" 2>/dev/null; then
                shopt -u nullglob
                return 0
            fi
        fi
    done
    shopt -u nullglob
    return 1
}

prepare_branch_once() {
    local attempt error_file
    error_file=$(mktemp "${TMPDIR:-/tmp}/c-pbe-prepare.XXXXXX")
    for attempt in 1 2 3 4 5 6; do
        if "$PYTHON_REAL" "$PREPARE" prepare \
            --root "$GATE_ROOT" --branch "$BRANCH" \
            --pseudo "$PSEUDO_REAL" --orbital "$ORBITAL_REAL" \
            2>"$error_file"; then
            rm -f "$error_file"
            return 0
        fi
        if grep -q 'active or stale preparation lock exists' "$error_file" \
            && active_prepare_lock && (( attempt < 6 )); then
            sleep "$attempt"
            continue
        fi
        cat "$error_file" >&2
        rm -f "$error_file"
        return 1
    done
    rm -f "$error_file"
    return 1
}

acquire_prepare_guard
if prepare_branch_once; then
    release_prepare_guard
else
    prepare_status=$?
    release_prepare_guard
    exit "$prepare_status"
fi
BRANCH_ROOT="$GATE_ROOT/runs/$BRANCH"

record_failure() {
    local exit_code=$1 line=$2 command=$3
    trap - ERR
    set +e
    "$PYTHON_REAL" - "$BRANCH_ROOT/RUN_FAILED.json" "$exit_code" "$line" "$command" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "schema": "c-pbe-reference-gate-run-failure",
    "version": 1,
    "status": "RUN_FAILED",
    "exit_code": int(sys.argv[2]),
    "line": int(sys.argv[3]),
    "command": sys.argv[4],
    "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "job_id": os.environ.get("SLURM_JOB_ID", "UNKNOWN"),
    "array_task_id": int(os.environ["SLURM_ARRAY_TASK_ID"]),
}
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".RUN_FAILED.", suffix=".tmp")
try:
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, path)
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY
    exit "$exit_code"
}
trap 'record_failure "$?" "$LINENO" "$BASH_COMMAND"' ERR

"$PYTHON_REAL" "$AUDIT" runner-init \
    --root "$GATE_ROOT" --branch "$BRANCH" \
    --abacus "$ABACUS_REAL" --runner "$RUNNER"

BRANCH_STARTED_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
BRANCH_STARTED_EPOCH=$(date '+%s')

run_phase() {
    local phase=$1 phase_root started_utc started_epoch ended_utc ended_epoch wall
    phase_root="$BRANCH_ROOT/$phase"
    "$PYTHON_REAL" "$AUDIT" preflight-phase \
        --root "$GATE_ROOT" --branch "$BRANCH" --phase "$phase"
    started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    started_epoch=$(date '+%s')
    (
        cd "$phase_root"
        mpirun -np 1 -ppn 1 "$ABACUS_REAL" >abacus.stdout 2>abacus.stderr
    )
    ended_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    ended_epoch=$(date '+%s')
    wall=$((ended_epoch - started_epoch))
    "$PYTHON_REAL" "$AUDIT" complete-phase \
        --root "$GATE_ROOT" --branch "$BRANCH" --phase "$phase" \
        --started-utc "$started_utc" --ended-utc "$ended_utc" \
        --wall-seconds "$wall"
}

prepare_restart() {
    local source=$1 destination=$2 input expected
    "$PYTHON_REAL" "$AUDIT" create-restart \
        --root "$GATE_ROOT" --branch "$BRANCH" \
        --source "$source" --destination "$destination"
    input="$BRANCH_ROOT/$destination/INPUT"
    if [[ $BRANCH == fixed ]]; then
        expected=$("$PYTHON_REAL" "$PREPARE" render --mode fixed --restart)
    else
        expected=$("$PYTHON_REAL" "$PREPARE" render --mode free --field-dir "${BRANCH#dir}" --restart)
        grep -q "^ocp 0$" "$input"
        grep -q "^efield_flag 0$" "$input"
        grep -q "^init_wfc file$" "$input"
        grep -q "^init_chg file$" "$input"
    fi
    [[ $(<"$input") == "$expected" ]] || {
        printf 'rendered restart INPUT differs from the frozen template: %s\n' "$input" >&2
        return 1
    }
}

if [[ $BRANCH == fixed ]]; then
    run_phase fixed_cold
    prepare_restart fixed_cold fixed_restart
    run_phase fixed_restart
else
    run_phase field_seed
    prepare_restart field_seed free_restart1
    run_phase free_restart1
    prepare_restart free_restart1 free_restart2
    run_phase free_restart2
fi

BRANCH_ENDED_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
BRANCH_ENDED_EPOCH=$(date '+%s')
BRANCH_WALL=$((BRANCH_ENDED_EPOCH - BRANCH_STARTED_EPOCH))
"$PYTHON_REAL" "$AUDIT" complete-branch \
    --root "$GATE_ROOT" --branch "$BRANCH" \
    --started-utc "$BRANCH_STARTED_UTC" --ended-utc "$BRANCH_ENDED_UTC" \
    --wall-seconds "$BRANCH_WALL"
trap - ERR
