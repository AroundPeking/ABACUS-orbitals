#!/usr/bin/env bash

set -euo pipefail
set -E

PYTHON_EXE=${PYTHON_EXE:-python3}

require_environment() {
    local name=$1
    if [[ -z ${!name:-} ]]; then
        printf 'required environment variable is missing: %s\n' "$name" >&2
        return 2
    fi
}

for required in \
    C_PBE_GATE_PROFILE C_PBE_GATE_ENTRYPOINT C_PBE_GATE_COMMON_RUNNER \
    GATE_ROOT ABACUS_ARTIFACT ABACUS_ENV_SCRIPT PSEUDO_ASSET ORBITAL_ASSET
do
    require_environment "$required"
done

resolve_regular() {
    "$PYTHON_EXE" - "$1" "$2" "${3:-0}" "${4:-0}" <<'PY'
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
if allow_symlink != "1" and resolved != path:
    raise SystemExit(f"{kind} path must already be resolved: {path}")
print(resolved)
PY
}

PYTHON_COMMAND=$(command -v -- "$PYTHON_EXE") || {
    printf 'PYTHON_EXE is not executable: %s\n' "$PYTHON_EXE" >&2
    exit 2
}
PYTHON_REAL=$(resolve_regular "$PYTHON_COMMAND" PYTHON_EXE 1 1)
COMMON_RUNNER_REAL=$(resolve_regular "$C_PBE_GATE_COMMON_RUNNER" run_pbe_branch_common.sh)
MODULE_DIR=$(cd -- "$(dirname -- "$COMMON_RUNNER_REAL")" && pwd -P)
PREPARE="$MODULE_DIR/prepare_gate.py"
AUDIT="$MODULE_DIR/audit_gate.py"
GATE_CONTRACT="$MODULE_DIR/gate_contract.py"
RESOURCE_PROFILES="$MODULE_DIR/resource_profiles.py"
PREPARE_REAL=$(resolve_regular "$PREPARE" prepare_gate.py)
AUDIT_REAL=$(resolve_regular "$AUDIT" audit_gate.py)
GATE_CONTRACT_REAL=$(resolve_regular "$GATE_CONTRACT" gate_contract.py)
RESOURCE_PROFILES_REAL=$(resolve_regular "$RESOURCE_PROFILES" resource_profiles.py)
ENTRYPOINT_REAL=$(resolve_regular "$C_PBE_GATE_ENTRYPOINT" C_PBE_GATE_ENTRYPOINT)
ABACUS_ENV_REAL=$(resolve_regular "$ABACUS_ENV_SCRIPT" ABACUS_ENV_SCRIPT)
ABACUS_REAL=$(resolve_regular "$ABACUS_ARTIFACT" ABACUS 1)
PSEUDO_REAL=$(resolve_regular "$PSEUDO_ASSET" pseudo)
ORBITAL_REAL=$(resolve_regular "$ORBITAL_ASSET" orbital)

PROFILE_CONTRACT=$(
    "$PYTHON_REAL" "$RESOURCE_PROFILES_REAL" shell "$C_PBE_GATE_PROFILE"
)
IFS='|' read -r \
    PROFILE_NAME PROFILE_PARTITION PROFILE_NODES PROFILE_NTASKS \
    PROFILE_CPUS_PER_TASK PROFILE_MEMORY_MB PROFILE_TIME_LIMIT \
    PROFILE_OVER_SUBSCRIBE <<<"$PROFILE_CONTRACT"
[[ $PROFILE_NAME == "$C_PBE_GATE_PROFILE" ]] || {
    printf 'resource profile identity mismatch: %s\n' "$C_PBE_GATE_PROFILE" >&2
    exit 2
}
PROFILE_TASKS_PER_NODE=$((PROFILE_NTASKS / PROFILE_NODES))
PROFILE_TASKS_PER_NODE_COMPACT="${PROFILE_TASKS_PER_NODE}(x${PROFILE_NODES})"

# The selected artifact owns its runtime stack through this one recorded script.
source "$ABACUS_ENV_REAL"

# Environment scripts may set threading defaults; the selected profile owns them.
set -euo pipefail
set -E
export OMP_NUM_THREADS=$PROFILE_CPUS_PER_TASK
export MKL_NUM_THREADS=$PROFILE_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$PROFILE_CPUS_PER_TASK

[[ ${SLURM_JOB_PARTITION:-} == "$PROFILE_PARTITION" ]] || {
    printf 'partition must be %s\n' "$PROFILE_PARTITION" >&2
    exit 2
}
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || {
    printf 'job ID must be numeric\n' >&2
    exit 2
}
[[ ${SLURM_ARRAY_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || {
    printf 'array job ID must be numeric\n' >&2
    exit 2
}
[[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-3]$ ]] || {
    printf 'array task must be 0-3\n' >&2
    exit 2
}
[[ ${SLURM_ARRAY_TASK_COUNT:-} == 4 ]] || {
    printf 'array task count must be 4\n' >&2
    exit 2
}
[[ ${SLURM_CPUS_PER_TASK:-} == "$PROFILE_CPUS_PER_TASK" ]] || {
    printf 'cpus per task must be %s\n' "$PROFILE_CPUS_PER_TASK" >&2
    exit 2
}
[[ ${SLURM_NTASKS:-} == "$PROFILE_NTASKS" ]] || {
    printf 'ntasks must be %s\n' "$PROFILE_NTASKS" >&2
    exit 2
}
[[ ${SLURM_JOB_NUM_NODES:-} == "$PROFILE_NODES" ]] || {
    printf 'node count must be %s\n' "$PROFILE_NODES" >&2
    exit 2
}
[[ ${SLURM_TASKS_PER_NODE:-} == "$PROFILE_TASKS_PER_NODE" \
    || ${SLURM_TASKS_PER_NODE:-} == "$PROFILE_TASKS_PER_NODE_COMPACT" ]] || {
    printf 'tasks per node must be %s\n' "$PROFILE_TASKS_PER_NODE" >&2
    exit 2
}
[[ ${SLURM_MEM_PER_NODE:-} == "$PROFILE_MEMORY_MB" ]] || {
    printf 'memory per node must be %s MB\n' "$PROFILE_MEMORY_MB" >&2
    exit 2
}

case "$SLURM_ARRAY_TASK_ID" in
    0) BRANCH=fixed ;;
    1) BRANCH=dir0 ;;
    2) BRANCH=dir1 ;;
    3) BRANCH=dir2 ;;
    *) printf 'invalid array task: %s\n' "$SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

MPIRUN_COMMAND=$(command -v -- mpirun) || {
    printf 'mpirun is unavailable after sourcing ABACUS_ENV_SCRIPT\n' >&2
    exit 2
}
MPIRUN_REAL=$(resolve_regular "$MPIRUN_COMMAND" mpirun 1 1)
"$PYTHON_REAL" "$AUDIT_REAL" check-scheduler --branch "$BRANCH" >/dev/null
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
PREPARE_MUTEX="$PREPARE_GUARD/prepare.lock"
PREPARE_MUTEX_HELD=0

owner_matches_this_array() {
    local owner=$1
    [[ -f $owner && ! -L $owner ]] || return 1
    grep -qx "array_job_id=${SLURM_ARRAY_JOB_ID}" "$owner" \
        && grep -Eq '^array_task_id=[0-3]$' "$owner"
}

write_owner() {
    local directory=$1 temporary_owner
    temporary_owner="$directory/.owner.$$"
    {
        printf 'array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
        printf 'array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'pid=%s\n' "$$"
    } >"$temporary_owner"
    mv "$temporary_owner" "$directory/owner"
}

ensure_prepare_guard() {
    local attempt
    for attempt in $(seq 1 120); do
        if mkdir "$PREPARE_GUARD" 2>/dev/null; then
            if ! write_owner "$PREPARE_GUARD"; then
                rm -f "$PREPARE_GUARD/.owner.$$"
                rmdir "$PREPARE_GUARD"
                return 1
            fi
            return 0
        fi
        [[ -d $PREPARE_GUARD && ! -L $PREPARE_GUARD ]] || {
            printf 'prepare guard is not a local directory: %s\n' "$PREPARE_GUARD" >&2
            return 1
        }
        if [[ ! -e $PREPARE_GUARD/owner && ! -L $PREPARE_GUARD/owner ]]; then
            sleep 1
            continue
        fi
        if owner_matches_this_array "$PREPARE_GUARD/owner"; then
            return 0
        fi
        printf 'prepare guard belongs to another array job: %s\n' "$PREPARE_GUARD" >&2
        return 1
    done
    printf 'timed out waiting for Task4 prepare guard identity\n' >&2
    return 1
}

acquire_prepare_mutex() {
    local attempt
    for attempt in $(seq 1 120); do
        if mkdir "$PREPARE_MUTEX" 2>/dev/null; then
            PREPARE_MUTEX_HELD=1
            if ! write_owner "$PREPARE_MUTEX"; then
                rm -f "$PREPARE_MUTEX/.owner.$$"
                rmdir "$PREPARE_MUTEX"
                PREPARE_MUTEX_HELD=0
                return 1
            fi
            return 0
        fi
        if [[ ! -e $PREPARE_MUTEX && ! -L $PREPARE_MUTEX ]]; then
            sleep 1
            continue
        fi
        [[ -d $PREPARE_MUTEX && ! -L $PREPARE_MUTEX ]] || {
            printf 'prepare mutex is not a local directory: %s\n' "$PREPARE_MUTEX" >&2
            return 1
        }
        if [[ -e $PREPARE_MUTEX/owner || -L $PREPARE_MUTEX/owner ]]; then
            if owner_matches_this_array "$PREPARE_MUTEX/owner"; then
                sleep 1
                continue
            fi
            if [[ ! -e $PREPARE_MUTEX || ( ! -e $PREPARE_MUTEX/owner && ! -L $PREPARE_MUTEX/owner ) ]]; then
                sleep 1
                continue
            fi
            printf 'prepare mutex belongs to another array job: %s\n' "$PREPARE_MUTEX" >&2
            return 1
        fi
        sleep 1
    done
    printf 'timed out waiting for Task4 prepare mutex\n' >&2
    return 1
}

release_prepare_mutex() {
    if (( PREPARE_MUTEX_HELD )); then
        rm -f "$PREPARE_MUTEX/owner"
        rmdir "$PREPARE_MUTEX"
        PREPARE_MUTEX_HELD=0
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
        if "$PYTHON_REAL" "$PREPARE_REAL" prepare \
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

ensure_prepare_guard
acquire_prepare_mutex
if prepare_branch_once; then
    release_prepare_mutex
else
    prepare_status=$?
    release_prepare_mutex
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
    "job_id": os.environ["SLURM_JOB_ID"],
    "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
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

"$PYTHON_REAL" "$AUDIT_REAL" runner-init \
    --root "$GATE_ROOT" --branch "$BRANCH" \
    --gate-profile "$C_PBE_GATE_PROFILE" \
    --python "$PYTHON_REAL" \
    --prepare-gate "$PREPARE_REAL" \
    --audit-gate "$AUDIT_REAL" \
    --gate-contract "$GATE_CONTRACT_REAL" \
    --resource-profiles "$RESOURCE_PROFILES_REAL" \
    --entrypoint "$ENTRYPOINT_REAL" \
    --common-runner "$COMMON_RUNNER_REAL" \
    --abacus "$ABACUS_REAL" \
    --environment-script "$ABACUS_ENV_REAL" --mpirun "$MPIRUN_REAL"

BRANCH_STARTED_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
BRANCH_STARTED_EPOCH=$(date '+%s')

run_phase() {
    local phase=$1 phase_root started_utc started_epoch ended_utc ended_epoch wall
    phase_root="$BRANCH_ROOT/$phase"
    "$PYTHON_REAL" "$AUDIT_REAL" preflight-phase \
        --root "$GATE_ROOT" --branch "$BRANCH" --phase "$phase"
    started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    started_epoch=$(date '+%s')
    (
        cd "$phase_root"
        "$MPIRUN_REAL" -np 1 -ppn 1 "$ABACUS_REAL" \
            >abacus.stdout 2>abacus.stderr
    )
    ended_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    ended_epoch=$(date '+%s')
    wall=$((ended_epoch - started_epoch))
    "$PYTHON_REAL" "$AUDIT_REAL" complete-phase \
        --root "$GATE_ROOT" --branch "$BRANCH" --phase "$phase" \
        --started-utc "$started_utc" --ended-utc "$ended_utc" \
        --wall-seconds "$wall"
}

prepare_restart() {
    local source=$1 destination=$2 input expected
    "$PYTHON_REAL" "$AUDIT_REAL" create-restart \
        --root "$GATE_ROOT" --branch "$BRANCH" \
        --source "$source" --destination "$destination"
    input="$BRANCH_ROOT/$destination/INPUT"
    if [[ $BRANCH == fixed ]]; then
        expected=$("$PYTHON_REAL" "$PREPARE_REAL" render --mode fixed --restart)
    else
        expected=$("$PYTHON_REAL" "$PREPARE_REAL" render --mode free --field-dir "${BRANCH#dir}" --restart)
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
"$PYTHON_REAL" "$AUDIT_REAL" complete-branch \
    --root "$GATE_ROOT" --branch "$BRANCH" \
    --started-utc "$BRANCH_STARTED_UTC" --ended-utc "$BRANCH_ENDED_UTC" \
    --wall-seconds "$BRANCH_WALL"
trap - ERR
