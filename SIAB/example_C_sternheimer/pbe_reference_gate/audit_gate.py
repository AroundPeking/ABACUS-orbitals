#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

if __package__:
    from .gate_contract import (
        FROZEN_PROTOCOL,
        HA_TO_KCAL_MOL,
        PhaseResult,
        audit_phase,
        compare_zero_field_results,
        render_input,
    )
    from . import prepare_gate
    from .resource_profiles import get_resource_profile
else:
    from gate_contract import (
        FROZEN_PROTOCOL,
        HA_TO_KCAL_MOL,
        PhaseResult,
        audit_phase,
        compare_zero_field_results,
        render_input,
    )
    import prepare_gate
    from resource_profiles import get_resource_profile


AUTHORITATIVE_RESULT = "RESULT_SUMMARY.json"
TEXT_SUMMARY = "RESULT_SUMMARY.txt"
RESTART_CHAIN_STATUS = "PENDING_TASK4"
RESTART_CHAIN_NOTE = (
    "Task2 verifies INPUT restart semantics only; actual WFC/CHG copy and load "
    "provenance must be verified by the Task4 runner before production use."
)
RESTART_FILES = ("wfs1_nao.txt", "wfs2_nao.txt", "chgs1.cube", "chgs2.cube")
BRANCH_PHASES = {
    "fixed": ("fixed_cold", "fixed_restart"),
    "dir0": ("field_seed", "free_restart1", "free_restart2"),
    "dir1": ("field_seed", "free_restart1", "free_restart2"),
    "dir2": ("field_seed", "free_restart1", "free_restart2"),
}
RUNTIME_RECORD_SPECS = {
    "python": ("Python interpreter", True),
    "prepare_gate": ("prepare_gate.py source", False),
    "audit_gate": ("audit_gate.py source", False),
    "gate_contract": ("gate_contract.py source", False),
    "resource_profiles": ("resource profile module", False),
    "entrypoint": ("selected Slurm entrypoint", False),
    "common_runner": ("common Task4 runner", False),
    "executable": ("ABACUS executable", True),
    "environment_script": ("ABACUS environment script", False),
    "mpirun": ("mpirun executable", True),
}


@dataclass(frozen=True)
class PhaseSpec:
    relative: str
    mode: str
    restart: bool
    field_dir: int | None = None


PHASES = (
    PhaseSpec("runs/fixed/fixed_cold", "fixed", False),
    PhaseSpec("runs/fixed/fixed_restart", "fixed", True),
    PhaseSpec("runs/dir0/field_seed", "field", False, 0),
    PhaseSpec("runs/dir0/free_restart1", "free", True),
    PhaseSpec("runs/dir0/free_restart2", "free", True),
    PhaseSpec("runs/dir1/field_seed", "field", False, 1),
    PhaseSpec("runs/dir1/free_restart1", "free", True),
    PhaseSpec("runs/dir1/free_restart2", "free", True),
    PhaseSpec("runs/dir2/field_seed", "field", False, 2),
    PhaseSpec("runs/dir2/free_restart1", "free", True),
    PhaseSpec("runs/dir2/free_restart2", "free", True),
)
_PHASE_LOOKUP = {spec.relative: spec for spec in PHASES}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_regular(path: Path, label: str, *, nonempty: bool = False) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a non-symlink regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed while being opened: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise ValueError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    if nonempty and not content:
        raise ValueError(f"{label} must be nonempty: {path}")
    return content


def _require_local_directory(path: Path, label: str) -> Path:
    absolute = path.absolute()
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {absolute}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symlink local directory: {absolute}")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute:
        raise ValueError(f"{label} must be a non-symlink local directory: {absolute}")
    return resolved


def _record(
    path: Path, base: Path, label: str, *, nonempty: bool = False
) -> dict[str, object]:
    content = _read_regular(path, label, nonempty=nonempty)
    return {
        "relative_path": str(path.relative_to(base)),
        "sha256": _sha256_bytes(content),
        "size": len(content),
    }


def _load_json(path: Path, label: str) -> dict[str, object]:
    content = _read_regular(path, label, nonempty=True)
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short write while recording Task4 evidence")
        remaining = remaining[written:]


def _atomic_write_json(
    path: Path, value: dict[str, object], *, noreplace: bool
) -> None:
    content = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        try:
            _write_all(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if noreplace:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _canonical_file(
    path: str | Path, label: str, *, executable: bool = False
) -> tuple[Path, dict[str, object]]:
    argument = Path(path).expanduser().absolute()
    content = _read_regular(argument, label, nonempty=True)
    resolved = argument.resolve(strict=True)
    if resolved != argument:
        raise ValueError(f"{label} must be supplied as a resolved non-symlink path")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} is not executable: {resolved}")
    return resolved, {
        "absolute_path": str(resolved),
        "sha256": _sha256_bytes(content),
        "size": len(content),
    }


def _positive_int_environment(name: str) -> int:
    value = os.environ.get(name)
    if value is None or not re.fullmatch(r"\d+", value) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_job_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or re.fullmatch(r"[1-9]\d*", value) is None:
        raise ValueError(f"{name} must be a positive numeric Slurm job ID")
    return value


def _selected_resource_profile() -> dict[str, object]:
    name = os.environ.get("C_PBE_GATE_PROFILE")
    if not name:
        raise ValueError("C_PBE_GATE_PROFILE must explicitly name a resource profile")
    try:
        return get_resource_profile(name)
    except ValueError as exc:
        raise ValueError(f"C_PBE_GATE_PROFILE is invalid: {name}") from exc


def _parse_scontrol_fields(output: str) -> dict[str, str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("scontrol must return exactly one job record")
    fields = dict(re.findall(r"(?:^|\s)([^=\s]+)=([^\s]+)", lines[0]))
    required = {
        "JobId",
        "ArrayJobId",
        "ArrayTaskId",
        "Partition",
        "NumNodes",
        "NumCPUs",
        "NumTasks",
        "CPUs/Task",
        "TimeLimit",
        "OverSubscribe",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError("scontrol job record lacks: " + ", ".join(missing))
    return fields


def _scontrol_positive_int(fields: dict[str, str], name: str) -> int:
    value = fields.get(name, "")
    if re.fullmatch(r"[1-9]\d*", value) is None:
        raise ValueError(f"scontrol {name} must be a positive integer")
    return int(value)


def _memory_megabytes(value: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", value)
    if match is None:
        raise ValueError(f"unsupported Slurm memory value: {value}")
    factors = {
        "": Decimal(1),
        "K": Decimal(1) / Decimal(1024),
        "M": Decimal(1),
        "G": Decimal(1024),
        "T": Decimal(1024 * 1024),
    }
    try:
        megabytes = Decimal(match.group(1)) * factors[match.group(2)]
    except InvalidOperation as exc:
        raise ValueError(f"unsupported Slurm memory value: {value}") from exc
    if megabytes != megabytes.to_integral_value():
        raise ValueError(f"Slurm memory is not an integral number of MB: {value}")
    return int(megabytes)


def _scontrol_memory(fields: dict[str, str]) -> tuple[str, int]:
    raw = fields.get("MinMemoryNode")
    if raw is None:
        request = fields.get("ReqTRES", "")
        match = re.search(r"(?:^|,)mem=([^,]+)", request)
        if match is None:
            raise ValueError("scontrol job record lacks per-node memory evidence")
        raw = match.group(1)
    return raw, _memory_megabytes(raw)


def _slurm_duration_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError(f"unsupported Slurm time limit: {value}")
    days = int(match.group(1) or 0)
    hours, minutes, seconds = (int(match.group(index)) for index in (2, 3, 4))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"unsupported Slurm time limit: {value}")
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _normalized_slurm_duration(value: str) -> str:
    total = _slurm_duration_seconds(value)
    days, remainder = divmod(total, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _scheduler_contract(profile: dict[str, object]) -> dict[str, object]:
    nodes = profile["nodes"]
    ntasks = profile["ntasks"]
    if type(nodes) is not int or type(ntasks) is not int or ntasks % nodes != 0:
        raise ValueError("resource profile has invalid task topology")
    return {
        "profile": profile["name"],
        "partition": profile["partition"],
        "nodes": nodes,
        "ntasks": ntasks,
        "tasks_per_node": ntasks // nodes,
        "cpus_per_task": profile["cpus_per_task"],
        "memory_mb": profile["memory_mb"],
        "time_limit": _normalized_slurm_duration(profile["time_limit"]),
        "over_subscribe": profile["over_subscribe"],
    }


def _query_scheduler(job_id: str) -> tuple[dict[str, str], str]:
    try:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"cannot query Slurm allocation with scontrol: {exc}") from exc
    if completed.returncode != 0:
        error = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ValueError(f"scontrol job query failed: {error}")
    return _parse_scontrol_fields(completed.stdout), completed.stdout


def _scheduler_record(branch: str) -> dict[str, object]:
    profile = _selected_resource_profile()
    contract = _scheduler_contract(profile)
    job_id = _positive_job_environment("SLURM_JOB_ID")
    array_job_id = _positive_job_environment("SLURM_ARRAY_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if partition != contract["partition"]:
        raise ValueError(f"Slurm partition must be {contract['partition']}")
    task_id = (
        _positive_int_environment("SLURM_ARRAY_TASK_ID")
        if os.environ.get("SLURM_ARRAY_TASK_ID") != "0"
        else 0
    )
    expected_task = tuple(BRANCH_PHASES).index(branch)
    if task_id != expected_task:
        raise ValueError(f"array task {task_id} does not map to branch {branch}")
    if _positive_int_environment("SLURM_ARRAY_TASK_COUNT") != 4:
        raise ValueError("Slurm array must contain exactly four tasks")
    if _positive_int_environment("SLURM_CPUS_PER_TASK") != contract["cpus_per_task"]:
        raise ValueError(
            f"SLURM_CPUS_PER_TASK must equal {contract['cpus_per_task']}"
        )
    if _positive_int_environment("SLURM_NTASKS") != contract["ntasks"]:
        raise ValueError(f"SLURM_NTASKS must equal {contract['ntasks']}")
    if _positive_int_environment("SLURM_JOB_NUM_NODES") != contract["nodes"]:
        raise ValueError(f"SLURM_JOB_NUM_NODES must equal {contract['nodes']}")
    tasks_per_node = os.environ.get("SLURM_TASKS_PER_NODE", "")
    match = re.fullmatch(
        rf"{contract['tasks_per_node']}(?:\(x{contract['nodes']}\))?",
        tasks_per_node,
    )
    if match is None:
        raise ValueError(
            f"SLURM_TASKS_PER_NODE must equal {contract['tasks_per_node']}"
        )
    if _positive_int_environment("SLURM_MEM_PER_NODE") != contract["memory_mb"]:
        raise ValueError(
            f"SLURM_MEM_PER_NODE must equal {contract['memory_mb']}"
        )
    fields, raw_scontrol = _query_scheduler(job_id)
    observed_job_id = fields["JobId"]
    if observed_job_id not in {job_id, f"{array_job_id}_{task_id}"}:
        raise ValueError("scontrol JobId differs from the running Slurm job")
    if fields["ArrayJobId"] != array_job_id:
        raise ValueError("scontrol ArrayJobId differs from SLURM_ARRAY_JOB_ID")
    if fields["ArrayTaskId"] != str(task_id):
        raise ValueError("scontrol ArrayTaskId differs from SLURM_ARRAY_TASK_ID")
    observed_partition = fields["Partition"]
    observed_nodes = _scontrol_positive_int(fields, "NumNodes")
    observed_cpus = _scontrol_positive_int(fields, "NumCPUs")
    observed_ntasks = _scontrol_positive_int(fields, "NumTasks")
    observed_cpus_per_task = _scontrol_positive_int(fields, "CPUs/Task")
    memory_raw, observed_memory = _scontrol_memory(fields)
    time_limit_raw = fields["TimeLimit"]
    observed_values = {
        "partition": observed_partition,
        "nodes": observed_nodes,
        "ntasks": observed_ntasks,
        "tasks_per_node": observed_ntasks // observed_nodes,
        "cpus_per_task": observed_cpus_per_task,
        "memory_mb": observed_memory,
        "time_limit": _normalized_slurm_duration(time_limit_raw),
        "over_subscribe": fields["OverSubscribe"],
    }
    if observed_cpus != contract["cpus_per_task"]:
        raise ValueError(
            f"scontrol NumCPUs must equal {contract['cpus_per_task']}"
        )
    for key, expected in contract.items():
        if key == "profile":
            continue
        if observed_values[key] != expected:
            raise ValueError(f"observed Slurm {key} must equal {expected}")
    return {
        **contract,
        "array_task_id": task_id,
        "array_task_count": 4,
        "job_id": job_id,
        "array_job_id": array_job_id,
        "observed": {
            "job_id": observed_job_id,
            "array_job_id": fields["ArrayJobId"],
            "array_task_id": int(fields["ArrayTaskId"]),
            "partition": observed_partition,
            "num_nodes": observed_nodes,
            "num_cpus": observed_cpus,
            "num_tasks": observed_ntasks,
            "cpus_per_task": observed_cpus_per_task,
            "memory_raw": memory_raw,
            "time_limit_raw": time_limit_raw,
            "over_subscribe": fields["OverSubscribe"],
            "raw_record": raw_scontrol,
            "scontrol_sha256": _sha256_bytes(raw_scontrol.encode("utf-8")),
        },
    }


def _validate_scheduler_record(value: object, branch: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{branch} scheduler evidence is not an object")
    profile_name = value.get("profile")
    try:
        profile = get_resource_profile(profile_name)
    except ValueError as exc:
        raise ValueError(f"{branch} scheduler evidence has invalid profile") from exc
    expected = {
        **_scheduler_contract(profile),
        "array_task_id": tuple(BRANCH_PHASES).index(branch),
        "array_task_count": 4,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"{branch} scheduler evidence has invalid {key}")
    for key in ("job_id", "array_job_id"):
        job_identity = value.get(key)
        if (
            not isinstance(job_identity, str)
            or re.fullmatch(r"[1-9]\d*", job_identity) is None
        ):
            raise ValueError(f"{branch} scheduler evidence has invalid {key}")
    observed = value.get("observed")
    expected_observed_keys = {
        "job_id",
        "array_job_id",
        "array_task_id",
        "partition",
        "num_nodes",
        "num_cpus",
        "num_tasks",
        "cpus_per_task",
        "memory_raw",
        "time_limit_raw",
        "over_subscribe",
        "raw_record",
        "scontrol_sha256",
    }
    if not isinstance(observed, dict) or set(observed) != expected_observed_keys:
        raise ValueError(f"{branch} scheduler observed evidence is incomplete")
    string_fields = {
        "job_id",
        "array_job_id",
        "partition",
        "memory_raw",
        "time_limit_raw",
        "over_subscribe",
        "raw_record",
        "scontrol_sha256",
    }
    integer_fields = {
        "array_task_id",
        "num_nodes",
        "num_cpus",
        "num_tasks",
        "cpus_per_task",
    }
    if any(not isinstance(observed[name], str) for name in string_fields) or any(
        type(observed[name]) is not int for name in integer_fields
    ):
        raise ValueError(f"{branch} scheduler observed evidence is invalid")
    raw_record = observed["raw_record"]
    if _sha256_bytes(raw_record.encode("utf-8")) != observed["scontrol_sha256"]:
        raise ValueError(f"{branch} raw scontrol record hash is invalid")
    raw_fields = _parse_scontrol_fields(raw_record)
    raw_memory, _ = _scontrol_memory(raw_fields)
    raw_values = {
        "job_id": raw_fields["JobId"],
        "array_job_id": raw_fields["ArrayJobId"],
        "array_task_id": int(raw_fields["ArrayTaskId"]),
        "partition": raw_fields["Partition"],
        "num_nodes": _scontrol_positive_int(raw_fields, "NumNodes"),
        "num_cpus": _scontrol_positive_int(raw_fields, "NumCPUs"),
        "num_tasks": _scontrol_positive_int(raw_fields, "NumTasks"),
        "cpus_per_task": _scontrol_positive_int(raw_fields, "CPUs/Task"),
        "memory_raw": raw_memory,
        "time_limit_raw": raw_fields["TimeLimit"],
        "over_subscribe": raw_fields["OverSubscribe"],
    }
    if any(observed[key] != raw_value for key, raw_value in raw_values.items()):
        raise ValueError(f"{branch} raw scontrol record content is inconsistent")
    if (
        observed["array_job_id"] != value["array_job_id"]
        or observed["array_task_id"] != value["array_task_id"]
        or observed["partition"] != value["partition"]
        or observed["num_nodes"] != value["nodes"]
        or observed["num_cpus"] != value["cpus_per_task"]
        or observed["num_tasks"] != value["ntasks"]
        or observed["cpus_per_task"] != value["cpus_per_task"]
        or _memory_megabytes(observed["memory_raw"]) != value["memory_mb"]
        or _normalized_slurm_duration(observed["time_limit_raw"])
        != value["time_limit"]
        or observed["over_subscribe"] != value["over_subscribe"]
        or not isinstance(observed["scontrol_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", observed["scontrol_sha256"]) is None
    ):
        raise ValueError(f"{branch} scheduler observed evidence is invalid")
    observed_job_id = observed["job_id"]
    valid_observed_job_ids = {
        value["job_id"],
        f"{value['array_job_id']}_{value['array_task_id']}",
    }
    if observed_job_id not in valid_observed_job_ids:
        raise ValueError(f"{branch} scheduler observed JobId is invalid")
    return value


def _branch_provenance(root: Path, branch: str) -> tuple[Path, dict[str, object]]:
    branch_root = root / "runs" / branch
    if branch_root.is_symlink() or not branch_root.is_dir():
        raise ValueError(f"branch directory is missing or a symlink: {branch}")
    provenance = _load_json(
        branch_root / "BRANCH_PROVENANCE.json", f"{branch} preparation provenance"
    )
    if (
        provenance.get("branch") != branch
        or provenance.get("schema") != "c-pbe-reference-gate-branch"
    ):
        raise ValueError(f"{branch} preparation provenance identity is invalid")
    _preparation_signature(provenance, branch)
    return branch_root, provenance


def _preparation_signature(
    provenance: dict[str, object], branch: str
) -> dict[str, object]:
    expected_keys = {
        "schema",
        "version",
        "branch",
        "mode",
        "field_dir",
        "box_angstrom",
        "atom_direct",
        "sources",
        "renderer",
        "frozen_protocol",
        "phase",
    }
    if set(provenance) != expected_keys:
        raise ValueError(f"{branch} preparation provenance fields are invalid")
    try:
        expected_mode, expected_direction = prepare_gate.BRANCHES[branch]
    except KeyError as exc:
        raise ValueError(f"unsupported preparation branch: {branch}") from exc
    expected_identity = (
        "c-pbe-reference-gate-branch",
        1,
        branch,
        expected_mode,
        expected_direction,
        prepare_gate._BOX_ANGSTROM,
        list(prepare_gate._ATOM_DIRECT),
    )
    identity = (
        provenance.get("schema"),
        provenance.get("version"),
        provenance.get("branch"),
        provenance.get("mode"),
        provenance.get("field_dir"),
        provenance.get("box_angstrom"),
        provenance.get("atom_direct"),
    )
    if identity != expected_identity:
        raise ValueError(f"{branch} preparation provenance identity is invalid")
    if provenance.get("frozen_protocol") != dict(FROZEN_PROTOCOL):
        raise ValueError(f"{branch} frozen preparation protocol is invalid")
    expected_renderer = {
        "function": "gate_contract.render_input",
        "mode": expected_mode,
        "field_dir": expected_direction,
        "restart": False,
    }
    if provenance.get("renderer") != expected_renderer:
        raise ValueError(f"{branch} preparation renderer identity is invalid")
    pseudo_name, orbital_name = _asset_names(provenance)
    sources = provenance.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"pseudo", "orbital"}:
        raise ValueError(f"{branch} preparation source records are invalid")
    source_keys = {
        "absolute_path",
        "basename",
        "sha256",
        "size",
        "device",
        "inode",
        "mtime_ns",
    }
    signatures = {}
    for label, basename in (("pseudo", pseudo_name), ("orbital", orbital_name)):
        source = sources[label]
        if not isinstance(source, dict) or set(source) != source_keys:
            raise ValueError(f"{branch} {label} source provenance is invalid")
        if (
            source.get("basename") != basename
            or not isinstance(source.get("absolute_path"), str)
            or not Path(source["absolute_path"]).is_absolute()
            or not isinstance(source.get("size"), int)
            or source["size"] < 1
            or not isinstance(source.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
            or any(
                not isinstance(source.get(name), int) or source[name] < 0
                for name in ("device", "inode", "mtime_ns")
            )
        ):
            raise ValueError(f"{branch} {label} source provenance is invalid")
        signatures[label] = {
            "basename": basename,
            "sha256": source["sha256"],
            "size": source["size"],
        }
    phase = provenance.get("phase")
    initial_phase = BRANCH_PHASES[branch][0]
    if (
        not isinstance(phase, dict)
        or set(phase) != {"relative_path", "files"}
        or phase.get("relative_path") != f"runs/{branch}/{initial_phase}"
        or not isinstance(phase.get("files"), dict)
    ):
        raise ValueError(f"{branch} preparation phase provenance is invalid")
    return {
        "box_angstrom": provenance["box_angstrom"],
        "atom_direct": provenance["atom_direct"],
        "frozen_protocol": provenance["frozen_protocol"],
        "pseudo": signatures["pseudo"],
        "orbital": signatures["orbital"],
    }


def _phase_spec(branch: str, phase: str) -> PhaseSpec:
    relative = f"runs/{branch}/{phase}"
    try:
        return _PHASE_LOOKUP[relative]
    except KeyError as exc:
        raise ValueError(f"unexpected phase for {branch}: {phase}") from exc


def _field_direction(branch: str, spec: PhaseSpec) -> int | None:
    if spec.mode == "fixed":
        return None
    return int(branch[-1])


def _expected_input(branch: str, spec: PhaseSpec) -> bytes:
    return render_input(
        mode=spec.mode,
        field_dir=_field_direction(branch, spec),
        restart=spec.restart,
    ).encode("utf-8")


def _asset_names(provenance: dict[str, object]) -> tuple[str, str]:
    try:
        pseudo = provenance["sources"]["pseudo"]["basename"]
        orbital = provenance["sources"]["orbital"]["basename"]
    except (KeyError, TypeError) as exc:
        raise ValueError("preparation provenance lacks source asset names") from exc
    if not isinstance(pseudo, str) or not isinstance(orbital, str):
        raise ValueError("preparation provenance has invalid source asset names")
    if Path(pseudo).name != pseudo or Path(orbital).name != orbital:
        raise ValueError("preparation provenance has unsafe asset basenames")
    try:
        prepare_gate._validate_asset_names(Path(pseudo), Path(orbital))
    except ValueError as exc:
        raise ValueError(
            f"preparation provenance has unsafe asset basenames: {exc}"
        ) from exc
    return pseudo, orbital


def _validate_phase_controls(
    root: Path, branch: str, phase: str
) -> tuple[Path, dict[str, object], PhaseSpec]:
    branch_root, provenance = _branch_provenance(root, branch)
    spec = _phase_spec(branch, phase)
    phase_root = branch_root / phase
    if phase_root.is_symlink() or not phase_root.is_dir():
        raise ValueError(f"phase directory is missing or a symlink: {branch}/{phase}")
    input_content = _read_regular(phase_root / "INPUT", "phase INPUT", nonempty=True)
    if input_content != _expected_input(branch, spec):
        raise ValueError(f"{branch}/{phase} INPUT does not match the frozen renderer")

    initial_phase = BRANCH_PHASES[branch][0]
    initial_root = branch_root / initial_phase
    pseudo_name, orbital_name = _asset_names(provenance)
    expected_controls = {
        "STRU": prepare_gate._render_stru(pseudo_name, orbital_name).encode("utf-8"),
        "KPT": prepare_gate._KPT_TEXT.encode("utf-8"),
    }
    for name, expected_content in expected_controls.items():
        initial_content = _read_regular(
            initial_root / name, f"prepared {initial_phase} {name}", nonempty=True
        )
        if initial_content != expected_content:
            raise ValueError(
                f"{branch}/{initial_phase} {name} differs from the frozen "
                "preparation provenance template"
            )
    if phase == initial_phase:
        prepared_phase = provenance.get("phase")
        if not isinstance(prepared_phase, dict):
            raise ValueError(f"{branch} preparation provenance lacks phase records")
        prepared_files = prepared_phase.get("files")
        prepared_names = {"INPUT", "STRU", "KPT", pseudo_name, orbital_name}
        if (
            prepared_phase.get("relative_path") != f"runs/{branch}/{initial_phase}"
            or not isinstance(prepared_files, dict)
            or set(prepared_files) != prepared_names
        ):
            raise ValueError(f"{branch} preparation provenance phase record is invalid")
        for name in prepared_names:
            record = prepared_files[name]
            content = _read_regular(
                phase_root / name, f"prepared {initial_phase} {name}", nonempty=True
            )
            expected_record = {
                "relative_path": f"runs/{branch}/{initial_phase}/{name}",
                "sha256": _sha256_bytes(content),
                "size": len(content),
            }
            if record != expected_record:
                raise ValueError(
                    f"{branch}/{initial_phase} {name} differs from preparation provenance"
                )
    for name in ("STRU", "KPT"):
        current = _read_regular(phase_root / name, f"{phase} {name}", nonempty=True)
        initial = _read_regular(
            initial_root / name, f"{initial_phase} {name}", nonempty=True
        )
        if current != initial:
            raise ValueError(
                f"{branch}/{phase} {name} differs from the prepared branch"
            )
    for label, name in zip(("pseudo", "orbital"), (pseudo_name, orbital_name)):
        content = _read_regular(phase_root / name, f"{phase} {label}", nonempty=True)
        source = provenance["sources"][label]
        if _sha256_bytes(content) != source.get("sha256") or len(content) != source.get(
            "size"
        ):
            raise ValueError(f"{branch}/{phase} {label} differs from branch provenance")
    return phase_root, provenance, spec


def initialize_branch_run(
    root: str | Path,
    branch: str,
    gate_profile: str,
    python: str | Path,
    prepare_gate_source: str | Path,
    audit_gate_source: str | Path,
    gate_contract: str | Path,
    resource_profiles: str | Path,
    entrypoint: str | Path,
    common_runner: str | Path,
    abacus: str | Path,
    environment_script: str | Path,
    mpirun: str | Path,
) -> dict[str, object]:
    root_path = Path(root).expanduser().absolute()
    branch_root, _ = _branch_provenance(root_path, branch)
    try:
        get_resource_profile(gate_profile)
    except ValueError as exc:
        raise ValueError(f"invalid C PBE gate runtime profile: {gate_profile}") from exc
    if os.environ.get("C_PBE_GATE_PROFILE") != gate_profile:
        raise ValueError("runner profile differs from C_PBE_GATE_PROFILE")
    runtime_paths = {
        "python": python,
        "prepare_gate": prepare_gate_source,
        "audit_gate": audit_gate_source,
        "gate_contract": gate_contract,
        "resource_profiles": resource_profiles,
        "entrypoint": entrypoint,
        "common_runner": common_runner,
        "executable": abacus,
        "environment_script": environment_script,
        "mpirun": mpirun,
    }
    runtime_records = {}
    for name, path in runtime_paths.items():
        label, executable = RUNTIME_RECORD_SPECS[name]
        _, runtime_records[name] = _canonical_file(
            path, label, executable=executable
        )
    scheduler = _scheduler_record(branch)
    if scheduler["profile"] != gate_profile:
        raise ValueError("scheduler profile differs from the runner profile")
    provenance = {
        "schema": "c-pbe-reference-gate-run",
        "version": 1,
        "status": "RUN_PROVENANCE",
        "branch": branch,
        "gate_profile": gate_profile,
        **runtime_records,
        "scheduler": scheduler,
        "preparation_provenance_sha256": _sha256_bytes(
            _read_regular(
                branch_root / "BRANCH_PROVENANCE.json",
                "preparation provenance",
                nonempty=True,
            )
        ),
    }
    _atomic_write_json(
        branch_root / "BRANCH_RUN_PROVENANCE.json", provenance, noreplace=True
    )
    return provenance


def _run_provenance(root: Path, branch: str) -> tuple[Path, dict[str, object]]:
    branch_root, _ = _branch_provenance(root, branch)
    value = _load_json(
        branch_root / "BRANCH_RUN_PROVENANCE.json", f"{branch} run provenance"
    )
    if (
        value.get("schema") != "c-pbe-reference-gate-run"
        or value.get("branch") != branch
    ):
        raise ValueError(f"{branch} run provenance identity is invalid")
    scheduler = _validate_scheduler_record(value.get("scheduler"), branch)
    if value.get("gate_profile") != scheduler["profile"]:
        raise ValueError(f"{branch} run and scheduler profiles differ")
    records = {
        name: value.get(name) for name in RUNTIME_RECORD_SPECS
    }
    if any(not isinstance(record, dict) for record in records.values()):
        raise ValueError(f"{branch} run provenance lacks runtime file records")
    for name, record in records.items():
        label, executable = RUNTIME_RECORD_SPECS[name]
        _, current = _canonical_file(
            record.get("absolute_path", ""),
            f"recorded {label}",
            executable=executable,
        )
        if current != record:
            raise ValueError(f"{branch} recorded {name} hash no longer matches")
    current_preparation_hash = _sha256_bytes(
        _read_regular(
            branch_root / "BRANCH_PROVENANCE.json",
            f"{branch} preparation provenance",
            nonempty=True,
        )
    )
    if value.get("preparation_provenance_sha256") != current_preparation_hash:
        raise ValueError(f"{branch} preparation provenance hash changed after launch")
    return branch_root, value


def preflight_phase(root: str | Path, branch: str, phase: str) -> None:
    root_path = Path(root).expanduser().absolute()
    phase_root, _, spec = _validate_phase_controls(root_path, branch, phase)
    _run_provenance(root_path, branch)
    for name in ("PHASE_COMPLETE.json", "abacus.stdout", "abacus.stderr"):
        if os.path.lexists(phase_root / name):
            raise ValueError(
                f"stale phase evidence exists before launch: {branch}/{phase}/{name}"
            )
    out = phase_root / "OUT.C_PBE_REFERENCE_GATE"
    if spec.restart:
        if out.is_symlink() or not out.is_dir():
            raise ValueError(
                f"restart phase lacks local OUT directory: {branch}/{phase}"
            )
        actual = set(os.listdir(out))
        if actual != set(RESTART_FILES):
            raise ValueError(
                f"restart phase contains stale or incomplete OUT files: {branch}/{phase}"
            )
        for name in RESTART_FILES:
            _read_regular(out / name, f"restart input {name}", nonempty=True)
        _verify_restart_provenance(
            root_path,
            branch,
            phase,
            require_verified=False,
            check_planned_destination=True,
        )
    elif os.path.lexists(out):
        raise ValueError(
            f"cold/seed phase contains stale OUT directory: {branch}/{phase}"
        )


def _copy_regular(source: Path, destination: Path, label: str) -> dict[str, object]:
    content = _read_regular(source, label, nonempty=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o644)
    try:
        _write_all(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"sha256": _sha256_bytes(content), "size": len(content)}


def create_restart_phase(
    root: str | Path, branch: str, source_phase: str, destination_phase: str
) -> Path:
    root_path = Path(root).expanduser().absolute()
    branch_root, provenance = _branch_provenance(root_path, branch)
    chain = BRANCH_PHASES[branch]
    try:
        source_index = chain.index(source_phase)
    except ValueError as exc:
        raise ValueError(f"invalid source phase for {branch}: {source_phase}") from exc
    if source_index + 1 >= len(chain) or chain[source_index + 1] != destination_phase:
        raise ValueError(f"invalid restart chain {source_phase}->{destination_phase}")
    source_root = branch_root / source_phase
    _verify_phase_evidence(root_path, branch, source_phase)
    source_manifest = _load_json(
        source_root / "PHASE_COMPLETE.json", "source phase completion"
    )
    if source_manifest.get("status") != "PHASE_COMPLETE":
        raise ValueError("source phase is not complete")
    destination_spec = _phase_spec(branch, destination_phase)
    if not destination_spec.restart:
        raise ValueError("restart destination must declare restart input")
    destination = branch_root / destination_phase
    if os.path.lexists(destination):
        raise FileExistsError(f"restart destination already exists: {destination}")

    temporary = branch_root / f".{destination_phase}.restart-{secrets.token_hex(8)}"
    temporary.mkdir(mode=0o700)
    published = False
    try:
        pseudo_name, orbital_name = _asset_names(provenance)
        for name in ("STRU", "KPT", pseudo_name, orbital_name):
            _copy_regular(source_root / name, temporary / name, f"source {name}")
        input_content = _expected_input(branch, destination_spec)
        (temporary / "INPUT").write_bytes(input_content)
        os.mkdir(temporary / "OUT.C_PBE_REFERENCE_GATE")
        os.mkdir(temporary / "restart_input_snapshot")
        file_records = {}
        for name in RESTART_FILES:
            source = source_root / "OUT.C_PBE_REFERENCE_GATE" / name
            destination_file = temporary / "OUT.C_PBE_REFERENCE_GATE" / name
            snapshot = temporary / "restart_input_snapshot" / name
            source_content = _read_regular(
                source, f"source restart file {name}", nonempty=True
            )
            _copy_regular(source, destination_file, f"source restart file {name}")
            _copy_regular(source, snapshot, f"source restart file {name}")
            digest = _sha256_bytes(source_content)
            size = len(source_content)
            file_records[name] = {
                "source_relative_path": str(source.relative_to(root_path)),
                "source_sha256": digest,
                "source_size": size,
                "destination_relative_path": str(
                    (destination / "OUT.C_PBE_REFERENCE_GATE" / name).relative_to(
                        root_path
                    )
                ),
                "destination_sha256": digest,
                "destination_size": size,
                "snapshot_relative_path": str(
                    (destination / "restart_input_snapshot" / name).relative_to(
                        root_path
                    )
                ),
                "snapshot_sha256": digest,
                "snapshot_size": size,
            }
        restart = {
            "schema": "c-pbe-reference-gate-restart",
            "version": 1,
            "status": "PLANNED",
            "branch": branch,
            "source_phase": source_phase,
            "destination_phase": destination_phase,
            "source_phase_complete_sha256": _sha256_bytes(
                _read_regular(
                    source_root / "PHASE_COMPLETE.json",
                    "source phase completion",
                    nonempty=True,
                )
            ),
            "destination_input_sha256": _sha256_bytes(input_content),
            "files": file_records,
            "load_evidence": None,
        }
        _atomic_write_json(
            temporary / "RESTART_PROVENANCE.json", restart, noreplace=True
        )
        _fsync_directory(temporary)
        branch_fd = os.open(
            branch_root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            prepare_gate._publish_noreplace(
                branch_fd, temporary.name, destination_phase
            )
            published = True
            prepare_gate._fsync_directory(branch_fd)
        finally:
            os.close(branch_fd)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _restart_load_lines(phase_root: Path) -> dict[str, object]:
    stdout_path = phase_root / "abacus.stdout"
    log_path = phase_root / "OUT.C_PBE_REFERENCE_GATE/running_scf.log"
    stdout = _read_regular(stdout_path, "ABACUS stdout").decode(
        "utf-8", errors="strict"
    )
    log = _read_regular(log_path, "running_scf.log", nonempty=True).decode(
        "utf-8", errors="strict"
    )
    wfc_lines = []
    charge_lines = []
    all_wfc_lines = re.findall(
        r"^.*Read NAO wave functions from .*\s*$", stdout, flags=re.MULTILINE
    )
    all_charge_lines = re.findall(
        r"^.*Read in electron density: .*\s*$", log, flags=re.MULTILINE
    )
    if len(all_wfc_lines) != 2:
        raise ValueError(
            "restart stdout must contain exactly two wave-function load messages"
        )
    if len(all_charge_lines) != 2:
        raise ValueError(
            "running_scf.log must contain exactly two charge-density load messages"
        )
    canonical_phase = phase_root.resolve(strict=True)
    out = canonical_phase / "OUT.C_PBE_REFERENCE_GATE"
    canonical_out = out.resolve(strict=True)
    if out != canonical_out:
        raise ValueError("restart OUT directory must be a canonical local directory")

    def require_exact_path(line: str, marker: str, name: str) -> None:
        loaded_path = line.split(marker, 1)[1].strip()
        expected_path = (canonical_out / name).resolve(strict=True)
        logged = Path(loaded_path)
        candidate = logged if logged.is_absolute() else canonical_phase / logged
        try:
            canonical_candidate = candidate.resolve(strict=True)
            canonical_candidate.relative_to(canonical_phase)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            canonical_candidate = None
        if canonical_candidate != expected_path:
            raise ValueError(
                f"restart load evidence must use the exact phase-local restart path: "
                f"{expected_path}"
            )

    for spin in (1, 2):
        wfc_pattern = re.compile(
            rf"^.*Read NAO wave functions from .*wfs{spin}_nao\.txt\s*$", re.MULTILINE
        )
        charge_pattern = re.compile(
            rf"^.*Read in electron density: .*chgs{spin}\.cube\s*$", re.MULTILINE
        )
        wfc_matches = wfc_pattern.findall(stdout)
        charge_matches = charge_pattern.findall(log)
        if len(wfc_matches) != 1 or len(charge_matches) != 1:
            raise ValueError(
                f"restart load evidence for spin {spin} is missing or ambiguous"
            )
        require_exact_path(
            wfc_matches[0], "Read NAO wave functions from ", f"wfs{spin}_nao.txt"
        )
        require_exact_path(
            charge_matches[0], "Read in electron density: ", f"chgs{spin}.cube"
        )
        wfc_lines.extend(wfc_matches)
        charge_lines.extend(charge_matches)
    return {
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8")),
        "running_scf_log_sha256": _sha256_bytes(log.encode("utf-8")),
        "wfc_load_lines": wfc_lines,
        "charge_load_lines": charge_lines,
    }


def _verify_restart_provenance(
    root: Path,
    branch: str,
    phase: str,
    *,
    require_verified: bool,
    check_planned_destination: bool = True,
) -> dict[str, object]:
    phase_root = root / "runs" / branch / phase
    _require_local_directory(
        phase_root / "restart_input_snapshot", "restart snapshot directory"
    )
    path = phase_root / "RESTART_PROVENANCE.json"
    value = _load_json(path, "restart provenance")
    expected_status = "VERIFIED" if require_verified else "PLANNED"
    if (
        value.get("schema") != "c-pbe-reference-gate-restart"
        or value.get("status") != expected_status
    ):
        raise ValueError(f"{branch}/{phase} restart provenance status is invalid")
    if value.get("branch") != branch or value.get("destination_phase") != phase:
        raise ValueError(f"{branch}/{phase} restart provenance identity is invalid")
    chain = BRANCH_PHASES[branch]
    phase_index = chain.index(phase)
    expected_source = chain[phase_index - 1]
    if phase_index == 0 or value.get("source_phase") != expected_source:
        raise ValueError(f"{branch}/{phase} restart source chain is invalid")
    source_root = root / "runs" / branch / expected_source
    source_manifest_path = source_root / "PHASE_COMPLETE.json"
    if value.get("source_phase_complete_sha256") != _sha256_bytes(
        _read_regular(source_manifest_path, "source phase completion", nonempty=True)
    ):
        raise ValueError(f"{branch}/{phase} source phase manifest hash changed")
    if value.get("destination_input_sha256") != _sha256_bytes(
        _read_regular(phase_root / "INPUT", "restart INPUT", nonempty=True)
    ):
        raise ValueError(f"{branch}/{phase} destination INPUT hash changed")
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != set(RESTART_FILES):
        raise ValueError(f"{branch}/{phase} restart file manifest is incomplete")
    for name in RESTART_FILES:
        record = files[name]
        if not isinstance(record, dict):
            raise ValueError(f"{branch}/{phase} restart record is invalid for {name}")
        expected_paths = {
            "source_relative_path": f"runs/{branch}/{expected_source}/OUT.C_PBE_REFERENCE_GATE/{name}",
            "destination_relative_path": f"runs/{branch}/{phase}/OUT.C_PBE_REFERENCE_GATE/{name}",
            "snapshot_relative_path": f"runs/{branch}/{phase}/restart_input_snapshot/{name}",
        }
        for key, expected_path in expected_paths.items():
            if record.get(key) != expected_path:
                raise ValueError(
                    f"{branch}/{phase} restart relative path is invalid for {name}"
                )
        source_content = _read_regular(
            source_root / "OUT.C_PBE_REFERENCE_GATE" / name,
            f"source {name}",
            nonempty=True,
        )
        snapshot_content = _read_regular(
            phase_root / "restart_input_snapshot" / name,
            f"snapshot {name}",
            nonempty=True,
        )
        digest = _sha256_bytes(source_content)
        size = len(source_content)
        if digest != _sha256_bytes(snapshot_content) or size != len(snapshot_content):
            raise ValueError(
                f"{branch}/{phase} snapshot differs from source output for {name}"
            )
        for prefix in ("source", "destination", "snapshot"):
            if (
                record.get(f"{prefix}_sha256") != digest
                or record.get(f"{prefix}_size") != size
            ):
                raise ValueError(
                    f"{branch}/{phase} recorded {prefix} hash is invalid for {name}"
                )
        if not require_verified and check_planned_destination:
            destination_content = _read_regular(
                phase_root / "OUT.C_PBE_REFERENCE_GATE" / name,
                f"destination restart input {name}",
                nonempty=True,
            )
            if (
                _sha256_bytes(destination_content) != digest
                or len(destination_content) != size
            ):
                raise ValueError(
                    f"{branch}/{phase} destination restart input differs for {name}"
                )
    if require_verified:
        actual_evidence = _restart_load_lines(phase_root)
        if value.get("load_evidence") != actual_evidence:
            raise ValueError(f"{branch}/{phase} recorded load evidence is invalid")
    elif value.get("load_evidence") is not None:
        raise ValueError(
            f"{branch}/{phase} planned restart already claims load evidence"
        )
    return value


def _verify_and_publish_restart(root: Path, branch: str, phase: str) -> str:
    phase_root = root / "runs" / branch / phase
    value = _verify_restart_provenance(
        root,
        branch,
        phase,
        require_verified=False,
        check_planned_destination=False,
    )
    value["status"] = "VERIFIED"
    value["load_evidence"] = _restart_load_lines(phase_root)
    _atomic_write_json(phase_root / "RESTART_PROVENANCE.json", value, noreplace=False)
    _verify_restart_provenance(root, branch, phase, require_verified=True)
    return _sha256_bytes(
        _read_regular(
            phase_root / "RESTART_PROVENANCE.json", "restart provenance", nonempty=True
        )
    )


def _parse_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO UTC timestamp") from exc
    return parsed


def complete_phase(
    root: str | Path,
    branch: str,
    phase: str,
    started_utc: str,
    ended_utc: str,
    wall_seconds: float,
) -> dict[str, object]:
    root_path = Path(root).expanduser().absolute()
    phase_root, provenance, spec = _validate_phase_controls(root_path, branch, phase)
    branch_root, run = _run_provenance(root_path, branch)
    started = _parse_utc(started_utc, "phase start")
    ended = _parse_utc(ended_utc, "phase end")
    if ended < started or wall_seconds < 0:
        raise ValueError("phase timing is invalid")
    result = audit_phase(
        phase_root,
        expected_mode=spec.mode,
        expected_restart=spec.restart,
        expected_field_dir=spec.field_dir,
    )
    out = phase_root / "OUT.C_PBE_REFERENCE_GATE"
    for name in RESTART_FILES:
        _read_regular(out / name, f"phase restart output {name}", nonempty=True)
    for name in ("abacus.stdout", "abacus.stderr"):
        _read_regular(phase_root / name, name)
    restart_hash = (
        _verify_and_publish_restart(root_path, branch, phase) if spec.restart else None
    )
    pseudo_name, orbital_name = _asset_names(provenance)
    controls = {
        name: _record(phase_root / name, phase_root, name, nonempty=True)
        for name in ("INPUT", "STRU", "KPT")
    }
    assets = {
        name: _record(phase_root / name, phase_root, name, nonempty=True)
        for name in (pseudo_name, orbital_name)
    }
    output_paths = {
        "running_scf.log": out / "running_scf.log",
        "eig_occ.txt": out / "eig_occ.txt",
        "abacus.stdout": phase_root / "abacus.stdout",
        "abacus.stderr": phase_root / "abacus.stderr",
        **{name: out / name for name in RESTART_FILES},
    }
    outputs = {
        name: _record(
            path,
            phase_root,
            name,
            nonempty=name not in {"abacus.stdout", "abacus.stderr"},
        )
        for name, path in output_paths.items()
    }
    manifest = {
        "schema": "c-pbe-reference-gate-phase-complete",
        "version": 1,
        "status": "PHASE_COMPLETE",
        "branch": branch,
        "phase": phase,
        "mode": spec.mode,
        "field_dir": _field_direction(branch, spec),
        "restart_loaded": spec.restart,
        "restart_provenance_sha256": restart_hash,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "wall_seconds": float(wall_seconds),
        "scheduler": run["scheduler"],
        "gate_profile": run["gate_profile"],
        **{name: run[name] for name in RUNTIME_RECORD_SPECS},
        "controls": controls,
        "assets": assets,
        "outputs": outputs,
        "energy_ev": result.energy_ev,
        "energy_ha": result.energy_ha,
        "spin_counts": {str(key): value for key, value in result.spin_counts.items()},
        "occupations": {
            str(key): list(values) for key, values in result.occupations.items()
        },
        "stage_sha256": result.stage_hash,
    }
    _atomic_write_json(phase_root / "PHASE_COMPLETE.json", manifest, noreplace=True)
    return manifest


def _verify_record(
    path: Path, record: object, base: Path, label: str, *, nonempty: bool = False
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is invalid")
    actual = _record(path, base, label, nonempty=nonempty)
    if record != actual:
        raise ValueError(f"{label} record does not match the current file")


def _verify_phase_evidence(
    root: Path, branch: str, phase: str, result: PhaseResult | None = None
) -> dict[str, object]:
    phase_root, provenance, spec = _validate_phase_controls(root, branch, phase)
    _, run = _run_provenance(root, branch)
    manifest_path = phase_root / "PHASE_COMPLETE.json"
    manifest = _load_json(manifest_path, f"{branch}/{phase} phase completion")
    identity = (
        manifest.get("schema"),
        manifest.get("status"),
        manifest.get("branch"),
        manifest.get("phase"),
        manifest.get("mode"),
        manifest.get("field_dir"),
        manifest.get("restart_loaded"),
    )
    expected_identity = (
        "c-pbe-reference-gate-phase-complete",
        "PHASE_COMPLETE",
        branch,
        phase,
        spec.mode,
        _field_direction(branch, spec),
        spec.restart,
    )
    if identity != expected_identity:
        raise ValueError(f"{branch}/{phase} phase manifest identity is invalid")
    if (
        manifest.get("scheduler") != run["scheduler"]
        or manifest.get("gate_profile") != run["gate_profile"]
        or any(
            manifest.get(name) != run[name] for name in RUNTIME_RECORD_SPECS
        )
    ):
        raise ValueError(
            f"{branch}/{phase} runtime provenance differs from branch evidence"
        )
    _validate_scheduler_record(manifest["scheduler"], branch)
    started = _parse_utc(manifest.get("started_utc"), "phase start")
    ended = _parse_utc(manifest.get("ended_utc"), "phase end")
    wall = manifest.get("wall_seconds")
    if ended < started or not isinstance(wall, (int, float)) or wall < 0:
        raise ValueError(f"{branch}/{phase} timing evidence is invalid")
    pseudo_name, orbital_name = _asset_names(provenance)
    controls = manifest.get("controls")
    assets = manifest.get("assets")
    outputs = manifest.get("outputs")
    if not isinstance(controls, dict) or set(controls) != {"INPUT", "STRU", "KPT"}:
        raise ValueError(f"{branch}/{phase} control manifest is incomplete")
    if not isinstance(assets, dict) or set(assets) != {pseudo_name, orbital_name}:
        raise ValueError(f"{branch}/{phase} asset manifest is incomplete")
    expected_outputs = {
        "running_scf.log",
        "eig_occ.txt",
        "abacus.stdout",
        "abacus.stderr",
        *RESTART_FILES,
    }
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        raise ValueError(f"{branch}/{phase} output manifest is incomplete")
    for name in controls:
        _verify_record(
            phase_root / name,
            controls[name],
            phase_root,
            f"{phase} {name}",
            nonempty=True,
        )
    for name in assets:
        _verify_record(
            phase_root / name,
            assets[name],
            phase_root,
            f"{phase} {name}",
            nonempty=True,
        )
    out = phase_root / "OUT.C_PBE_REFERENCE_GATE"
    output_paths = {
        "running_scf.log": out / "running_scf.log",
        "eig_occ.txt": out / "eig_occ.txt",
        "abacus.stdout": phase_root / "abacus.stdout",
        "abacus.stderr": phase_root / "abacus.stderr",
        **{name: out / name for name in RESTART_FILES},
    }
    for name, path in output_paths.items():
        _verify_record(
            path,
            outputs[name],
            phase_root,
            f"{phase} {name}",
            nonempty=name not in {"abacus.stdout", "abacus.stderr"},
        )
    if result is None:
        result = audit_phase(phase_root, spec.mode, spec.restart, spec.field_dir)
    if (
        manifest.get("energy_ev") != result.energy_ev
        or manifest.get("energy_ha") != result.energy_ha
    ):
        raise ValueError(f"{branch}/{phase} recorded energy differs from ABACUS output")
    if manifest.get("spin_counts") != {
        str(key): value for key, value in result.spin_counts.items()
    }:
        raise ValueError(
            f"{branch}/{phase} recorded spin counts differ from ABACUS output"
        )
    if manifest.get("occupations") != {
        str(key): list(values) for key, values in result.occupations.items()
    }:
        raise ValueError(
            f"{branch}/{phase} recorded occupations differ from ABACUS output"
        )
    if manifest.get("stage_sha256") != result.stage_hash:
        raise ValueError(
            f"{branch}/{phase} recorded stage hash differs from ABACUS output"
        )
    if spec.restart:
        _verify_restart_provenance(root, branch, phase, require_verified=True)
        restart_hash = _sha256_bytes(
            _read_regular(
                phase_root / "RESTART_PROVENANCE.json",
                "restart provenance",
                nonempty=True,
            )
        )
        if manifest.get("restart_provenance_sha256") != restart_hash:
            raise ValueError(f"{branch}/{phase} restart provenance hash differs")
    elif manifest.get("restart_provenance_sha256") is not None:
        raise ValueError(f"{branch}/{phase} cold phase claims restart evidence")
    return manifest


def complete_branch(
    root: str | Path, branch: str, started_utc: str, ended_utc: str, wall_seconds: float
) -> dict[str, object]:
    root_path = Path(root).expanduser().absolute()
    branch_root, run = _run_provenance(root_path, branch)
    if os.path.lexists(branch_root / "RUN_FAILED.json"):
        raise ValueError(f"{branch} has failure evidence and cannot be completed")
    started = _parse_utc(started_utc, "branch start")
    ended = _parse_utc(ended_utc, "branch end")
    if ended < started or wall_seconds < 0:
        raise ValueError("branch timing is invalid")
    phase_hashes = {}
    restart_hashes = {}
    for phase in BRANCH_PHASES[branch]:
        _verify_phase_evidence(root_path, branch, phase)
        phase_hashes[phase] = _sha256_bytes(
            _read_regular(
                branch_root / phase / "PHASE_COMPLETE.json",
                "phase completion",
                nonempty=True,
            )
        )
        spec = _phase_spec(branch, phase)
        if spec.restart:
            restart_hashes[phase] = _sha256_bytes(
                _read_regular(
                    branch_root / phase / "RESTART_PROVENANCE.json",
                    "restart provenance",
                    nonempty=True,
                )
            )
    manifest = {
        "schema": "c-pbe-reference-gate-branch-complete",
        "version": 1,
        "status": "BRANCH_COMPLETE",
        "branch": branch,
        "phase_order": list(BRANCH_PHASES[branch]),
        "phase_complete_sha256": phase_hashes,
        "restart_provenance_sha256": restart_hashes,
        "gate_profile": run["gate_profile"],
        **{name: run[name] for name in RUNTIME_RECORD_SPECS},
        "scheduler": run["scheduler"],
        "branch_run_provenance_sha256": _sha256_bytes(
            _read_regular(
                branch_root / "BRANCH_RUN_PROVENANCE.json",
                "branch run provenance",
                nonempty=True,
            )
        ),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "wall_seconds": float(wall_seconds),
    }
    _atomic_write_json(branch_root / "BRANCH_COMPLETE.json", manifest, noreplace=True)
    return manifest


def _verify_execution_evidence(
    root: Path, results: dict[str, PhaseResult]
) -> dict[str, object] | None:
    failed_branches = [
        branch
        for branch in BRANCH_PHASES
        if os.path.lexists(root / "runs" / branch / "RUN_FAILED.json")
    ]
    if failed_branches:
        raise ValueError(
            "RUN_FAILED Task4 evidence exists for: " + ", ".join(failed_branches)
        )
    evidence_paths = []
    for branch, phases in BRANCH_PHASES.items():
        branch_root = root / "runs" / branch
        evidence_paths.extend(
            (
                branch_root / "BRANCH_RUN_PROVENANCE.json",
                branch_root / "BRANCH_COMPLETE.json",
            )
        )
        for phase in phases:
            evidence_paths.append(branch_root / phase / "PHASE_COMPLETE.json")
            if _phase_spec(branch, phase).restart:
                evidence_paths.append(branch_root / phase / "RESTART_PROVENANCE.json")
    present = [path for path in evidence_paths if os.path.lexists(path)]
    if not present:
        return None
    if len(present) != len(evidence_paths):
        raise ValueError("Task4 execution evidence is incomplete")

    branch_summaries = {}
    gate_profiles = []
    runtime_records = {name: [] for name in RUNTIME_RECORD_SPECS}
    array_job_ids = []
    preparation_signatures = []
    for branch, phases in BRANCH_PHASES.items():
        _, preparation = _branch_provenance(root, branch)
        preparation_signatures.append(_preparation_signature(preparation, branch))
        branch_root, run = _run_provenance(root, branch)
        gate_profiles.append(run["gate_profile"])
        for name in RUNTIME_RECORD_SPECS:
            runtime_records[name].append(run[name])
        array_job_ids.append(run["scheduler"]["array_job_id"])
        for phase in phases:
            _verify_phase_evidence(
                root, branch, phase, results[f"runs/{branch}/{phase}"]
            )
        complete_path = branch_root / "BRANCH_COMPLETE.json"
        complete = _load_json(complete_path, f"{branch} completion")
        if (
            complete.get("schema") != "c-pbe-reference-gate-branch-complete"
            or complete.get("status") != "BRANCH_COMPLETE"
            or complete.get("branch") != branch
            or complete.get("phase_order") != list(phases)
            or complete.get("scheduler") != run["scheduler"]
            or complete.get("gate_profile") != run["gate_profile"]
            or any(
                complete.get(name) != run[name]
                for name in RUNTIME_RECORD_SPECS
            )
        ):
            raise ValueError(f"{branch} completion manifest identity is invalid")
        expected_phase_hashes = {
            phase: _sha256_bytes(
                _read_regular(
                    branch_root / phase / "PHASE_COMPLETE.json",
                    "phase completion",
                    nonempty=True,
                )
            )
            for phase in phases
        }
        expected_restart_hashes = {
            phase: _sha256_bytes(
                _read_regular(
                    branch_root / phase / "RESTART_PROVENANCE.json",
                    "restart provenance",
                    nonempty=True,
                )
            )
            for phase in phases
            if _phase_spec(branch, phase).restart
        }
        if (
            complete.get("phase_complete_sha256") != expected_phase_hashes
            or complete.get("restart_provenance_sha256") != expected_restart_hashes
        ):
            raise ValueError(f"{branch} completion manifest hashes are invalid")
        run_hash = _sha256_bytes(
            _read_regular(
                branch_root / "BRANCH_RUN_PROVENANCE.json",
                "run provenance",
                nonempty=True,
            )
        )
        if complete.get("branch_run_provenance_sha256") != run_hash:
            raise ValueError(
                f"{branch} run provenance hash differs from branch completion"
            )
        started = _parse_utc(complete.get("started_utc"), "branch start")
        ended = _parse_utc(complete.get("ended_utc"), "branch end")
        if (
            ended < started
            or not isinstance(complete.get("wall_seconds"), (int, float))
            or complete["wall_seconds"] < 0
        ):
            raise ValueError(f"{branch} completion timing is invalid")
        branch_summaries[branch] = {
            "branch_complete_sha256": _sha256_bytes(
                _read_regular(complete_path, "branch completion", nonempty=True)
            ),
            "wall_seconds": complete["wall_seconds"],
            "scheduler": complete["scheduler"],
        }
    if any(profile != gate_profiles[0] for profile in gate_profiles[1:]):
        raise ValueError("C PBE gate profile differs across branches")
    for name, records in runtime_records.items():
        if any(record != records[0] for record in records[1:]):
            raise ValueError(f"{name} provenance differs across branches")
    if any(job_id != array_job_ids[0] for job_id in array_job_ids[1:]):
        raise ValueError("branches do not belong to one Slurm array job")
    if any(
        signature != preparation_signatures[0]
        for signature in preparation_signatures[1:]
    ):
        raise ValueError(
            "asset content or frozen preparation identity differs across branches"
        )
    return {
        "status": "RESTART_CHAIN_VERIFIED",
        "branches": branch_summaries,
        "gate_profile": gate_profiles[0],
        **{name: records[0] for name, records in runtime_records.items()},
        "preparation": preparation_signatures[0],
    }


def _phase_dict(phase: PhaseResult, root: Path) -> dict[str, object]:
    return {
        "path": str(Path(phase.path).relative_to(root)),
        "expected_mode": phase.expected_mode,
        "expected_restart": phase.expected_restart,
        "expected_field_dir": phase.expected_field_dir,
        "energy_ev": phase.energy_ev,
        "energy_ha": phase.energy_ha,
        "spin_counts": {str(key): value for key, value in phase.spin_counts.items()},
        "occupations": {
            str(key): list(values) for key, values in phase.occupations.items()
        },
        "integer_occupations": phase.integer_occupations,
        "file_sha256": dict(phase.file_hashes),
        "stage_sha256": phase.stage_hash,
    }


def audit_gate(root: str | Path) -> dict[str, object]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"gate root does not exist: {root_path}")

    phases = {
        spec.relative: audit_phase(
            root_path / spec.relative,
            expected_mode=spec.mode,
            expected_restart=spec.restart,
            expected_field_dir=spec.field_dir,
        )
        for spec in PHASES
    }
    fixed_cold = phases["runs/fixed/fixed_cold"]
    fixed_restart = phases["runs/fixed/fixed_restart"]
    fixed_drift = abs(fixed_restart.energy_ha - fixed_cold.energy_ha) * HA_TO_KCAL_MOL
    free_energies = {
        direction: phases[f"runs/dir{direction}/free_restart2"].energy_ha
        for direction in range(3)
    }
    free_drifts = {
        direction: abs(
            phases[f"runs/dir{direction}/free_restart2"].energy_ha
            - phases[f"runs/dir{direction}/free_restart1"].energy_ha
        )
        * HA_TO_KCAL_MOL
        for direction in range(3)
    }
    comparison = compare_zero_field_results(
        fixed_energy_ha=fixed_restart.energy_ha,
        free_energies_ha=free_energies,
        fixed_drift_kcal=fixed_drift,
        free_drifts_kcal=free_drifts,
    )
    execution_evidence = _verify_execution_evidence(root_path, phases)
    evidence_complete = execution_evidence is not None
    return {
        "status": "PBE_GATE_PASSED" if evidence_complete else "DIAGNOSTIC_ONLY",
        "zero_field_comparison_status": comparison["status"],
        "authoritative_result": AUTHORITATIVE_RESULT,
        "restart_chain_evidence": execution_evidence
        or {
            "status": RESTART_CHAIN_STATUS,
            "note": RESTART_CHAIN_NOTE,
        },
        "blocked_on": None if evidence_complete else "restart_chain_evidence",
        "phases": {
            spec.relative: _phase_dict(phases[spec.relative], root_path)
            for spec in PHASES
        },
        "comparison": comparison,
    }


def _summary_text(summary: dict[str, object]) -> str:
    lines = [
        f"status={summary['status']}",
        f"authoritative_result={AUTHORITATIVE_RESULT}",
    ]
    if summary["status"] == "PBE_GATE_FAILED":
        lines.append(f"error={summary.get('error', 'unknown audit failure')}")
        return "\n".join(lines) + "\n"
    if summary["status"] not in {"DIAGNOSTIC_ONLY", "PBE_GATE_PASSED"}:
        raise ValueError(f"unsupported audit status: {summary['status']}")

    lines.append(
        "zero_field_comparison_status=" f"{summary['zero_field_comparison_status']}"
    )
    restart_status = summary["restart_chain_evidence"]["status"]
    lines.append(f"restart_chain_evidence={restart_status}")
    lines.append(f"blocked_on={summary['blocked_on']}")
    if summary["status"] == "DIAGNOSTIC_ONLY":
        lines.append(f"restart_chain_note={RESTART_CHAIN_NOTE}")
    phases = summary["phases"]
    for spec in PHASES:
        phase = phases[spec.relative]
        counts = phase["spin_counts"]
        occupations = phase["occupations"]
        spin1_occupations = ",".join(f"{value:.16g}" for value in occupations["1"])
        spin2_occupations = ",".join(f"{value:.16g}" for value in occupations["2"])
        lines.append(
            f"phase={spec.relative} energy_ev={phase['energy_ev']:.16g} "
            f"energy_ha={phase['energy_ha']:.16g} spin1={counts['1']:.1f} "
            f"spin2={counts['2']:.1f} integer_occupations="
            f"{str(phase['integer_occupations']).lower()} "
            f"spin1_occupations={spin1_occupations} "
            f"spin2_occupations={spin2_occupations} "
            f"INPUT_sha256={phase['file_sha256']['INPUT']} "
            f"running_scf.log_sha256="
            f"{phase['file_sha256']['running_scf.log']} "
            f"eig_occ.txt_sha256={phase['file_sha256']['eig_occ.txt']} "
            f"stage_sha256={phase['stage_sha256']}"
        )

    comparison = summary["comparison"]
    lines.append(f"fixed_drift_kcal={comparison['fixed_drift_kcal']:.16g}")
    for direction in range(3):
        lines.append(
            f"free_direction_{direction}_drift_kcal="
            f"{comparison['free_drifts_kcal'][direction]:.16g}"
        )
        lines.append(
            f"fixed_free_direction_{direction}_difference_ha="
            f"{comparison['fixed_free_differences_ha'][direction]:.16g}"
        )
    for pair, difference in comparison["free_pair_differences_ha"].items():
        lines.append(f"free_pair_{pair}_difference_ha={difference:.16g}")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def _summary_paths(root: Path) -> tuple[Path, Path]:
    return root / AUTHORITATIVE_RESULT, root / TEXT_SUMMARY


def _best_effort_remove_summaries(root: Path) -> None:
    for path in _summary_paths(root):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def invalidate_summaries(root: Path) -> None:
    errors = []
    for path in _summary_paths(root):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(exc)
    if errors:
        raise OSError(f"cannot invalidate old audit summaries: {errors[0]}")


def write_summaries(root: Path, summary: dict[str, object]) -> None:
    authoritative_path, text_path = _summary_paths(root)
    try:
        json_content = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        text_content = _summary_text(summary)
        _atomic_write(text_path, text_content)
        # JSON is the sole authority and is therefore published last.
        _atomic_write(authoritative_path, json_content)
    except Exception:
        # Cleanup is best-effort; the original exception is always re-raised.
        _best_effort_remove_summaries(root)
        raise


def _runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal Task4 runner evidence operations"
    )
    subparsers = parser.add_subparsers(dest="runner_command", required=True)

    scheduler = subparsers.add_parser("check-scheduler")
    scheduler.add_argument("--branch", required=True, choices=BRANCH_PHASES)

    initialize = subparsers.add_parser("runner-init")
    initialize.add_argument("--root", required=True)
    initialize.add_argument("--branch", required=True, choices=BRANCH_PHASES)
    initialize.add_argument("--gate-profile", required=True)
    initialize.add_argument("--python", required=True)
    initialize.add_argument("--prepare-gate", required=True)
    initialize.add_argument("--audit-gate", required=True)
    initialize.add_argument("--gate-contract", required=True)
    initialize.add_argument("--resource-profiles", required=True)
    initialize.add_argument("--entrypoint", required=True)
    initialize.add_argument("--common-runner", required=True)
    initialize.add_argument("--abacus", required=True)
    initialize.add_argument("--environment-script", required=True)
    initialize.add_argument("--mpirun", required=True)

    preflight = subparsers.add_parser("preflight-phase")
    preflight.add_argument("--root", required=True)
    preflight.add_argument("--branch", required=True, choices=BRANCH_PHASES)
    preflight.add_argument("--phase", required=True)

    restart = subparsers.add_parser("create-restart")
    restart.add_argument("--root", required=True)
    restart.add_argument("--branch", required=True, choices=BRANCH_PHASES)
    restart.add_argument("--source", required=True)
    restart.add_argument("--destination", required=True)

    phase = subparsers.add_parser("complete-phase")
    phase.add_argument("--root", required=True)
    phase.add_argument("--branch", required=True, choices=BRANCH_PHASES)
    phase.add_argument("--phase", required=True)
    phase.add_argument("--started-utc", required=True)
    phase.add_argument("--ended-utc", required=True)
    phase.add_argument("--wall-seconds", required=True, type=float)

    branch = subparsers.add_parser("complete-branch")
    branch.add_argument("--root", required=True)
    branch.add_argument("--branch", required=True, choices=BRANCH_PHASES)
    branch.add_argument("--started-utc", required=True)
    branch.add_argument("--ended-utc", required=True)
    branch.add_argument("--wall-seconds", required=True, type=float)
    return parser


def _runner_main(argv: list[str]) -> int:
    parser = _runner_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.runner_command == "check-scheduler":
            result = _scheduler_record(arguments.branch)
        elif arguments.runner_command == "runner-init":
            result = initialize_branch_run(
                arguments.root,
                arguments.branch,
                arguments.gate_profile,
                arguments.python,
                arguments.prepare_gate,
                arguments.audit_gate,
                arguments.gate_contract,
                arguments.resource_profiles,
                arguments.entrypoint,
                arguments.common_runner,
                arguments.abacus,
                arguments.environment_script,
                arguments.mpirun,
            )
        elif arguments.runner_command == "preflight-phase":
            preflight_phase(arguments.root, arguments.branch, arguments.phase)
            result = {"status": "PREFLIGHT_OK"}
        elif arguments.runner_command == "create-restart":
            path = create_restart_phase(
                arguments.root,
                arguments.branch,
                arguments.source,
                arguments.destination,
            )
            result = {"status": "RESTART_PREPARED", "path": str(path)}
        elif arguments.runner_command == "complete-phase":
            result = complete_phase(
                arguments.root,
                arguments.branch,
                arguments.phase,
                arguments.started_utc,
                arguments.ended_utc,
                arguments.wall_seconds,
            )
        else:
            result = complete_branch(
                arguments.root,
                arguments.branch,
                arguments.started_utc,
                arguments.ended_utc,
                arguments.wall_seconds,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv and effective_argv[0] in {
        "check-scheduler",
        "runner-init",
        "preflight-phase",
        "create-restart",
        "complete-phase",
        "complete-branch",
    }:
        return _runner_main(effective_argv)
    parser = argparse.ArgumentParser(description="Audit the C atom PBE reference gate")
    parser.add_argument("root_positional", nargs="?", help="gate root directory")
    parser.add_argument("--root", dest="root_option", help="gate root directory")
    args = parser.parse_args(effective_argv)
    if args.root_positional is not None and args.root_option is not None:
        parser.error("positional root and --root cannot be used together")
    root_argument = args.root_option or args.root_positional or "."
    root = Path(root_argument).resolve()

    try:
        if not root.is_dir():
            raise ValueError(f"gate root does not exist: {root}")
        invalidate_summaries(root)
        summary = audit_gate(root)
    except (ValueError, OSError) as exc:
        failure = {
            "status": "PBE_GATE_FAILED",
            "authoritative_result": AUTHORITATIVE_RESULT,
            "error": str(exc),
        }
        if root.is_dir():
            _best_effort_remove_summaries(root)
            try:
                write_summaries(root, failure)
            except OSError:
                _best_effort_remove_summaries(root)
        print(f"PBE gate audit failed: {exc}", file=sys.stderr)
        return 1

    try:
        write_summaries(root, summary)
    except OSError as exc:
        _best_effort_remove_summaries(root)
        print(f"PBE gate summary write failed: {exc}", file=sys.stderr)
        return 1
    print(f"status={summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
