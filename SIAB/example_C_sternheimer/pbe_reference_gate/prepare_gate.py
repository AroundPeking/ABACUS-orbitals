#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .gate_contract import FROZEN_PROTOCOL, VALID_MODES, render_input
else:
    from gate_contract import FROZEN_PROTOCOL, VALID_MODES, render_input


BRANCHES = {
    "fixed": ("fixed", None),
    "dir0": ("field", 0),
    "dir1": ("field", 1),
    "dir2": ("field", 2),
}

_PHASE_NAMES = {"fixed": "fixed_cold", "field": "field_seed"}
_RESERVED_ASSET_NAMES = frozenset({"INPUT", "STRU", "KPT", "BRANCH_PROVENANCE.json"})
_SAFE_ASSET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]*$")
_BOX_ANGSTROM = 20.0
_BOX_BOHR = "37.79452249150619"
_ATOM_DIRECT = (0.5, 0.5, 0.5)
_KPT_TEXT = "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"
_GLOBAL_LOCK = ".gate-assets.prepare.lock"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
_SOURCE_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")


@dataclass(frozen=True)
class SourceAsset:
    path: Path
    basename: str
    content: bytes
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int


@dataclass
class PreparationLock:
    name: str
    descriptor: int


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _same_source_state(first: os.stat_result, second: os.stat_result) -> bool:
    return all(
        getattr(first, field) == getattr(second, field) for field in _SOURCE_STAT_FIELDS
    )


def _read_descriptor(descriptor: int) -> bytes:
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _after_source_read(_path: Path, _label: str) -> None:
    """Test seam for proving that a source replacement is detected."""


def _read_regular_source(path: str | Path, label: str) -> SourceAsset:
    argument = Path(path).expanduser().absolute()
    try:
        argument_before = argument.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} source does not exist: {argument}")
    if stat.S_ISLNK(argument_before.st_mode):
        raise ValueError(f"{label} source must not be a symlink: {argument}")
    if not stat.S_ISREG(argument_before.st_mode):
        raise ValueError(f"{label} source must be a regular file: {argument}")

    try:
        resolved = argument.resolve(strict=True)
        resolved_before = resolved.lstat()
    except FileNotFoundError as exc:
        raise ValueError(
            f"{label} source changed while being resolved: {argument}"
        ) from exc
    if stat.S_ISLNK(resolved_before.st_mode) or not stat.S_ISREG(
        resolved_before.st_mode
    ):
        raise ValueError(f"{label} resolved source is not a regular file: {resolved}")
    if not _same_identity(argument_before, resolved_before):
        raise ValueError(f"{label} source changed while being resolved: {argument}")

    descriptor = os.open(resolved, _READ_FLAGS)
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise ValueError(f"{label} source must be a regular file: {resolved}")
        if not _same_source_state(resolved_before, opened_before):
            raise ValueError(f"{label} source changed while being opened: {resolved}")
        content = _read_descriptor(descriptor)
        _after_source_read(resolved, label)
        opened_after = os.fstat(descriptor)
        resolved_after = resolved.lstat()
        argument_after = argument.lstat()
    except FileNotFoundError as exc:
        raise ValueError(
            f"{label} source changed while being read: {resolved}"
        ) from exc
    finally:
        os.close(descriptor)

    source_states = (
        opened_before,
        opened_after,
        resolved_after,
        argument_after,
    )
    if any(
        not _same_source_state(resolved_before, current) for current in source_states
    ):
        raise ValueError(f"{label} source changed while being read: {resolved}")
    return SourceAsset(
        path=resolved,
        basename=resolved.name,
        content=content,
        sha256=_sha256_bytes(content),
        size=len(content),
        device=opened_before.st_dev,
        inode=opened_before.st_ino,
        mtime_ns=opened_before.st_mtime_ns,
    )


def _validate_asset_names(pseudo: Path, orbital: Path) -> None:
    if pseudo.name == orbital.name:
        raise ValueError("pseudo and orbital basenames must be distinct")
    for label, path in (("pseudo", pseudo), ("orbital", orbital)):
        if not _SAFE_ASSET_NAME.fullmatch(path.name):
            raise ValueError(
                f"{label} basename must be a single safe token: {path.name!r}"
            )
        if path.name in _RESERVED_ASSET_NAMES:
            raise ValueError(f"{label} basename is reserved by the gate: {path.name}")


def _render_stru(pseudo_name: str, orbital_name: str) -> str:
    return (
        "ATOMIC_SPECIES\n"
        f"C 12.011 {pseudo_name}\n\n"
        "LATTICE_CONSTANT\n"
        f"{_BOX_BOHR}\n\n"
        "LATTICE_VECTORS\n"
        "1 0 0\n"
        "0 1 0\n"
        "0 0 1\n\n"
        "ATOMIC_POSITIONS\n"
        "Direct\n\n"
        "C\n"
        "0.0\n"
        "1\n"
        "0.5 0.5 0.5 0 0 0 mag 2.0\n\n"
        "NUMERICAL_ORBITAL\n"
        f"{orbital_name}\n"
    )


def _open_directory(path: Path, label: str) -> int:
    try:
        return os.open(path, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise ValueError(
            f"{label} must be a real, non-symlink directory: {path}"
        ) from exc


def _open_directory_at(parent_fd: int, name: str, label: str) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"{label} must be a real, non-symlink directory") from exc


def _entry_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _verify_directory_identities(root_path: Path, root_fd: int, runs_fd: int) -> None:
    try:
        root_entry = root_path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("root directory identity changed after opening") from exc
    root_opened = os.fstat(root_fd)
    if not stat.S_ISDIR(root_entry.st_mode) or not _same_identity(
        root_entry, root_opened
    ):
        raise ValueError("root directory identity changed after opening")

    runs_entry = _entry_stat(root_fd, "runs")
    runs_opened = os.fstat(runs_fd)
    if (
        runs_entry is None
        or not stat.S_ISDIR(runs_entry.st_mode)
        or not _same_identity(runs_entry, runs_opened)
    ):
        raise ValueError("runs directory identity changed after opening")


def _open_gate_directories(root: Path) -> tuple[int, int]:
    root.mkdir(parents=True, exist_ok=True)
    root_fd = _open_directory(root, "gate root")
    try:
        try:
            os.mkdir("runs", mode=0o755, dir_fd=root_fd)
        except FileExistsError:
            pass
        runs_fd = _open_directory_at(root_fd, "runs", "runs directory")
    except BaseException:
        os.close(root_fd)
        raise
    return root_fd, runs_fd


def _after_directory_fds_opened(_root_path: Path, _runs_path: Path) -> None:
    """Test seam for verifying path-swap rejection after directory open."""


def _after_preparation_lock_acquired(_branch: str) -> None:
    """Test seam for deterministic concurrent-preparation tests."""


def _before_publish(_runs_path: Path, _branch: str) -> None:
    """Test seam for failure and external-target races before publication."""


def _acquire_lock(runs_fd: int, name: str, branch: str) -> PreparationLock:
    try:
        descriptor = os.open(name, _WRITE_FLAGS, 0o600, dir_fd=runs_fd)
    except FileExistsError as exc:
        raise RuntimeError(f"active or stale preparation lock exists: {name}") from exc
    try:
        payload = f"pid={os.getpid()} branch={branch}\n".encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(name, dir_fd=runs_fd)
        except OSError:
            pass
        raise
    return PreparationLock(name=name, descriptor=descriptor)


def _release_lock(runs_fd: int, lock: PreparationLock) -> None:
    opened = os.fstat(lock.descriptor)
    entry = _entry_stat(runs_fd, lock.name)
    if entry is None or not _same_identity(opened, entry):
        raise RuntimeError(f"preparation lock identity changed: {lock.name}")
    os.unlink(lock.name, dir_fd=runs_fd)
    os.close(lock.descriptor)
    lock.descriptor = -1


def _close_lock(lock: PreparationLock | None) -> None:
    if lock is not None and lock.descriptor >= 0:
        os.close(lock.descriptor)
        lock.descriptor = -1


def _check_for_stale_preparation(runs_fd: int, branch: str) -> None:
    lock_names = [_GLOBAL_LOCK]
    lock_names.extend(f".{name}.prepare.lock" for name in BRANCHES)
    for lock_name in lock_names:
        if _entry_stat(runs_fd, lock_name) is not None:
            raise RuntimeError(f"active or stale preparation lock exists: {lock_name}")
    hidden_pattern = re.compile(r"^\.(?:fixed|dir0|dir1|dir2)\.prepare-[A-Za-z0-9]+$")
    hidden = sorted(
        name for name in os.listdir(runs_fd) if hidden_pattern.fullmatch(name)
    )
    if hidden:
        raise RuntimeError(
            "stale preparation directories require manual inspection: "
            + ", ".join(hidden)
        )


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while staging gate file")
        view = view[written:]


def _write_binary_file(directory_fd: int, name: str, content: bytes) -> None:
    descriptor = os.open(name, _WRITE_FLAGS, 0o644, dir_fd=directory_fd)
    try:
        _write_all(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_text_file(directory_fd: int, name: str, content: str) -> None:
    _write_binary_file(directory_fd, name, content.encode("utf-8"))


def _read_regular_at(
    directory_fd: int, name: str, label: str
) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        content = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
        if not _same_source_state(before, after):
            raise ValueError(f"{label} changed while being read")
        return content, after
    finally:
        os.close(descriptor)


def _file_record_at(phase_fd: int, name: str, relative_path: str) -> dict[str, object]:
    content, metadata = _read_regular_at(phase_fd, name, name)
    return {
        "relative_path": relative_path,
        "sha256": _sha256_bytes(content),
        "size": metadata.st_size,
    }


def _atomic_write_json_at(
    directory_fd: int, name: str, value: dict[str, object]
) -> None:
    temporary = f".{name}.{secrets.token_hex(8)}.tmp"
    try:
        serialized = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        _write_binary_file(directory_fd, temporary, serialized)
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _source_record(asset: SourceAsset) -> dict[str, object]:
    return {
        "absolute_path": str(asset.path),
        "basename": asset.basename,
        "sha256": asset.sha256,
        "size": asset.size,
        "device": asset.device,
        "inode": asset.inode,
        "mtime_ns": asset.mtime_ns,
    }


def _require_exact_keys(
    value: object, expected: set[str], label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields are incomplete or unexpected")
    return value


def _validate_existing_branch_provenance(
    runs_fd: int,
    existing_branch: str,
    pseudo: SourceAsset,
    orbital: SourceAsset,
) -> None:
    branch_fd = _open_directory_at(
        runs_fd, existing_branch, f"existing branch {existing_branch}"
    )
    try:
        try:
            raw, _ = _read_regular_at(
                branch_fd,
                "BRANCH_PROVENANCE.json",
                "BRANCH_PROVENANCE.json",
            )
            provenance = json.loads(raw.decode("utf-8"))
            top = _require_exact_keys(
                provenance,
                {
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
                },
                "provenance",
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"existing branch {existing_branch} provenance is missing, corrupt, or incomplete"
            ) from exc

        expected_mode, expected_field_dir = BRANCHES[existing_branch]
        if (
            top["schema"] != "c-pbe-reference-gate-branch"
            or top["version"] != 1
            or top["branch"] != existing_branch
            or top["mode"] != expected_mode
            or top["field_dir"] != expected_field_dir
            or top["box_angstrom"] != _BOX_ANGSTROM
            or top["atom_direct"] != list(_ATOM_DIRECT)
        ):
            raise ValueError(
                f"existing branch {existing_branch} provenance identity is invalid"
            )
        if top["frozen_protocol"] != dict(FROZEN_PROTOCOL):
            raise ValueError(
                f"existing branch {existing_branch} frozen_protocol does not match"
            )

        renderer = _require_exact_keys(
            top["renderer"],
            {"function", "mode", "field_dir", "restart"},
            "renderer provenance",
        )
        if renderer != {
            "function": "gate_contract.render_input",
            "mode": expected_mode,
            "field_dir": expected_field_dir,
            "restart": False,
        }:
            raise ValueError(
                f"existing branch {existing_branch} renderer provenance is invalid"
            )

        sources = _require_exact_keys(
            top["sources"], {"pseudo", "orbital"}, "source provenance"
        )
        source_fields = {
            "absolute_path",
            "basename",
            "sha256",
            "size",
            "device",
            "inode",
            "mtime_ns",
        }
        for label, current in (("pseudo", pseudo), ("orbital", orbital)):
            recorded = _require_exact_keys(
                sources[label], source_fields, f"{label} source provenance"
            )
            signature = (
                recorded["basename"],
                recorded["sha256"],
                recorded["size"],
            )
            if signature != (current.basename, current.sha256, current.size):
                raise ValueError(
                    f"{label} asset does not match existing branch {existing_branch} provenance"
                )

        phase_name = _PHASE_NAMES[expected_mode]
        phase = _require_exact_keys(
            top["phase"], {"relative_path", "files"}, "phase provenance"
        )
        expected_phase_path = f"runs/{existing_branch}/{phase_name}"
        if phase["relative_path"] != expected_phase_path:
            raise ValueError(
                f"existing branch {existing_branch} phase provenance is invalid"
            )
        expected_files = {
            "INPUT",
            "STRU",
            "KPT",
            pseudo.basename,
            orbital.basename,
        }
        files = phase["files"]
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ValueError(
                f"existing branch {existing_branch} phase file provenance is incomplete"
            )

        phase_fd = _open_directory_at(
            branch_fd,
            phase_name,
            f"existing phase {existing_branch}/{phase_name}",
        )
        try:
            for name in expected_files:
                record = _require_exact_keys(
                    files[name],
                    {"relative_path", "sha256", "size"},
                    f"{name} provenance",
                )
                content, metadata = _read_regular_at(
                    phase_fd, name, f"existing staged file {name}"
                )
                expected_relative = f"{expected_phase_path}/{name}"
                if (
                    record["relative_path"] != expected_relative
                    or record["sha256"] != _sha256_bytes(content)
                    or record["size"] != metadata.st_size
                ):
                    raise ValueError(
                        f"existing branch {existing_branch} staged file {name} does not match provenance"
                    )
        finally:
            os.close(phase_fd)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("existing branch"):
            raise
        raise ValueError(
            f"existing branch {existing_branch} provenance is invalid: {exc}"
        ) from exc
    finally:
        os.close(branch_fd)


def _validate_all_existing_branches(
    runs_fd: int, pseudo: SourceAsset, orbital: SourceAsset
) -> None:
    for existing_branch in BRANCHES:
        if _entry_stat(runs_fd, existing_branch) is not None:
            _validate_existing_branch_provenance(
                runs_fd, existing_branch, pseudo, orbital
            )


def _make_temporary_directory(runs_fd: int, branch: str) -> tuple[str, int]:
    for _ in range(100):
        name = f".{branch}.prepare-{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=runs_fd)
        except FileExistsError:
            continue
        return name, _open_directory_at(runs_fd, name, "temporary preparation")
    raise RuntimeError("could not reserve a unique preparation directory")


def _remove_tree_at(parent_fd: int, name: str) -> None:
    try:
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    try:
        for child in os.listdir(directory_fd):
            metadata = os.stat(child, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                _remove_tree_at(directory_fd, child)
            else:
                os.unlink(child, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _remove_published_if_owned(runs_fd: int, branch: str, staged_fd: int) -> None:
    entry = _entry_stat(runs_fd, branch)
    if entry is not None and _same_identity(entry, os.fstat(staged_fd)):
        _remove_tree_at(runs_fd, branch)


def _raise_rename_error(error_number: int, destination: str) -> None:
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _darwin_rename_noreplace(directory_fd: int, source: str, destination: str) -> bool:
    if sys.platform != "darwin":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameatx_np", None)
    if function is None:
        return False
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        directory_fd,
        os.fsencode(source),
        directory_fd,
        os.fsencode(destination),
        0x00000004,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.ENOTSUP, errno.EINVAL, errno.ENOSYS}:
        return False
    _raise_rename_error(error_number, destination)
    return False


def _linux_rename_noreplace(directory_fd: int, source: str, destination: str) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        return False
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        directory_fd,
        os.fsencode(source),
        directory_fd,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.ENOTSUP, errno.EINVAL, errno.ENOSYS}:
        return False
    _raise_rename_error(error_number, destination)
    return False


def _publish_noreplace(runs_fd: int, source: str, destination: str) -> None:
    if _darwin_rename_noreplace(runs_fd, source, destination):
        return
    if _linux_rename_noreplace(runs_fd, source, destination):
        return

    # Portable POSIX rename has no NOREPLACE flag. The global and per-branch
    # O_EXCL locks serialize this tool, and the checks immediately around this
    # call reject a pre-existing target. A non-cooperating external writer can
    # still race this fallback; Darwin and Linux use native exclusive rename.
    if _entry_stat(runs_fd, destination) is not None:
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)
    os.rename(
        source,
        destination,
        src_dir_fd=runs_fd,
        dst_dir_fd=runs_fd,
    )


def _verify_published_identity(runs_fd: int, branch: str, staged_fd: int) -> None:
    published_fd = _open_directory_at(runs_fd, branch, "published branch")
    try:
        if not _same_identity(os.fstat(staged_fd), os.fstat(published_fd)):
            raise RuntimeError(
                "published branch identity does not match staging directory"
            )
    finally:
        os.close(published_fd)


def _fsync_directory(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise


def prepare_branch(
    root: str | Path,
    branch: str,
    pseudo: str | Path,
    orbital: str | Path,
) -> Path:
    if branch not in BRANCHES:
        raise ValueError(f"unsupported branch: {branch}")

    root_path = Path(root).expanduser().absolute()
    runs_path = root_path / "runs"
    root_fd, runs_fd = _open_gate_directories(root_path)
    global_lock = None
    branch_lock = None
    temporary_name = None
    temporary_fd = None
    published = False
    try:
        _after_directory_fds_opened(root_path, runs_path)
        _verify_directory_identities(root_path, root_fd, runs_fd)
        if _entry_stat(runs_fd, branch) is not None:
            raise FileExistsError(
                errno.EEXIST,
                os.strerror(errno.EEXIST),
                str(runs_path / branch),
            )
        _check_for_stale_preparation(runs_fd, branch)

        global_lock = _acquire_lock(runs_fd, _GLOBAL_LOCK, branch)
        branch_lock = _acquire_lock(runs_fd, f".{branch}.prepare.lock", branch)
        if _entry_stat(runs_fd, branch) is not None:
            raise FileExistsError(
                errno.EEXIST,
                os.strerror(errno.EEXIST),
                str(runs_path / branch),
            )
        _after_preparation_lock_acquired(branch)
        _verify_directory_identities(root_path, root_fd, runs_fd)

        pseudo_path = Path(pseudo).expanduser().absolute()
        orbital_path = Path(orbital).expanduser().absolute()
        _validate_asset_names(pseudo_path, orbital_path)
        pseudo_asset = _read_regular_source(pseudo_path, "pseudo")
        orbital_asset = _read_regular_source(orbital_path, "orbital")
        if pseudo_asset.basename != pseudo_path.name:
            raise ValueError("pseudo resolved basename differs from requested basename")
        if orbital_asset.basename != orbital_path.name:
            raise ValueError(
                "orbital resolved basename differs from requested basename"
            )
        _validate_all_existing_branches(runs_fd, pseudo_asset, orbital_asset)

        temporary_name, temporary_fd = _make_temporary_directory(runs_fd, branch)
        mode, field_dir = BRANCHES[branch]
        phase_name = _PHASE_NAMES[mode]
        os.mkdir(phase_name, mode=0o700, dir_fd=temporary_fd)
        phase_fd = _open_directory_at(temporary_fd, phase_name, "temporary phase")
        try:
            input_text = render_input(mode=mode, field_dir=field_dir, restart=False)
            _write_text_file(phase_fd, "INPUT", input_text)
            _write_text_file(
                phase_fd,
                "STRU",
                _render_stru(pseudo_asset.basename, orbital_asset.basename),
            )
            _write_text_file(phase_fd, "KPT", _KPT_TEXT)
            _write_binary_file(phase_fd, pseudo_asset.basename, pseudo_asset.content)
            _write_binary_file(phase_fd, orbital_asset.basename, orbital_asset.content)
            names = (
                "INPUT",
                "STRU",
                "KPT",
                pseudo_asset.basename,
                orbital_asset.basename,
            )
            phase_prefix = f"runs/{branch}/{phase_name}"
            file_records = {
                name: _file_record_at(phase_fd, name, f"{phase_prefix}/{name}")
                for name in names
            }
            _fsync_directory(phase_fd)
        finally:
            os.close(phase_fd)

        provenance = {
            "schema": "c-pbe-reference-gate-branch",
            "version": 1,
            "branch": branch,
            "mode": mode,
            "field_dir": field_dir,
            "box_angstrom": _BOX_ANGSTROM,
            "atom_direct": list(_ATOM_DIRECT),
            "sources": {
                "pseudo": _source_record(pseudo_asset),
                "orbital": _source_record(orbital_asset),
            },
            "renderer": {
                "function": "gate_contract.render_input",
                "mode": mode,
                "field_dir": field_dir,
                "restart": False,
            },
            "frozen_protocol": dict(FROZEN_PROTOCOL),
            "phase": {
                "relative_path": f"runs/{branch}/{phase_name}",
                "files": file_records,
            },
        }
        _atomic_write_json_at(temporary_fd, "BRANCH_PROVENANCE.json", provenance)
        _fsync_directory(temporary_fd)

        _before_publish(runs_path, branch)
        _verify_directory_identities(root_path, root_fd, runs_fd)
        if _entry_stat(runs_fd, branch) is not None:
            raise FileExistsError(
                errno.EEXIST,
                os.strerror(errno.EEXIST),
                str(runs_path / branch),
            )
        _publish_noreplace(runs_fd, temporary_name, branch)
        published = True
        _verify_published_identity(runs_fd, branch, temporary_fd)
        _verify_directory_identities(root_path, root_fd, runs_fd)
        _fsync_directory(runs_fd)

        _release_lock(runs_fd, branch_lock)
        branch_lock = None
        _release_lock(runs_fd, global_lock)
        global_lock = None
        return runs_path / branch
    except BaseException:
        try:
            if published:
                _remove_published_if_owned(runs_fd, branch, temporary_fd)
            elif temporary_name is not None:
                _remove_tree_at(runs_fd, temporary_name)
        except OSError:
            pass
        for lock in (branch_lock, global_lock):
            if lock is not None and lock.descriptor >= 0:
                try:
                    _release_lock(runs_fd, lock)
                except (OSError, RuntimeError):
                    pass
        raise
    finally:
        _close_lock(branch_lock)
        _close_lock(global_lock)
        if temporary_fd is not None:
            os.close(temporary_fd)
        os.close(runs_fd)
        os.close(root_fd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare immutable C PBE equivalence-gate branches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", required=True)
    prepare_parser.add_argument("--branch", required=True, choices=BRANCHES)
    prepare_parser.add_argument("--pseudo", required=True)
    prepare_parser.add_argument("--orbital", required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--mode", required=True, choices=sorted(VALID_MODES))
    render_parser.add_argument("--field-dir", type=int)
    render_parser.add_argument("--restart", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "prepare":
            branch_path = prepare_branch(
                arguments.root,
                branch=arguments.branch,
                pseudo=arguments.pseudo,
                orbital=arguments.orbital,
            )
            print(branch_path)
        else:
            sys.stdout.write(
                render_input(
                    mode=arguments.mode,
                    field_dir=arguments.field_dir,
                    restart=arguments.restart,
                )
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
