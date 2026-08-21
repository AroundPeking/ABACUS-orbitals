#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
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
_RESERVED_ASSET_NAMES = frozenset(
    {"INPUT", "STRU", "KPT", "BRANCH_PROVENANCE.json"}
)
_BOX_ANGSTROM = 20.0
_BOX_BOHR = "37.79452249150619"
_ATOM_DIRECT = (0.5, 0.5, 0.5)
_KPT_TEXT = "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_regular_source(path: str | Path, label: str) -> tuple[Path, bytes]:
    source = Path(path).expanduser().absolute()
    try:
        before = source.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} source does not exist: {source}")
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} source must not be a symlink: {source}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} source must be a regular file: {source}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} source must be a regular file: {source}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} source changed while being opened: {source}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read()
    finally:
        os.close(descriptor)
    return source.resolve(strict=True), content


def _validate_asset_names(pseudo: Path, orbital: Path) -> None:
    if pseudo.name == orbital.name:
        raise ValueError("pseudo and orbital basenames must be distinct")
    for label, path in (("pseudo", pseudo), ("orbital", orbital)):
        if path.name in _RESERVED_ASSET_NAMES:
            raise ValueError(
                f"{label} basename is reserved by the gate: {path.name}"
            )


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


def _write_text_file(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_binary_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


def _file_record(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256_bytes(content),
        "size": len(content),
    }


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _create_parent_directories(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError(f"gate root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f"gate root must be a directory: {root}")

    runs = root / "runs"
    if runs.is_symlink():
        raise ValueError(f"runs directory must not be a symlink: {runs}")
    runs.mkdir(exist_ok=True)
    if not runs.is_dir():
        raise ValueError(f"runs path must be a directory: {runs}")
    return runs


def prepare_branch(
    root: str | Path,
    branch: str,
    pseudo: str | Path,
    orbital: str | Path,
) -> Path:
    if branch not in BRANCHES:
        raise ValueError(f"unsupported branch: {branch}")

    root_path = Path(root).expanduser().absolute()
    runs = _create_parent_directories(root_path)
    branch_path = runs / branch
    branch_path.mkdir()

    try:
        pseudo_path = Path(pseudo).expanduser().absolute()
        orbital_path = Path(orbital).expanduser().absolute()
        _validate_asset_names(pseudo_path, orbital_path)
        pseudo_source, pseudo_content = _read_regular_source(
            pseudo_path, "pseudo"
        )
        orbital_source, orbital_content = _read_regular_source(
            orbital_path, "orbital"
        )

        mode, field_dir = BRANCHES[branch]
        phase_name = _PHASE_NAMES[mode]
        phase = branch_path / phase_name
        phase.mkdir()

        input_path = phase / "INPUT"
        stru_path = phase / "STRU"
        kpt_path = phase / "KPT"
        pseudo_copy = phase / pseudo_path.name
        orbital_copy = phase / orbital_path.name
        _write_text_file(
            input_path,
            render_input(
                mode=mode,
                field_dir=field_dir,
                restart=False,
            ),
        )
        _write_text_file(
            stru_path,
            _render_stru(pseudo_path.name, orbital_path.name),
        )
        _write_text_file(kpt_path, _KPT_TEXT)
        _write_binary_file(pseudo_copy, pseudo_content)
        _write_binary_file(orbital_copy, orbital_content)

        staged_files = (
            input_path,
            stru_path,
            kpt_path,
            pseudo_copy,
            orbital_copy,
        )
        provenance = {
            "schema": "c-pbe-reference-gate-branch",
            "version": 1,
            "branch": branch,
            "mode": mode,
            "field_dir": field_dir,
            "box_angstrom": _BOX_ANGSTROM,
            "atom_direct": list(_ATOM_DIRECT),
            "sources": {
                "pseudo": {
                    "absolute_path": str(pseudo_source),
                    "sha256": _sha256_bytes(pseudo_content),
                    "size": len(pseudo_content),
                },
                "orbital": {
                    "absolute_path": str(orbital_source),
                    "sha256": _sha256_bytes(orbital_content),
                    "size": len(orbital_content),
                },
            },
            "renderer": {
                "function": "gate_contract.render_input",
                "mode": mode,
                "field_dir": field_dir,
                "restart": False,
            },
            "frozen_protocol": dict(FROZEN_PROTOCOL),
            "phase": {
                "relative_path": phase.relative_to(root_path).as_posix(),
                "files": {
                    path.name: _file_record(path, root_path)
                    for path in staged_files
                },
            },
        }
        _atomic_write_json(branch_path / "BRANCH_PROVENANCE.json", provenance)
    except BaseException:
        shutil.rmtree(branch_path)
        raise

    return branch_path


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
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
