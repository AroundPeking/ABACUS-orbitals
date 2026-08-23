#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

from producer_contract import render_siab_input, render_siab_stru


RESTART_NAMES = ("wfs1_nao.txt", "wfs2_nao.txt", "chgs1.cube", "chgs2.cube")
SOURCE_NAMES = (
    "STRU",
    "KPT",
    "C_ONCV_PBE-1.0.upf",
    "C_gga_10au_100Ry_3s3p2d.orb",
    "SOURCE_EIG_OCC.txt",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"size": path.stat().st_size, "sha256": sha256(path)}


def require_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    return path


def parse_gate_result(path: Path) -> dict[str, str]:
    values = {}
    for raw_line in require_regular(path, "response gate result").read_text(
        encoding="ascii"
    ).splitlines():
        if not raw_line.strip():
            continue
        if "=" not in raw_line:
            raise ValueError("response gate result contains a malformed line")
        key, value = raw_line.split("=", 1)
        if key in values:
            raise ValueError(f"response gate result repeats {key}")
        values[key] = value
    if values.get("status") != "DELTA_RESPONSE_GATE_PASSED" or values.get(
        "blocked_on"
    ) not in (None, "None"):
        raise ValueError("response gate has not passed")
    return values


def parse_frequency_grid(path: Path) -> tuple[tuple[float, float], ...]:
    rows = []
    for raw_line in require_regular(path, "frequency grid").read_text(
        encoding="ascii"
    ).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("frequency row must contain index, omega and weight")
        index = int(fields[0])
        omega, weight = map(float, fields[1:])
        if index != len(rows) + 1 or not all(
            math.isfinite(value) and value > 0.0 for value in (omega, weight)
        ):
            raise ValueError("frequency grid has invalid indices or values")
        if rows and omega <= rows[-1][0]:
            raise ValueError("frequency values must be strictly increasing")
        rows.append((omega, weight))
    if len(rows) != 16:
        raise ValueError(f"expected 16 frequency rows, found {len(rows)}")
    return tuple(rows)


def _require_manifest_hash(path: Path, record: dict, label: str) -> Path:
    require_regular(path, label)
    if record.get("sha256") != sha256(path) or record.get("size") != path.stat().st_size:
        raise ValueError(f"source hash mismatch for {label}")
    return path


def prepare_siab_reference(
    root: Path, response_gate_root: Path, frequency_grid: Path, abfs: Path
) -> dict:
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"SIAB reference target already exists: {root}")

    response_gate_root = response_gate_root.resolve(strict=True)
    result_path = response_gate_root / "DELTA_RESPONSE_GATE_RESULT.txt"
    gate_result = parse_gate_result(result_path)
    manifest_path = require_regular(
        response_gate_root / "PREPARATION_MANIFEST.json", "response manifest"
    )
    response_manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    fixed = response_manifest.get("branches", {}).get("fixed")
    if response_manifest.get("status") != "prepared" or not isinstance(fixed, dict):
        raise ValueError("response preparation manifest lacks the fixed branch")

    source_phase_value = fixed.get("source_phase_absolute")
    if not isinstance(source_phase_value, str) or not source_phase_value:
        raise ValueError("response manifest lacks the original PBE source phase")
    source_phase_path = Path(source_phase_value)
    if source_phase_path.is_symlink() or not source_phase_path.is_dir():
        raise ValueError("original PBE source phase must be a real directory")
    source_phase = source_phase_path.resolve(strict=True)
    source_restart = source_phase / "OUT.C_PBE_REFERENCE_GATE"
    if source_restart.is_symlink() or not source_restart.is_dir():
        raise ValueError("original PBE output must be a real directory")
    file_records = fixed.get("files", {})
    restart_records = fixed.get("restart_files", {})
    for name in SOURCE_NAMES:
        source = (
            source_restart / "eig_occ.txt"
            if name == "SOURCE_EIG_OCC.txt"
            else source_phase / name
        )
        _require_manifest_hash(source, file_records.get(name, {}), name)
    for name in RESTART_NAMES:
        _require_manifest_hash(
            source_restart / name, restart_records.get(name, {}), name
        )

    frequency_grid = frequency_grid.resolve(strict=True)
    frequencies = parse_frequency_grid(frequency_grid)
    abfs = require_regular(abfs.resolve(strict=True), "C auxiliary basis")

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    try:
        (temporary / "INPUT").write_text(render_siab_input(), encoding="ascii")
        source_stru = (source_phase / "STRU").read_text(encoding="ascii")
        (temporary / "STRU").write_text(
            render_siab_stru(source_stru, abfs.name), encoding="ascii"
        )
        for name in SOURCE_NAMES[1:-1]:
            shutil.copy2(source_phase / name, temporary / name)
        shutil.copy2(source_restart / "eig_occ.txt", temporary / "SOURCE_EIG_OCC.txt")
        shutil.copy2(abfs, temporary / abfs.name)
        shutil.copy2(frequency_grid, temporary / "fixed_frequency_grid_nfreq16.dat")

        restart = temporary / "OUT.C_SIAB_REFERENCE"
        restart.mkdir()
        for name in RESTART_NAMES:
            shutil.copy2(source_restart / name, restart / name)

        manifest = {
            "status": "prepared",
            "response_gate_root": str(response_gate_root),
            "response_gate_status": gate_result["status"],
            "response_gate_result_sha256": sha256(result_path),
            "response_manifest_sha256": sha256(manifest_path),
            "source_branch": "fixed",
            "source_phase": fixed.get("source_phase"),
            "source_phase_absolute": str(source_phase),
            "frequency_count": len(frequencies),
            "frequency_grid": file_record(temporary / "fixed_frequency_grid_nfreq16.dat"),
            "abfs": {"name": abfs.name, **file_record(temporary / abfs.name)},
            "files": {
                path.name: file_record(path)
                for path in sorted(temporary.iterdir())
                if path.is_file()
            },
            "restart_files": {
                path.name: file_record(path) for path in sorted(restart.iterdir())
            },
        }
        (temporary / "PREPARATION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        temporary.rename(root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--response-gate-root", required=True, type=Path)
    parser.add_argument("--frequency-grid", required=True, type=Path)
    parser.add_argument("--abfs", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare_siab_reference(
        args.root, args.response_gate_root, args.frequency_grid, args.abfs
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
