#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from generate_frequency_grid import parse_greenx_frequency_file
from response_contract import ACCEPTED_PHASES, parse_eig_occ, render_response_input


RESTART_NAMES = ("wfs1_nao.txt", "wfs2_nao.txt", "chgs1.cube", "chgs2.cube")
COMMON_NAMES = (
    "STRU",
    "KPT",
    "C_ONCV_PBE-1.0.upf",
    "C_gga_10au_100Ry_3s3p2d.orb",
    "fixed_frequency_grid.dat",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"size": path.stat().st_size, "sha256": sha256(path)}


def require_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a nonempty regular file")
    return path


def _unique_output_directory(phase: Path) -> Path:
    outputs = [path for path in phase.glob("OUT.*") if path.is_dir() and not path.is_symlink()]
    if len(outputs) != 1:
        raise ValueError(f"expected one real OUT directory in {phase}, found {len(outputs)}")
    return outputs[0]


def prepare_response_gate(root: Path, pbe_gate_root: Path, frequency_grid: Path) -> dict:
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"response gate root already exists: {root}")
    pbe_gate_root = pbe_gate_root.resolve(strict=True)
    result_path = require_regular(pbe_gate_root / "RESULT_SUMMARY.json", "PBE result")
    result = json.loads(result_path.read_text(encoding="ascii"))
    if (
        result.get("status") != "PBE_GATE_PASSED"
        or result.get("zero_field_comparison_status") != "ZERO_FIELD_COMPARISON_PASSED"
        or result.get("blocked_on") is not None
    ):
        raise ValueError("PBE gate has not passed")
    frequency_grid = require_regular(frequency_grid.resolve(strict=True), "frequency grid")
    parse_greenx_frequency_file(frequency_grid, expected_size=6)

    sources = {}
    for branch, relative in ACCEPTED_PHASES.items():
        phase = (pbe_gate_root / relative).resolve(strict=True)
        if not str(phase).startswith(str(pbe_gate_root) + os.sep):
            raise ValueError("accepted phase escapes the PBE gate root")
        output = _unique_output_directory(phase)
        eig = require_regular(output / "eig_occ.txt", f"{branch} eig_occ")
        phase_record = result.get("phases", {}).get(relative)
        expected_hash = (phase_record or {}).get("file_sha256", {}).get("eig_occ.txt")
        if expected_hash != sha256(eig):
            raise ValueError(f"{branch} eig_occ hash mismatch")
        parse_eig_occ(eig)
        sources[branch] = (phase, output, eig)

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    try:
        branches = {}
        for branch, relative in ACCEPTED_PHASES.items():
            phase, output, eig = sources[branch]
            case = temporary / "branches" / branch
            case.mkdir(parents=True)
            restart_dir = case / "OUT.C_DELTA_RESPONSE_GATE"
            restart_dir.mkdir()
            (case / "INPUT").write_text(render_response_input(branch), encoding="ascii")
            for name in ("STRU", "KPT", "C_ONCV_PBE-1.0.upf", "C_gga_10au_100Ry_3s3p2d.orb"):
                shutil.copy2(require_regular(phase / name, f"{branch} {name}"), case / name)
            shutil.copy2(frequency_grid, case / "fixed_frequency_grid.dat")
            for name in RESTART_NAMES:
                shutil.copy2(
                    require_regular(output / name, f"{branch} {name}"), restart_dir / name
                )
            shutil.copy2(eig, case / "SOURCE_EIG_OCC.txt")
            branches[branch] = {
                "source_phase": relative,
                "source_phase_absolute": str(phase),
                "source_eig_occ_sha256": sha256(eig),
                "files": {
                    path.name: file_record(path)
                    for path in sorted(case.iterdir())
                    if path.is_file()
                },
                "restart_files": {
                    path.name: file_record(path)
                    for path in sorted(restart_dir.iterdir())
                    if path.is_file()
                },
            }

        fixed_dir = temporary / "branches/fixed"
        free_dir = temporary / "branches/free"
        common_files = {}
        for name in COMMON_NAMES:
            fixed_path = fixed_dir / name
            free_path = free_dir / name
            if sha256(fixed_path) != sha256(free_path):
                raise ValueError(f"fixed/free common file differs: {name}")
            common_files[name] = file_record(fixed_path)
        manifest = {
            "status": "prepared",
            "pbe_gate_root": str(pbe_gate_root),
            "pbe_result_sha256": sha256(result_path),
            "branches": branches,
            "common_files": common_files,
        }
        (temporary / "PREPARATION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        temporary.rename(root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pbe-gate-root", required=True, type=Path)
    parser.add_argument("--frequency-grid", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare_response_gate(args.root, args.pbe_gate_root, args.frequency_grid)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
