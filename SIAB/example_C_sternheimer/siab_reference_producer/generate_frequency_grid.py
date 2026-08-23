#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DELTA_GATE_DIR = Path(__file__).resolve().parents[1] / "delta_response_gate"
sys.path.insert(0, str(DELTA_GATE_DIR))
from response_contract import parse_eig_occ


NFREQ = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_greenx_grid(path: Path) -> tuple[tuple[float, float], ...]:
    rows = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) == 2:
            omega_token, weight_token = fields
        elif len(fields) == 3:
            if int(fields[0]) != len(rows) + 1:
                raise ValueError("GreenX frequency indices must be consecutive")
            omega_token, weight_token = fields[1:]
        else:
            raise ValueError("GreenX frequency row must contain omega and weight")
        omega, weight = float(omega_token), float(weight_token)
        if not all(math.isfinite(value) and value > 0.0 for value in (omega, weight)):
            raise ValueError("GreenX frequency and weight must be positive and finite")
        if rows and omega <= rows[-1][0]:
            raise ValueError("GreenX frequencies must be strictly increasing")
        rows.append((omega, weight))
    if len(rows) != NFREQ:
        raise ValueError(f"expected {NFREQ} GreenX rows, found {len(rows)}")
    return tuple(rows)


def generate_frequency_grid(
    *, eig_occ: Path, greenx_executable: Path, output_dir: Path
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"frequency output already exists: {output_dir}")
    record = parse_eig_occ(eig_occ)
    executable = greenx_executable.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("GreenX executable must be an executable regular file")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        input_path = temporary / "inputs.dat"
        input_path.write_text(
            f"{temporary}/\n"
            f"n_mesh_points {NFREQ}\n"
            f"e_transition_min {record.transition_min_ha:.17g}\n"
            f"e_transition_max {record.transition_max_ha:.17g}\n",
            encoding="ascii",
        )
        result = subprocess.run(
            [str(executable), str(input_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        (temporary / "greenx.stdout").write_text(result.stdout, encoding="ascii")
        (temporary / "greenx.stderr").write_text(result.stderr, encoding="ascii")
        rows = parse_greenx_grid(temporary / "freq.dat")
        grid = temporary / "fixed_frequency_grid_nfreq16.dat"
        with grid.open("w", encoding="ascii") as handle:
            handle.write(
                f"# transition_window_Ha {record.transition_min_ha:.17E} "
                f"{record.transition_max_ha:.17E}\n"
                "# generator GreenX gx_minimax_grid nfreq16\n"
                "# source accepted field-free fixed-occupation C triplet\n"
                "# index omega_Ha weight_Ha\n"
            )
            for index, (omega, weight) in enumerate(rows, 1):
                handle.write(f"{index} {omega:.17E} {weight:.17E}\n")
        manifest = {
            "status": "success",
            "nfreq": NFREQ,
            "transition_window_ha": [
                record.transition_min_ha,
                record.transition_max_ha,
            ],
            "eig_occ": str(record.path),
            "eig_occ_sha256": sha256(record.path),
            "greenx_executable": str(executable),
            "greenx_executable_sha256": sha256(executable),
            "grid_sha256": sha256(grid),
        }
        (temporary / "FREQUENCY_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        temporary.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eig-occ", required=True, type=Path)
    parser.add_argument("--greenx-executable", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = generate_frequency_grid(
        eig_occ=args.eig_occ,
        greenx_executable=args.greenx_executable,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
