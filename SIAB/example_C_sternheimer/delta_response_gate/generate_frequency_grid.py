#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from response_contract import parse_eig_occ, union_transition_window


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_greenx_frequency_file(path: Path, expected_size: int) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) == 2:
            omega_token, weight_token = fields
        elif len(fields) == 3:
            if int(fields[0]) != len(rows) + 1:
                raise ValueError("frequency indices must be consecutive from one")
            omega_token, weight_token = fields[1:]
        else:
            raise ValueError("frequency row must contain omega and weight")
        omega = float(omega_token)
        weight = float(weight_token)
        if not (math.isfinite(omega) and math.isfinite(weight)):
            raise ValueError("frequency rows must be finite")
        if omega <= 0.0 or weight <= 0.0:
            raise ValueError("frequency and weight must be positive")
        if rows and omega <= rows[-1][0]:
            raise ValueError("frequencies must be strictly increasing")
        rows.append((omega, weight))
    if len(rows) != expected_size:
        raise ValueError(f"expected {expected_size} frequency rows, found {len(rows)}")
    return rows


def generate_frequency_grid(
    *,
    fixed_eig_occ: Path,
    free_eig_occ: Path,
    greenx_executable: Path,
    output_dir: Path,
    nfreq: int = 6,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"frequency output already exists: {output_dir}")
    if nfreq != 6:
        raise ValueError("the C response equivalence gate requires nfreq=6")
    executable = greenx_executable.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("GreenX executable must be an executable regular file")

    fixed = parse_eig_occ(fixed_eig_occ)
    free = parse_eig_occ(free_eig_occ)
    transition_min, transition_max = union_transition_window((fixed, free))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        input_path = temporary / "inputs.dat"
        input_path.write_text(
            f"{temporary}/\n"
            f"n_mesh_points {nfreq}\n"
            f"e_transition_min {transition_min:.17g}\n"
            f"e_transition_max {transition_max:.17g}\n",
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
        rows = parse_greenx_frequency_file(temporary / "freq.dat", nfreq)
        grid = temporary / "fixed_frequency_grid.dat"
        with grid.open("w", encoding="ascii") as handle:
            handle.write(
                f"# transition_window_Ha {transition_min:.17E} {transition_max:.17E}\n"
                "# generator GreenX gx_minimax_grid nfreq6\n"
                "# index omega_Ha weight_Ha\n"
            )
            for index, (omega, weight) in enumerate(rows, 1):
                handle.write(f"{index} {omega:.17E} {weight:.17E}\n")
        manifest = {
            "status": "success",
            "nfreq": nfreq,
            "transition_window_ha": [transition_min, transition_max],
            "fixed_eig_occ": str(fixed.path),
            "fixed_eig_occ_sha256": sha256(fixed.path),
            "free_eig_occ": str(free.path),
            "free_eig_occ_sha256": sha256(free.path),
            "greenx_executable": str(executable),
            "greenx_executable_sha256": sha256(executable),
            "grid_sha256": sha256(grid),
        }
        (temporary / "FREQUENCY_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        temporary.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-eig-occ", required=True, type=Path)
    parser.add_argument("--free-eig-occ", required=True, type=Path)
    parser.add_argument("--greenx-executable", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nfreq", default=6, type=int)
    args = parser.parse_args()
    manifest = generate_frequency_grid(
        fixed_eig_occ=args.fixed_eig_occ,
        free_eig_occ=args.free_eig_occ,
        greenx_executable=args.greenx_executable,
        output_dir=args.output_dir,
        nfreq=args.nfreq,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
