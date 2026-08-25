#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_frequency_block(source: Path, expected_nfreq: int) -> list[tuple[float, float]]:
    if expected_nfreq <= 0:
        raise ValueError("expected_nfreq must be positive")
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    starts = [
        index for index, line in enumerate(lines) if line.strip() == "Frequency node & weight:"
    ]
    if len(starts) != 1:
        raise ValueError("LibRPA output must contain exactly one frequency block")

    rows: list[tuple[float, float]] = []
    terminated = False
    for line in lines[starts[0] + 1 :]:
        if line.strip() == "Time node & weight:":
            terminated = True
            break
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("malformed LibRPA frequency row")
        index = int(fields[0])
        if index != len(rows):
            raise ValueError("LibRPA frequency indices must be consecutive from zero")
        omega, weight = float(fields[1]), float(fields[2])
        if not all(math.isfinite(value) and value > 0.0 for value in (omega, weight)):
            raise ValueError("LibRPA frequencies and weights must be positive and finite")
        if rows and omega <= rows[-1][0]:
            raise ValueError("LibRPA frequencies must be strictly increasing")
        rows.append((omega, weight))
    if not terminated:
        raise ValueError("LibRPA frequency block has no Time node terminator")
    if len(rows) != expected_nfreq:
        raise ValueError(f"expected {expected_nfreq} frequency rows, found {len(rows)}")
    return rows


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def extract_frequency_grid(
    *, source: Path, output: Path, manifest: Path, expected_nfreq: int
) -> dict:
    source = source.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"frequency grid already exists: {output}")
    if manifest.exists() or manifest.is_symlink():
        raise FileExistsError(f"frequency manifest already exists: {manifest}")

    rows = _parse_frequency_block(source, expected_nfreq)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    lines = [
        f"# source_librpa_output {source}",
        f"# source_librpa_sha256 {source_sha256}",
        f"# nfreq {expected_nfreq}",
        "# index omega_Ha weight_Ha",
    ]
    lines.extend(
        f"{index} {omega:.17E} {weight:.17E}"
        for index, (omega, weight) in enumerate(rows, 1)
    )
    grid_data = ("\n".join(lines) + "\n").encode("ascii")
    grid_sha256 = _sha256_bytes(grid_data)
    payload = {
        "format_version": 1,
        "status": "success",
        "nfreq": expected_nfreq,
        "source": str(source),
        "source_sha256": source_sha256,
        "grid_sha256": grid_sha256,
        "frequencies_ha": [omega for omega, _ in rows],
        "weights_ha": [weight for _, weight in rows],
    }
    manifest_data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "ascii"
    )
    _atomic_write(output, grid_data)
    try:
        _atomic_write(manifest, manifest_data)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the exact LibRPA imaginary-frequency quadrature for ABACUS"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-nfreq", required=True, type=int)
    args = parser.parse_args()
    result = extract_frequency_grid(
        source=args.source,
        output=args.output,
        manifest=args.manifest,
        expected_nfreq=args.expected_nfreq,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
