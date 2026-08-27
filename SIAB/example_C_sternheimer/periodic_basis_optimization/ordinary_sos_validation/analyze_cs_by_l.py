#!/usr/bin/env python3
"""Stream reader-v1 LRI coefficients and resolve their norm by auxiliary l."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np


CS_V1_MARKER = -10267453
CS_V1_HEADER_SIZE = 28
CS_V1_RECORD_SIZE = 36


def read_uniform_basis_metadata(path: Path) -> dict[str, Any]:
    lines = [line.split() for line in path.read_text(encoding="ascii").splitlines() if line.split()]
    if len(lines) < 3 or len(lines[0]) < 2 or len(lines[1]) < 2 or len(lines[2]) < 2:
        raise ValueError(f"invalid ABACUS basis metadata: {path}")
    number_of_types = int(lines[0][0])
    if number_of_types != 1:
        raise ValueError("the Cs angular-momentum diagnostic currently requires one atom type")
    total_basis = int(lines[0][1])
    basis_per_atom = int(lines[1][1])
    number_of_radial_functions = int(lines[2][1])
    if len(lines) < 3 + number_of_radial_functions:
        raise ValueError(f"truncated ABACUS basis metadata: {path}")
    radial_angular_momenta = [
        int(lines[index][0]) for index in range(3, 3 + number_of_radial_functions)
    ]
    labels = [
        angular_momentum
        for angular_momentum in radial_angular_momenta
        for _ in range(2 * angular_momentum + 1)
    ]
    if any(value < 0 for value in radial_angular_momenta):
        raise ValueError(f"negative angular momentum in ABACUS basis metadata: {path}")
    if len(labels) != basis_per_atom or total_basis <= 0 or total_basis % basis_per_atom != 0:
        raise ValueError(f"inconsistent ABACUS basis dimensions: {path}")
    return {
        "total_basis": total_basis,
        "basis_per_atom": basis_per_atom,
        "radial_angular_momenta": radial_angular_momenta,
        "angular_momentum_labels": labels,
    }


def _read_cs_header(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_header = handle.read(CS_V1_HEADER_SIZE)
        if len(raw_header) != CS_V1_HEADER_SIZE:
            raise ValueError(f"truncated reader-v1 Cs header: {path}")
        marker, number_of_atoms, number_of_cells, number_of_blocks, reserved_blocks = struct.unpack(
            "=iiiqq", raw_header
        )
        if (
            marker != CS_V1_MARKER
            or number_of_atoms <= 0
            or number_of_cells < 0
            or number_of_blocks < 0
            or reserved_blocks < number_of_blocks
        ):
            raise ValueError(f"invalid reader-v1 Cs header: {path}")
        header_size = CS_V1_HEADER_SIZE + reserved_blocks * CS_V1_RECORD_SIZE
        if header_size > file_size:
            raise ValueError(f"invalid reader-v1 Cs header size: {path}")

        records = []
        for index in range(reserved_blocks):
            raw_record = handle.read(CS_V1_RECORD_SIZE)
            if len(raw_record) != CS_V1_RECORD_SIZE:
                raise ValueError(f"truncated reader-v1 Cs block table: {path}")
            ia1, ia2, cell_x, cell_y, cell_z, maximum_abs, offset = struct.unpack(
                "=5idq", raw_record
            )
            if index >= number_of_blocks:
                if any((ia1, ia2, cell_x, cell_y, cell_z, maximum_abs, offset)):
                    raise ValueError(f"nonzero reader-v1 Cs padding record: {path}")
                continue
            if (
                ia1 <= 0
                or ia1 > number_of_atoms
                or ia2 <= 0
                or ia2 > number_of_atoms
                or not math.isfinite(maximum_abs)
                or maximum_abs < 0.0
                or offset < header_size
                or offset >= file_size
            ):
                raise ValueError(f"invalid reader-v1 Cs block record: {path}")
            records.append(
                {
                    "ia1": ia1,
                    "ia2": ia2,
                    "cell": (cell_x, cell_y, cell_z),
                    "maximum_abs": maximum_abs,
                    "offset": offset,
                }
            )
    return {
        "number_of_atoms": number_of_atoms,
        "number_of_cells": number_of_cells,
        "number_of_blocks": number_of_blocks,
        "reserved_blocks": reserved_blocks,
        "header_size": header_size,
        "file_size": file_size,
        "records": records,
    }


def summarize_cs_by_l(
    path: Path,
    wavefunction_basis: dict[str, Any],
    auxiliary_basis: dict[str, Any],
    *,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    header = _read_cs_header(path)
    number_of_atoms = header["number_of_atoms"]
    n_i = wavefunction_basis["basis_per_atom"]
    n_j = n_i
    n_mu = auxiliary_basis["basis_per_atom"]
    if wavefunction_basis["total_basis"] != number_of_atoms * n_i:
        raise ValueError("wavefunction basis metadata and reader-v1 atom count differ")
    if auxiliary_basis["total_basis"] != number_of_atoms * n_mu:
        raise ValueError("auxiliary basis metadata and reader-v1 atom count differ")

    rows_per_block = n_i * n_j
    bytes_per_row = n_mu * 8
    rows_per_chunk = max(1, chunk_bytes // bytes_per_row)
    column_sum_squared = np.zeros(n_mu, dtype=np.float64)
    column_maximum_abs = np.zeros(n_mu, dtype=np.float64)
    total_rows = 0
    maximum_recorded_abs = 0.0
    ranges = []

    with path.open("rb") as handle:
        for record in header["records"]:
            payload_bytes = rows_per_block * bytes_per_row
            begin = record["offset"]
            end = begin + payload_bytes
            if end > header["file_size"]:
                raise ValueError(f"truncated reader-v1 Cs payload: {path}")
            ranges.append((begin, end))
            maximum_recorded_abs = max(maximum_recorded_abs, record["maximum_abs"])
            handle.seek(begin)
            rows_left = rows_per_block
            while rows_left:
                row_count = min(rows_left, rows_per_chunk)
                raw = handle.read(row_count * bytes_per_row)
                if len(raw) != row_count * bytes_per_row:
                    raise ValueError(f"truncated reader-v1 Cs payload: {path}")
                values = np.frombuffer(raw, dtype=np.float64).reshape(row_count, n_mu)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite reader-v1 Cs coefficient: {path}")
                column_sum_squared += np.einsum("ij,ij->j", values, values)
                column_maximum_abs = np.maximum(column_maximum_abs, np.max(np.abs(values), axis=0))
                total_rows += row_count
                rows_left -= row_count

    ranges.sort()
    if any(current[0] < previous[1] for previous, current in zip(ranges, ranges[1:])):
        raise ValueError(f"overlapping reader-v1 Cs payload blocks: {path}")

    labels = np.asarray(auxiliary_basis["angular_momentum_labels"], dtype=int)
    total_sum_squared = float(np.sum(column_sum_squared))
    if total_sum_squared <= 0.0 or not math.isfinite(total_sum_squared):
        raise ValueError(f"reader-v1 Cs coefficients have invalid zero norm: {path}")
    channels = []
    for angular_momentum in sorted(set(int(value) for value in labels)):
        mask = labels == angular_momentum
        sum_squared = float(np.sum(column_sum_squared[mask]))
        coefficient_count = total_rows * int(np.count_nonzero(mask))
        channels.append(
            {
                "l": angular_momentum,
                "radial_function_count": auxiliary_basis["radial_angular_momenta"].count(
                    angular_momentum
                ),
                "magnetic_component_count": int(np.count_nonzero(mask)),
                "coefficient_count": coefficient_count,
                "frobenius": math.sqrt(sum_squared),
                "rms": math.sqrt(sum_squared / coefficient_count),
                "maximum_abs": float(np.max(column_maximum_abs[mask])),
                "fraction_of_squared_norm": sum_squared / total_sum_squared,
            }
        )

    coefficient_count = total_rows * n_mu
    actual_maximum_abs = float(np.max(column_maximum_abs))
    return {
        "status": "success",
        "path": str(path.resolve()),
        "file_size_bytes": header["file_size"],
        "number_of_atoms": number_of_atoms,
        "number_of_cells": header["number_of_cells"],
        "number_of_blocks": header["number_of_blocks"],
        "reserved_blocks": header["reserved_blocks"],
        "wavefunction_basis_per_atom": n_i,
        "auxiliary_basis_per_atom": n_mu,
        "coefficient_count": coefficient_count,
        "frobenius": math.sqrt(total_sum_squared),
        "rms": math.sqrt(total_sum_squared / coefficient_count),
        "maximum_abs": actual_maximum_abs,
        "maximum_abs_from_block_table": maximum_recorded_abs,
        "maximum_abs_table_difference": abs(actual_maximum_abs - maximum_recorded_abs),
        "angular_momentum_channels": channels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-wfc", type=Path, required=True)
    parser.add_argument("--basis-aux", type=Path, required=True)
    parser.add_argument("--cs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()

    result = summarize_cs_by_l(
        args.cs,
        read_uniform_basis_metadata(args.basis_wfc),
        read_uniform_basis_metadata(args.basis_aux),
        chunk_bytes=args.chunk_mib * 1024 * 1024,
    )
    if args.label is not None:
        result["label"] = args.label
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
