#!/usr/bin/env python3
"""Create a shape-correct zero Coulomb-v1 file for sparse q-star SOS runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import tempfile


COULOMB_V1_MARKER = -20129433


def _atom_pair(pair_index: int, atom_count: int) -> tuple[int, int]:
    cursor = 0
    for atom_i in range(atom_count):
        for atom_j in range(atom_i, atom_count):
            if cursor == pair_index:
                return atom_i, atom_j
            cursor += 1
    raise ValueError("Coulomb-v1 block has an invalid atom-pair index")


def _read_layout(data: bytes) -> dict:
    if len(data) < 24:
        raise ValueError("Coulomb-v1 header is truncated")
    marker, source_iq, naux, value_flag, natoms, nblocks = struct.unpack_from(
        "<6i", data, 0
    )
    if marker != COULOMB_V1_MARKER:
        raise ValueError("Coulomb-v1 marker is invalid")
    if source_iq <= 0 or naux <= 0 or natoms <= 0 or nblocks <= 0:
        raise ValueError("Coulomb-v1 header dimensions are invalid")
    if value_flag not in (0, 1):
        raise ValueError("Coulomb-v1 value flag is invalid")
    expected_blocks = natoms * (natoms + 1) // 2
    if nblocks != expected_blocks:
        raise ValueError("Coulomb-v1 block count is incomplete")

    dimensions_end = 24 + 4 * natoms
    table_end = dimensions_end + 12 * nblocks
    if len(data) < table_end:
        raise ValueError("Coulomb-v1 block table is truncated")
    atom_naux = struct.unpack_from(f"<{natoms}i", data, 24)
    if any(value <= 0 for value in atom_naux) or sum(atom_naux) != naux:
        raise ValueError("Coulomb-v1 atom dimensions are inconsistent")

    blocks = []
    seen = set()
    bytes_per_value = 16 if value_flag == 1 else 8
    for block in range(nblocks):
        pair_index, offset = struct.unpack_from("<iq", data, dimensions_end + 12 * block)
        if pair_index in seen:
            raise ValueError("Coulomb-v1 atom-pair index is duplicated")
        seen.add(pair_index)
        atom_i, atom_j = _atom_pair(pair_index, natoms)
        count = atom_naux[atom_i] * atom_naux[atom_j]
        size = bytes_per_value * count
        if offset < table_end or offset + size > len(data):
            raise ValueError("Coulomb-v1 payload is truncated or overlaps the header")
        blocks.append((offset, size, count))
    return {
        "source_iq": source_iq,
        "naux": naux,
        "value_flag": value_flag,
        "natoms": natoms,
        "nblocks": nblocks,
        "atom_naux": atom_naux,
        "blocks": blocks,
    }


def write_zero_coulomb_v1(source: Path, output: Path, *, iq: int) -> dict:
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if type(iq) is not int or iq <= 0:
        raise ValueError("target q-point index must be a positive integer")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("output parent directory does not exist")

    data = bytearray(source.read_bytes())
    layout = _read_layout(data)
    struct.pack_into("<i", data, 4, iq)
    zeroed_values = 0
    for offset, size, count in layout["blocks"]:
        data[offset : offset + size] = bytes(size)
        zeroed_values += count

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "status": "success",
        "source": str(source),
        "output": str(output),
        "source_iq": layout["source_iq"],
        "iq": iq,
        "naux": layout["naux"],
        "natoms": layout["natoms"],
        "nblocks": layout["nblocks"],
        "value_flag": layout["value_flag"],
        "atom_naux": list(layout["atom_naux"]),
        "zeroed_complex_values": zeroed_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iq", required=True, type=int)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    report = write_zero_coulomb_v1(args.source, args.output, iq=args.iq)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.manifest is not None:
        manifest = args.manifest.resolve()
        if manifest.exists() or manifest.is_symlink():
            raise FileExistsError(manifest)
        if not manifest.parent.is_dir():
            raise ValueError("manifest parent directory does not exist")
        manifest.write_text(text, encoding="ascii")
    print(text, end="")


if __name__ == "__main__":
    main()
