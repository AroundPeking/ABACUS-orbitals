#!/usr/bin/env python3
"""Extract selected frequencies from a SIAB response without loading full Q."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _scan_layout(path):
    header = {}
    metadata = []
    section = None
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("<") and stripped.endswith(">"):
                if stripped.startswith("</"):
                    section = None
                else:
                    section = stripped[1:-1]
                continue
            if not stripped or stripped.startswith("#"):
                continue
            if section == "STERNHEIMER_SIAB_HEADER":
                fields = stripped.split(None, 1)
                if len(fields) == 2:
                    header[fields[0]] = fields[1]
            elif section == "REFERENCE_METADATA":
                fields = stripped.split()
                if len(fields) != 6:
                    raise ValueError("REFERENCE_METADATA row must have 6 fields")
                try:
                    frequency = float(fields[2])
                except ValueError as error:
                    raise ValueError("invalid REFERENCE_METADATA frequency") from error
                if not math.isfinite(frequency):
                    raise ValueError("REFERENCE_METADATA frequency must be finite")
                metadata.append((line, frequency))

    try:
        n_reference = int(header["n_reference"])
        n_primitive = int(header["n_primitive"])
    except (KeyError, ValueError) as error:
        raise ValueError("invalid SIAB response header") from error
    if n_reference <= 0 or n_primitive <= 0:
        raise ValueError("SIAB response dimensions must be positive")
    if len(metadata) != n_reference:
        raise ValueError(
            f"REFERENCE_METADATA expected {n_reference} rows, found {len(metadata)}"
        )
    return n_reference, n_primitive, metadata


def extract_frequency_subset(input_path, output_path, *, frequency_index):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if (
        not input_path.is_file()
        or input_path.is_symlink()
        or input_path.stat().st_size == 0
    ):
        raise ValueError("input must be a nonempty regular file")
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("output already exists")
    if type(frequency_index) is not int or frequency_index < 0:
        raise ValueError("frequency index must be a nonnegative integer")

    n_reference, n_primitive, metadata = _scan_layout(input_path)
    frequencies = sorted({frequency for _, frequency in metadata})
    if frequency_index >= len(frequencies):
        raise ValueError("frequency index is outside the available grid")
    selected_frequency = frequencies[frequency_index]
    selected_rows = [
        row for row, (_, frequency) in enumerate(metadata)
        if frequency == selected_frequency
    ]
    if not selected_rows:
        raise RuntimeError("selected frequency has no response rows")
    selected_row_set = set(selected_rows)

    temporary = output_path.with_name("." + output_path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("temporary output already exists")
    section = None
    metadata_row = 0
    q_value = 0
    try:
        with input_path.open("r", encoding="ascii") as source, temporary.open(
            "x", encoding="ascii"
        ) as destination:
            for line in source:
                stripped = line.strip()
                if stripped.startswith("<") and stripped.endswith(">"):
                    destination.write(line)
                    if stripped.startswith("</"):
                        section = None
                    else:
                        section = stripped[1:-1]
                    continue
                if section == "STERNHEIMER_SIAB_HEADER" and stripped.startswith(
                    "n_reference "
                ):
                    destination.write(f"n_reference {len(selected_rows)}\n")
                elif section == "REFERENCE_METADATA" and stripped and not stripped.startswith(
                    "#"
                ):
                    if metadata_row in selected_row_set:
                        destination.write(line)
                    metadata_row += 1
                elif section == "OVERLAP_Q" and stripped and not stripped.startswith("#"):
                    row = q_value // n_primitive
                    if row in selected_row_set:
                        destination.write(line)
                    q_value += 1
                else:
                    destination.write(line)

        if metadata_row != n_reference:
            raise RuntimeError("metadata row count changed during extraction")
        if q_value != n_reference * n_primitive:
            raise RuntimeError("OVERLAP_Q value count is inconsistent with the header")
        temporary.replace(output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "format_version": 1,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "frequency_index": frequency_index,
        "selected_frequency_ha": selected_frequency,
        "selected_rows": selected_rows,
        "n_primitive": n_primitive,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frequency-index", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = extract_frequency_subset(
        args.input,
        args.output,
        frequency_index=args.frequency_index,
    )
    if args.report is not None:
        if args.report.exists() or args.report.is_symlink():
            raise ValueError("report already exists")
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
