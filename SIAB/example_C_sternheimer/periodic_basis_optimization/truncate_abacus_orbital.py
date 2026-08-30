#!/usr/bin/env python3
"""Remove complete high-angular-momentum blocks from an ABACUS orbital file."""

import argparse
import json
import re
from pathlib import Path


LMAX_RE = re.compile(r"^(Lmax\s+)(\d+)(\s*)$")
NU_RE = re.compile(r"^Number of ([A-Z])orbital-->\s+(\d+)\s*$")
MESH_RE = re.compile(r"^Mesh\s+(\d+)\s*$")
BLOCK_HEADER = "                Type                   L                   N"
BLOCK_INDEX_RE = re.compile(r"^\s*0\s+(\d+)\s+(\d+)\s*$")


def _parse_header(lines):
    source_lmax = None
    mesh = None
    nu_entries = []
    for index, line in enumerate(lines):
        match = LMAX_RE.match(line)
        if match:
            source_lmax = int(match.group(2))
        match = NU_RE.match(line)
        if match:
            nu_entries.append((index, match.group(1), int(match.group(2))))
        match = MESH_RE.match(line)
        if match:
            mesh = int(match.group(1))
    if source_lmax is None:
        raise ValueError("missing Lmax in orbital header")
    if mesh is None or mesh <= 0:
        raise ValueError("missing or invalid Mesh in orbital header")
    if len(nu_entries) != source_lmax + 1:
        raise ValueError("orbital channel counts do not match Lmax")
    return source_lmax, mesh, nu_entries


def _parse_radial_blocks(lines, mesh):
    starts = [index for index, line in enumerate(lines) if line == BLOCK_HEADER]
    blocks = []
    for ordinal, start in enumerate(starts):
        stop = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(lines)
        if start + 1 >= stop:
            raise ValueError("radial block is missing its L/N index")
        match = BLOCK_INDEX_RE.match(lines[start + 1])
        if not match:
            raise ValueError("radial block has an invalid L/N index")
        l_value = int(match.group(1))
        n_value = int(match.group(2))
        value_count = 0
        for line in lines[start + 2:stop]:
            if line.strip():
                try:
                    values = [float(token) for token in line.split()]
                except ValueError as error:
                    raise ValueError("radial block contains a non-numeric value") from error
                value_count += len(values)
        if value_count != mesh:
            raise ValueError(
                "radial block L={} N={} has {} values; expected {}".format(
                    l_value, n_value, value_count, mesh
                )
            )
        blocks.append((start, stop, l_value, n_value))
    return blocks


def truncate_abacus_orbital(source, output, *, target_lmax):
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("orbital output parent directory does not exist")

    lines = source.read_text(encoding="ascii").splitlines()
    source_lmax, mesh, nu_entries = _parse_header(lines)
    if target_lmax < 0 or target_lmax >= source_lmax:
        raise ValueError("target_lmax must be non-negative and smaller than source Lmax")

    source_nu = [entry[2] for entry in nu_entries]
    blocks = _parse_radial_blocks(lines, mesh)
    expected_blocks = sum(source_nu)
    if len(blocks) != expected_blocks:
        raise ValueError(
            "radial block count {} does not match header count {}".format(
                len(blocks), expected_blocks
            )
        )
    observed = {}
    for _start, _stop, l_value, n_value in blocks:
        observed.setdefault(l_value, []).append(n_value)
    for l_value, count in enumerate(source_nu):
        if sorted(observed.get(l_value, [])) != list(range(count)):
            raise ValueError("radial block indices do not match channel counts")

    first_block = blocks[0][0] if blocks else len(lines)
    header = []
    nu_line_indices = {entry[0]: l_value for l_value, entry in enumerate(nu_entries)}
    for index, line in enumerate(lines[:first_block]):
        match = LMAX_RE.match(line)
        if match:
            header.append("{}{}{}".format(match.group(1), target_lmax, match.group(3)))
            continue
        if index in nu_line_indices and nu_line_indices[index] > target_lmax:
            continue
        header.append(line)

    output_lines = list(header)
    for start, stop, l_value, _n_value in blocks:
        if l_value <= target_lmax:
            output_lines.extend(lines[start:stop])
    output.write_text("\n".join(output_lines).rstrip() + "\n", encoding="ascii")

    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_lmax": source_lmax,
        "target_lmax": target_lmax,
        "mesh": mesh,
        "source_nu": source_nu,
        "output_nu": source_nu[:target_lmax + 1],
        "source_nao": sum((2 * l_value + 1) * count for l_value, count in enumerate(source_nu)),
        "output_nao": sum(
            (2 * l_value + 1) * count
            for l_value, count in enumerate(source_nu[:target_lmax + 1])
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-lmax", type=int, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = truncate_abacus_orbital(
        args.input,
        args.output,
        target_lmax=args.target_lmax,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.write_text(encoded, encoding="ascii")
    print(encoded, end="")


if __name__ == "__main__":
    main()
