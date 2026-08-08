import json
import math
from pathlib import Path

import torch

from sternheimer_data import PrimitiveBlock


def read_sections(path, required_sections, optional_sections=()):
    required_sections = tuple(required_sections)
    optional_sections = tuple(optional_sections)
    accepted_sections = required_sections + optional_sections
    if len(set(accepted_sections)) != len(accepted_sections):
        raise ValueError("required and optional section names must be unique")
    sections = {}
    opened_sections = set()
    current_name = None
    current_lines = []

    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            stripped_line = raw_line.strip()
            if (
                current_name == "PROVENANCE_JSON"
                and stripped_line
                and not stripped_line.startswith("#")
                and not stripped_line.startswith("</")
            ):
                line = stripped_line
            else:
                line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            if line.startswith("<") and line.endswith(">"):
                tag = line[1:-1]
                closing = tag.startswith("/")
                name = tag[1:] if closing else tag
                if not name or "<" in name or ">" in name:
                    raise ValueError(
                        f"malformed section tag on line {line_number}: {line}"
                    )
                if name not in accepted_sections:
                    raise ValueError(f"unknown section tag: {name}")

                if closing:
                    if current_name is None:
                        raise ValueError(
                            f"unexpected closing tag </{name}> on line {line_number}"
                        )
                    if name != current_name:
                        raise ValueError(
                            "mismatched closing tag: expected "
                            f"</{current_name}>, got </{name}>"
                        )
                    sections[current_name] = current_lines
                    current_name = None
                    current_lines = []
                else:
                    if name in opened_sections:
                        raise ValueError(f"repeated section: {name}")
                    if current_name is not None:
                        raise ValueError(
                            f"opening tag <{name}> on line {line_number} "
                            f"appears before </{current_name}>"
                        )
                    opened_sections.add(name)
                    current_name = name
                continue

            if line.startswith("<") or line.endswith(">"):
                raise ValueError(
                    f"malformed section tag on line {line_number}: {line}"
                )
            if current_name is None:
                raise ValueError(
                    f"text outside tagged sections on line {line_number}: {line}"
                )
            current_lines.append(line)

    if current_name is not None:
        raise ValueError(f"missing closing tag for section: {current_name}")
    for name in required_sections:
        if name not in sections:
            raise ValueError(f"missing required section: {name}")
    return sections


def parse_key_value_header(lines, allowed_keys, section):
    allowed_keys = tuple(allowed_keys)
    values = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{section} invalid row: {line}")
        key, value = fields
        if key not in allowed_keys:
            raise ValueError(f"unknown header key: {key}")
        if key in values:
            raise ValueError(f"repeated header key: {key}")
        values[key] = value
    for key in allowed_keys:
        if key not in values:
            raise ValueError(f"missing header key: {key}")
    return values


def parse_blocks(lines, expected_count):
    if len(lines) != expected_count:
        raise ValueError(
            f"PRIMITIVE_BLOCKS expected {expected_count} rows, "
            f"found {len(lines)}"
        )
    blocks = []
    for row_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(
                f"PRIMITIVE_BLOCKS row {row_index} expected 6 fields"
            )
        blocks.append(
            PrimitiveBlock(
                element=fields[0],
                atom_index=parse_int(
                    fields[1], f"PRIMITIVE_BLOCKS[{row_index}].atom_index"
                ),
                l=parse_int(fields[2], f"PRIMITIVE_BLOCKS[{row_index}].l"),
                m=parse_int(fields[3], f"PRIMITIVE_BLOCKS[{row_index}].m"),
                n_primitive=parse_int(
                    fields[4], f"PRIMITIVE_BLOCKS[{row_index}].n_primitive"
                ),
                offset=parse_int(
                    fields[5], f"PRIMITIVE_BLOCKS[{row_index}].offset"
                ),
            )
        )
    return tuple(blocks)


def parse_complex_matrix(lines, section, rows, columns):
    expected_count = rows * columns
    if len(lines) != expected_count:
        raise ValueError(
            f"{section} expected {expected_count} complex values, "
            f"found {len(lines)}"
        )
    values = []
    for value_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(
                f"{section} complex value {value_index} expected real and imaginary parts"
            )
        real = parse_float(fields[0], f"{section}[{value_index}].real")
        imag = parse_float(fields[1], f"{section}[{value_index}].imag")
        values.append(complex(real, imag))
    return torch.tensor(
        values, dtype=torch.complex128, device="cpu"
    ).reshape(rows, columns)


def parse_provenance(lines):
    if len(lines) != 1:
        raise ValueError("PROVENANCE_JSON must contain exactly one JSON object line")
    try:
        provenance = json.loads(
            lines[0],
            object_pairs_hook=_reject_duplicate_provenance_keys,
            parse_constant=_reject_provenance_constant,
            parse_float=_parse_provenance_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"PROVENANCE_JSON invalid JSON: {exc.msg}") from exc
    if not isinstance(provenance, dict):
        raise ValueError("PROVENANCE_JSON must contain a JSON object")
    return provenance


def _reject_duplicate_provenance_keys(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"PROVENANCE_JSON duplicate key: {key}")
        values[key] = value
    return values


def _reject_provenance_constant(value):
    raise ValueError(f"PROVENANCE_JSON invalid constant: {value}")


def _parse_provenance_float(value):
    return parse_float(value, "PROVENANCE_JSON numeric value")


def parse_count(value, field):
    parsed = parse_int(value, field)
    if parsed < 0:
        raise ValueError(f"{field} must be nonnegative")
    return parsed


def parse_int(value, field):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def parse_float(value, field):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a floating-point value") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed
