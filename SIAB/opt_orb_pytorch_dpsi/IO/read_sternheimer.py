import json
import math
from pathlib import Path

import torch

from sternheimer_data import PrimitiveBlock, SternheimerData


_REQUIRED_SECTIONS = (
    "STERNHEIMER_SIAB_HEADER",
    "PRIMITIVE_BLOCKS",
    "REFERENCE_METADATA",
    "OVERLAP_Q",
    "OVERLAP_S",
    "PROVENANCE_JSON",
)
_HEADER_KEYS = (
    "format_version",
    "n_reference",
    "n_primitive",
    "n_blocks",
    "grid_volume_bohr3",
)


def read_sternheimer(path) -> SternheimerData:
    sections = _read_sections(path)
    header = _parse_header(sections["STERNHEIMER_SIAB_HEADER"])

    format_version = _parse_int(header["format_version"], "format_version")
    if format_version != 1:
        raise ValueError(f"unsupported format_version {format_version}")
    n_reference = _parse_count(header["n_reference"], "n_reference")
    n_primitive = _parse_count(header["n_primitive"], "n_primitive")
    n_blocks = _parse_count(header["n_blocks"], "n_blocks")
    grid_volume_bohr3 = _parse_float(
        header["grid_volume_bohr3"], "grid_volume_bohr3"
    )

    blocks = _parse_blocks(sections["PRIMITIVE_BLOCKS"], n_blocks)
    metadata = _parse_metadata(
        sections["REFERENCE_METADATA"], n_reference
    )
    q = _parse_complex_matrix(
        sections["OVERLAP_Q"],
        "OVERLAP_Q",
        n_reference,
        n_primitive,
    )
    overlap = _parse_complex_matrix(
        sections["OVERLAP_S"],
        "OVERLAP_S",
        n_primitive,
        n_primitive,
    )
    provenance = _parse_provenance(sections["PROVENANCE_JSON"])

    return SternheimerData(
        format_version=format_version,
        grid_volume_bohr3=grid_volume_bohr3,
        blocks=blocks,
        occupied_state=torch.tensor(
            metadata["occupied_state"], dtype=torch.int64, device="cpu"
        ),
        auxiliary_channel=torch.tensor(
            metadata["auxiliary_channel"], dtype=torch.int64, device="cpu"
        ),
        frequency_ha=torch.tensor(
            metadata["frequency_ha"], dtype=torch.float64, device="cpu"
        ),
        occupation=torch.tensor(
            metadata["occupation"], dtype=torch.float64, device="cpu"
        ),
        frequency_weight=torch.tensor(
            metadata["frequency_weight"], dtype=torch.float64, device="cpu"
        ),
        norm=torch.tensor(
            metadata["norm"], dtype=torch.float64, device="cpu"
        ),
        q=q,
        overlap=overlap,
        provenance=provenance,
    )


def _read_sections(path):
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
                if name not in _REQUIRED_SECTIONS:
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
    for name in _REQUIRED_SECTIONS:
        if name not in sections:
            raise ValueError(f"missing required section: {name}")
    return sections


def _parse_header(lines):
    values = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"STERNHEIMER_SIAB_HEADER invalid row: {line}")
        key, value = fields
        if key not in _HEADER_KEYS:
            raise ValueError(f"unknown header key: {key}")
        if key in values:
            raise ValueError(f"repeated header key: {key}")
        values[key] = value
    for key in _HEADER_KEYS:
        if key not in values:
            raise ValueError(f"missing header key: {key}")
    return values


def _parse_blocks(lines, expected_count):
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
        element = fields[0]
        blocks.append(
            PrimitiveBlock(
                element=element,
                atom_index=_parse_int(
                    fields[1], f"PRIMITIVE_BLOCKS[{row_index}].atom_index"
                ),
                l=_parse_int(fields[2], f"PRIMITIVE_BLOCKS[{row_index}].l"),
                m=_parse_int(fields[3], f"PRIMITIVE_BLOCKS[{row_index}].m"),
                n_primitive=_parse_int(
                    fields[4], f"PRIMITIVE_BLOCKS[{row_index}].n_primitive"
                ),
                offset=_parse_int(
                    fields[5], f"PRIMITIVE_BLOCKS[{row_index}].offset"
                ),
            )
        )
    return tuple(blocks)


def _parse_metadata(lines, expected_count):
    if len(lines) != expected_count:
        raise ValueError(
            f"REFERENCE_METADATA expected {expected_count} rows, "
            f"found {len(lines)}"
        )
    values = {
        "occupied_state": [],
        "auxiliary_channel": [],
        "frequency_ha": [],
        "occupation": [],
        "frequency_weight": [],
        "norm": [],
    }
    for row_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(
                f"REFERENCE_METADATA row {row_index} expected 6 fields"
            )
        values["occupied_state"].append(
            _parse_int(
                fields[0], f"REFERENCE_METADATA[{row_index}].occupied_state"
            )
        )
        values["auxiliary_channel"].append(
            _parse_int(
                fields[1],
                f"REFERENCE_METADATA[{row_index}].auxiliary_channel",
            )
        )
        for field, value in zip(
            ("frequency_ha", "occupation", "frequency_weight", "norm"),
            fields[2:],
        ):
            values[field].append(
                _parse_float(
                    value, f"REFERENCE_METADATA[{row_index}].{field}"
                )
            )
    return values


def _parse_complex_matrix(lines, section, rows, columns):
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
        real = _parse_float(fields[0], f"{section}[{value_index}].real")
        imag = _parse_float(fields[1], f"{section}[{value_index}].imag")
        values.append(complex(real, imag))
    return torch.tensor(
        values, dtype=torch.complex128, device="cpu"
    ).reshape(rows, columns)


def _parse_provenance(lines):
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
    return _parse_float(value, "PROVENANCE_JSON numeric value")


def _parse_count(value, field):
    parsed = _parse_int(value, field)
    if parsed < 0:
        raise ValueError(f"{field} must be nonnegative")
    return parsed


def _parse_int(value, field):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _parse_float(value, field):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a floating-point value") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed
