import torch

from IO.sternheimer_text import (
    parse_blocks,
    parse_complex_matrix,
    parse_count,
    parse_float,
    parse_int,
    parse_key_value_header,
    parse_provenance,
    read_sections,
)
from sternheimer_source_data import SternheimerSourceData


_REQUIRED_SECTIONS = (
    "STERNHEIMER_SIAB_SOURCE_HEADER",
    "PRIMITIVE_BLOCKS",
    "SOURCE_METADATA",
    "OVERLAP_D",
    "OVERLAP_S",
    "PROVENANCE_JSON",
)
_HEADER_KEYS = (
    "format_version",
    "n_source",
    "n_primitive",
    "n_blocks",
    "grid_volume_bohr3",
)


def read_sternheimer_source(path) -> SternheimerSourceData:
    sections = read_sections(path, _REQUIRED_SECTIONS)
    header = parse_key_value_header(
        sections["STERNHEIMER_SIAB_SOURCE_HEADER"],
        _HEADER_KEYS,
        "STERNHEIMER_SIAB_SOURCE_HEADER",
    )

    format_version = parse_int(header["format_version"], "format_version")
    if format_version != 1:
        raise ValueError(f"unsupported format_version {format_version}")
    n_source = parse_count(header["n_source"], "n_source")
    n_primitive = parse_count(header["n_primitive"], "n_primitive")
    n_blocks = parse_count(header["n_blocks"], "n_blocks")
    grid_volume_bohr3 = parse_float(
        header["grid_volume_bohr3"], "grid_volume_bohr3"
    )

    blocks = parse_blocks(sections["PRIMITIVE_BLOCKS"], n_blocks)
    metadata = _parse_source_metadata(
        sections["SOURCE_METADATA"], n_source
    )
    d = parse_complex_matrix(
        sections["OVERLAP_D"],
        "OVERLAP_D",
        n_source,
        n_primitive,
    )
    overlap = parse_complex_matrix(
        sections["OVERLAP_S"],
        "OVERLAP_S",
        n_primitive,
        n_primitive,
    )
    provenance = parse_provenance(sections["PROVENANCE_JSON"])

    return SternheimerSourceData(
        format_version=format_version,
        grid_volume_bohr3=grid_volume_bohr3,
        blocks=blocks,
        occupied_state=torch.tensor(
            metadata["occupied_state"], dtype=torch.int64, device="cpu"
        ),
        auxiliary_channel=torch.tensor(
            metadata["auxiliary_channel"], dtype=torch.int64, device="cpu"
        ),
        occupation=torch.tensor(
            metadata["occupation"], dtype=torch.float64, device="cpu"
        ),
        norm=torch.tensor(
            metadata["norm"], dtype=torch.float64, device="cpu"
        ),
        d=d,
        overlap=overlap,
        provenance=provenance,
    )


def _parse_source_metadata(lines, expected_count):
    if len(lines) != expected_count:
        raise ValueError(
            f"SOURCE_METADATA expected {expected_count} rows, "
            f"found {len(lines)}"
        )
    values = {
        "occupied_state": [],
        "auxiliary_channel": [],
        "occupation": [],
        "norm": [],
    }
    for row_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"SOURCE_METADATA row {row_index} expected 4 fields"
            )
        values["occupied_state"].append(
            parse_int(
                fields[0], f"SOURCE_METADATA[{row_index}].occupied_state"
            )
        )
        values["auxiliary_channel"].append(
            parse_int(
                fields[1], f"SOURCE_METADATA[{row_index}].auxiliary_channel"
            )
        )
        values["occupation"].append(
            parse_float(
                fields[2], f"SOURCE_METADATA[{row_index}].occupation"
            )
        )
        values["norm"].append(
            parse_float(fields[3], f"SOURCE_METADATA[{row_index}].norm")
        )
    return values
