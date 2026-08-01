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
from sternheimer_data import SternheimerData


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
    sections = read_sections(path, _REQUIRED_SECTIONS)
    header = parse_key_value_header(
        sections["STERNHEIMER_SIAB_HEADER"],
        _HEADER_KEYS,
        "STERNHEIMER_SIAB_HEADER",
    )

    format_version = parse_int(header["format_version"], "format_version")
    if format_version != 1:
        raise ValueError(f"unsupported format_version {format_version}")
    n_reference = parse_count(header["n_reference"], "n_reference")
    n_primitive = parse_count(header["n_primitive"], "n_primitive")
    n_blocks = parse_count(header["n_blocks"], "n_blocks")
    grid_volume_bohr3 = parse_float(
        header["grid_volume_bohr3"], "grid_volume_bohr3"
    )

    blocks = parse_blocks(sections["PRIMITIVE_BLOCKS"], n_blocks)
    metadata = _parse_metadata(
        sections["REFERENCE_METADATA"], n_reference
    )
    q = parse_complex_matrix(
        sections["OVERLAP_Q"],
        "OVERLAP_Q",
        n_reference,
        n_primitive,
    )
    overlap = parse_complex_matrix(
        sections["OVERLAP_S"],
        "OVERLAP_S",
        n_primitive,
        n_primitive,
    )
    provenance = parse_provenance(sections["PROVENANCE_JSON"])

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
            parse_int(
                fields[0], f"REFERENCE_METADATA[{row_index}].occupied_state"
            )
        )
        values["auxiliary_channel"].append(
            parse_int(
                fields[1],
                f"REFERENCE_METADATA[{row_index}].auxiliary_channel",
            )
        )
        for field, value in zip(
            ("frequency_ha", "occupation", "frequency_weight", "norm"),
            fields[2:],
        ):
            values[field].append(
                parse_float(
                    value, f"REFERENCE_METADATA[{row_index}].{field}"
                )
            )
    return values
