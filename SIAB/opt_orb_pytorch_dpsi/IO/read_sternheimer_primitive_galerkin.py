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
from sternheimer_fixed_ao_data import AuxiliaryChannel
from sternheimer_primitive_galerkin_data import (
    SternheimerPrimitiveGalerkinData,
)


_REQUIRED_SECTIONS = (
    "STERNHEIMER_GALERKIN_PRIMITIVE_HEADER",
    "PRIMITIVE_BLOCKS",
    "AUXILIARY_CHANNELS",
    "FIXED_AO_OCCUPATIONS",
    "OVERLAP_S",
    "HAMILTONIAN_H",
    "PERTURBATION_V",
    "PRIMITIVE_AO_OVERLAP",
    "FIXED_AO_GRID_OVERLAP",
    "FIXED_AO_GRID_HAMILTONIAN",
    "FREQUENCY_GRID",
    "PROVENANCE_JSON",
)
_HEADER_KEYS = (
    "format_version",
    "representation",
    "energy_unit",
    "n_primitive",
    "n_fixed_ao",
    "n_blocks",
    "n_spin",
    "n_auxiliary",
    "n_frequency",
)


def read_sternheimer_primitive_galerkin(path):
    sections = read_sections(path, _REQUIRED_SECTIONS)
    header = parse_key_value_header(
        sections["STERNHEIMER_GALERKIN_PRIMITIVE_HEADER"],
        _HEADER_KEYS,
        "STERNHEIMER_GALERKIN_PRIMITIVE_HEADER",
    )
    format_version = parse_int(header["format_version"], "format_version")
    if format_version != 1:
        raise ValueError(f"unsupported format_version {format_version}")
    representation = header["representation"]
    if representation != "bessel_primitive_uniform_grid_gamma":
        raise ValueError(f"unsupported representation {representation}")
    energy_unit = header["energy_unit"]
    if energy_unit != "Ha":
        raise ValueError(f"unsupported energy_unit {energy_unit}")

    n_primitive = _positive_count(header["n_primitive"], "n_primitive")
    n_fixed_ao = _positive_count(header["n_fixed_ao"], "n_fixed_ao")
    n_blocks = _positive_count(header["n_blocks"], "n_blocks")
    n_spin = _positive_count(header["n_spin"], "n_spin")
    n_auxiliary = _positive_count(header["n_auxiliary"], "n_auxiliary")
    n_frequency = _positive_count(header["n_frequency"], "n_frequency")

    blocks = parse_blocks(sections["PRIMITIVE_BLOCKS"], n_blocks)
    channels = _parse_channels(sections["AUXILIARY_CHANNELS"], n_auxiliary)
    occupation = _parse_occupations(
        sections["FIXED_AO_OCCUPATIONS"], n_spin, n_fixed_ao
    )
    overlap = parse_complex_matrix(
        sections["OVERLAP_S"], "OVERLAP_S", n_primitive, n_primitive
    )
    hamiltonian_ha = parse_complex_matrix(
        sections["HAMILTONIAN_H"],
        "HAMILTONIAN_H",
        n_spin * n_primitive,
        n_primitive,
    ).reshape(n_spin, n_primitive, n_primitive)
    perturbation_ha = parse_complex_matrix(
        sections["PERTURBATION_V"],
        "PERTURBATION_V",
        n_auxiliary * n_primitive,
        n_primitive,
    ).reshape(n_auxiliary, n_primitive, n_primitive)
    primitive_ao_overlap = parse_complex_matrix(
        sections["PRIMITIVE_AO_OVERLAP"],
        "PRIMITIVE_AO_OVERLAP",
        n_primitive,
        n_fixed_ao,
    )
    fixed_ao_grid_overlap = parse_complex_matrix(
        sections["FIXED_AO_GRID_OVERLAP"],
        "FIXED_AO_GRID_OVERLAP",
        n_fixed_ao,
        n_fixed_ao,
    )
    fixed_ao_grid_hamiltonian_ha = parse_complex_matrix(
        sections["FIXED_AO_GRID_HAMILTONIAN"],
        "FIXED_AO_GRID_HAMILTONIAN",
        n_spin * n_fixed_ao,
        n_fixed_ao,
    ).reshape(n_spin, n_fixed_ao, n_fixed_ao)
    frequency_ha, frequency_weight_ha = _parse_frequency_grid(
        sections["FREQUENCY_GRID"], n_frequency
    )

    return SternheimerPrimitiveGalerkinData(
        format_version=format_version,
        representation=representation,
        energy_unit=energy_unit,
        blocks=blocks,
        channels=channels,
        occupation=occupation,
        overlap=overlap,
        hamiltonian_ha=hamiltonian_ha,
        perturbation_ha=perturbation_ha,
        primitive_ao_overlap=primitive_ao_overlap,
        fixed_ao_grid_overlap=fixed_ao_grid_overlap,
        fixed_ao_grid_hamiltonian_ha=fixed_ao_grid_hamiltonian_ha,
        frequency_ha=frequency_ha,
        frequency_weight_ha=frequency_weight_ha,
        provenance=parse_provenance(sections["PROVENANCE_JSON"]),
    )


def _positive_count(value, field):
    result = parse_count(value, field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _parse_channels(lines, expected_count):
    if len(lines) != expected_count:
        raise ValueError(
            f"AUXILIARY_CHANNELS expected {expected_count} rows, found {len(lines)}"
        )
    channels = []
    for row_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"AUXILIARY_CHANNELS row {row_index} expected 6 fields")
        channel = AuxiliaryChannel(
            channel_index=parse_int(
                fields[0], f"AUXILIARY_CHANNELS[{row_index}].channel_index"
            ),
            atom_index=parse_int(
                fields[1], f"AUXILIARY_CHANNELS[{row_index}].atom_index"
            ),
            l=parse_int(fields[2], f"AUXILIARY_CHANNELS[{row_index}].l"),
            radial_index=parse_int(
                fields[3], f"AUXILIARY_CHANNELS[{row_index}].radial_index"
            ),
            m=parse_int(fields[4], f"AUXILIARY_CHANNELS[{row_index}].m"),
            label=fields[5],
        )
        if channel.channel_index != row_index:
            raise ValueError(
                f"AUXILIARY_CHANNELS row {row_index} expected channel_index "
                f"{row_index}, got {channel.channel_index}"
            )
        channels.append(channel)
    return tuple(channels)


def _parse_occupations(lines, n_spin, n_fixed_ao):
    expected_count = n_spin * n_fixed_ao
    if len(lines) != expected_count:
        raise ValueError(
            f"FIXED_AO_OCCUPATIONS expected {expected_count} rows, "
            f"found {len(lines)}"
        )
    occupations = []
    for row_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(
                f"FIXED_AO_OCCUPATIONS row {row_index} expected 3 fields"
            )
        spin_index = parse_int(
            fields[0], f"FIXED_AO_OCCUPATIONS[{row_index}].spin_index"
        )
        state_index = parse_int(
            fields[1], f"FIXED_AO_OCCUPATIONS[{row_index}].state_index"
        )
        expected_indices = (row_index // n_fixed_ao, row_index % n_fixed_ao)
        if (spin_index, state_index) != expected_indices:
            raise ValueError(
                f"FIXED_AO_OCCUPATIONS row {row_index} expected indices "
                f"{expected_indices}, got ({spin_index}, {state_index})"
            )
        occupations.append(
            parse_float(
                fields[2], f"FIXED_AO_OCCUPATIONS[{row_index}].occupation"
            )
        )
    return torch.tensor(
        occupations, dtype=torch.float64, device="cpu"
    ).reshape(n_spin, n_fixed_ao)


def _parse_frequency_grid(lines, expected_count):
    if len(lines) != expected_count:
        raise ValueError(
            f"FREQUENCY_GRID expected {expected_count} rows, found {len(lines)}"
        )
    frequency = []
    weight = []
    for row_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"FREQUENCY_GRID row {row_index} expected 3 fields")
        frequency_index = parse_int(
            fields[0], f"FREQUENCY_GRID[{row_index}].frequency_index"
        )
        if frequency_index != row_index:
            raise ValueError(
                f"FREQUENCY_GRID row {row_index} expected frequency_index "
                f"{row_index}, got {frequency_index}"
            )
        frequency.append(
            parse_float(fields[1], f"FREQUENCY_GRID[{row_index}].frequency_ha")
        )
        weight.append(
            parse_float(fields[2], f"FREQUENCY_GRID[{row_index}].weight_ha")
        )
    return (
        torch.tensor(frequency, dtype=torch.float64, device="cpu"),
        torch.tensor(weight, dtype=torch.float64, device="cpu"),
    )
