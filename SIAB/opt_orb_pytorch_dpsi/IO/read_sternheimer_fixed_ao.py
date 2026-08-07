import torch

from IO.sternheimer_text import (
    parse_complex_matrix,
    parse_count,
    parse_float,
    parse_int,
    parse_key_value_header,
    parse_provenance,
    read_sections,
)
from sternheimer_fixed_ao_data import AuxiliaryChannel, SternheimerFixedAOData


_REQUIRED_SECTIONS = (
    "STERNHEIMER_GALERKIN_HEADER",
    "AUXILIARY_CHANNELS",
    "SPIN_METADATA",
    "OVERLAP_S",
    "HAMILTONIAN_H",
    "PERTURBATION_V",
    "FREQUENCY_GRID",
    "PROVENANCE_JSON",
)
_HEADER_KEYS = (
    "format_version",
    "representation",
    "energy_unit",
    "n_basis",
    "n_spin",
    "n_auxiliary",
    "n_frequency",
)


def read_sternheimer_fixed_ao(path) -> SternheimerFixedAOData:
    sections = read_sections(path, _REQUIRED_SECTIONS)
    header = parse_key_value_header(
        sections["STERNHEIMER_GALERKIN_HEADER"],
        _HEADER_KEYS,
        "STERNHEIMER_GALERKIN_HEADER",
    )
    format_version = parse_int(header["format_version"], "format_version")
    if format_version != 1:
        raise ValueError(f"unsupported format_version {format_version}")
    representation = header["representation"]
    if representation != "fixed_lcao_gamma":
        raise ValueError(f"unsupported representation {representation}")
    energy_unit = header["energy_unit"]
    if energy_unit != "Ha":
        raise ValueError(f"unsupported energy_unit {energy_unit}")
    n_basis = _positive_count(header["n_basis"], "n_basis")
    n_spin = _positive_count(header["n_spin"], "n_spin")
    n_auxiliary = _positive_count(header["n_auxiliary"], "n_auxiliary")
    n_frequency = _positive_count(header["n_frequency"], "n_frequency")

    channels = _parse_channels(sections["AUXILIARY_CHANNELS"], n_auxiliary)
    eigenvalue_ha, occupation = _parse_spin_metadata(
        sections["SPIN_METADATA"], n_spin, n_basis
    )
    overlap = parse_complex_matrix(
        sections["OVERLAP_S"], "OVERLAP_S", n_basis, n_basis
    )
    hamiltonian_ha = parse_complex_matrix(
        sections["HAMILTONIAN_H"],
        "HAMILTONIAN_H",
        n_spin * n_basis,
        n_basis,
    ).reshape(n_spin, n_basis, n_basis)
    perturbation_ha = parse_complex_matrix(
        sections["PERTURBATION_V"],
        "PERTURBATION_V",
        n_auxiliary * n_basis,
        n_basis,
    ).reshape(n_auxiliary, n_basis, n_basis)
    frequency_ha, frequency_weight_ha = _parse_frequency_grid(
        sections["FREQUENCY_GRID"], n_frequency
    )
    provenance = parse_provenance(sections["PROVENANCE_JSON"])

    return SternheimerFixedAOData(
        format_version=format_version,
        representation=representation,
        energy_unit=energy_unit,
        channels=channels,
        eigenvalue_ha=eigenvalue_ha,
        occupation=occupation,
        overlap=overlap,
        hamiltonian_ha=hamiltonian_ha,
        perturbation_ha=perturbation_ha,
        frequency_ha=frequency_ha,
        frequency_weight_ha=frequency_weight_ha,
        provenance=provenance,
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


def _parse_spin_metadata(lines, n_spin, n_basis):
    expected_count = n_spin * n_basis
    if len(lines) != expected_count:
        raise ValueError(
            f"SPIN_METADATA expected {expected_count} rows, found {len(lines)}"
        )
    eigenvalue = []
    occupation = []
    for row_index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"SPIN_METADATA row {row_index} expected 4 fields")
        spin_index = parse_int(fields[0], f"SPIN_METADATA[{row_index}].spin_index")
        state_index = parse_int(fields[1], f"SPIN_METADATA[{row_index}].state_index")
        expected_spin = row_index // n_basis
        expected_state = row_index % n_basis
        if (spin_index, state_index) != (expected_spin, expected_state):
            raise ValueError(
                f"SPIN_METADATA row {row_index} expected indices "
                f"({expected_spin}, {expected_state}), got ({spin_index}, {state_index})"
            )
        eigenvalue.append(
            parse_float(fields[2], f"SPIN_METADATA[{row_index}].eigenvalue_ha")
        )
        occupation.append(
            parse_float(fields[3], f"SPIN_METADATA[{row_index}].occupation")
        )
    return (
        torch.tensor(eigenvalue, dtype=torch.float64, device="cpu").reshape(
            n_spin, n_basis
        ),
        torch.tensor(occupation, dtype=torch.float64, device="cpu").reshape(
            n_spin, n_basis
        ),
    )


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
