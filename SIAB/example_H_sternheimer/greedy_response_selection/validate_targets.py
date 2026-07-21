#!/usr/bin/env python3
"""Validate immutable Sternheimer response targets before SIAB selection."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from IO.read_sternheimer import read_sternheimer


DEFAULT_AUXILIARY_RADIAL_COUNTS = (8, 7, 6, 4, 4, 3, 2, 1, 1)


@dataclass(frozen=True)
class ABFSChannel:
    channel: int
    atom: int
    atom_local: int
    type_index: int
    l: int
    radial: int
    magnetic: int
    label: str
    max_abs: float


@dataclass(frozen=True)
class TargetValidationSummary:
    atoms: int
    primitive_columns: int
    auxiliary_channels: int
    response_channels: int
    occupied_states: tuple
    frequencies: tuple
    references: int
    abfs_source: str
    auxiliary_basis_sha256: str
    primitive_representation: str


def _integer(value, field):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if str(result) != str(value):
        raise ValueError(f"{field} must be an integer")
    return result


def _finite_float(value, field):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def parse_abfs_channels(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 9:
                raise ValueError(
                    f"ABFS channel line {line_number} must contain 9 fields"
                )
            rows.append(
                ABFSChannel(
                    channel=_integer(fields[0], "channel"),
                    atom=_integer(fields[1], "atom"),
                    atom_local=_integer(fields[2], "atom_local"),
                    type_index=_integer(fields[3], "type"),
                    l=_integer(fields[4], "l"),
                    radial=_integer(fields[5], "radial"),
                    magnetic=_integer(fields[6], "magnetic"),
                    label=fields[7],
                    max_abs=_finite_float(fields[8], "max_abs"),
                )
            )
    if not rows:
        raise ValueError("ABFS channel diagnostic is empty")
    return tuple(rows)


def parse_sternheimer_status(path):
    values = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                continue
            key, value = fields
            if key in values:
                raise ValueError(
                    f"Sternheimer status repeats {key!r} on line {line_number}"
                )
            values[key] = value
    if not values:
        raise ValueError("Sternheimer status is empty")
    return values


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _sha256(value, field):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{field} must be a 64-digit SHA256")
    return value.lower()


def _validate_primitives(data, expected_atoms, primitive_lmax, radial_primitives):
    atom_indices = sorted({block.atom_index for block in data.blocks})
    if atom_indices != list(range(expected_atoms)):
        raise ValueError(
            f"primitive atom indices expected {list(range(expected_atoms))}, "
            f"got {atom_indices}"
        )

    groups = {}
    for block in data.blocks:
        if block.l > primitive_lmax:
            raise ValueError(f"primitive l={block.l} exceeds lmax={primitive_lmax}")
        if block.n_primitive != radial_primitives:
            raise ValueError(
                f"primitive block {block.key!r} has {block.n_primitive} radial "
                f"columns, expected {radial_primitives}"
            )
        groups.setdefault((block.atom_index, block.l), []).append(block.m)

    for atom in range(expected_atoms):
        for l in range(primitive_lmax + 1):
            actual = tuple(sorted(groups.get((atom, l), ())))
            expected = tuple(range(-l, l + 1))
            if actual != expected:
                raise ValueError(
                    "incomplete primitive m group for "
                    f"atom={atom}, l={l}: expected {expected}, got {actual}"
                )

    expected_columns = expected_atoms * radial_primitives * sum(
        2 * l + 1 for l in range(primitive_lmax + 1)
    )
    if data.q.shape[1] != expected_columns:
        raise ValueError(
            f"primitive column count expected {expected_columns}, "
            f"got {data.q.shape[1]}"
        )
    return expected_columns


def _validate_auxiliary_channels(channels, expected_atoms, radial_counts):
    expected_per_atom = sum(
        radial_count * (2 * l + 1)
        for l, radial_count in enumerate(radial_counts)
    )
    expected_total = expected_atoms * expected_per_atom
    if len(channels) != expected_total:
        raise ValueError(
            f"auxiliary channel count expected {expected_total}, got {len(channels)}"
        )
    if [row.channel for row in channels] != list(range(expected_total)):
        raise ValueError("auxiliary channel indices are not contiguous")

    by_atom = {}
    keys = set()
    for row in channels:
        if row.atom < 0 or row.atom >= expected_atoms:
            raise ValueError(f"auxiliary channel has invalid atom {row.atom}")
        if row.l < 0 or row.l >= len(radial_counts):
            raise ValueError(f"auxiliary channel has invalid l={row.l}")
        if row.radial < 0 or row.radial >= radial_counts[row.l]:
            raise ValueError(
                f"auxiliary channel has invalid radial={row.radial} for l={row.l}"
            )
        if row.magnetic < 0 or row.magnetic >= 2 * row.l + 1:
            raise ValueError(
                f"auxiliary channel has invalid magnetic={row.magnetic} for l={row.l}"
            )
        if row.max_abs <= 0.0:
            raise ValueError("every auxiliary Hartree potential must be nonzero")
        key = (row.atom, row.l, row.radial, row.magnetic)
        if key in keys:
            raise ValueError(f"duplicate auxiliary channel {key!r}")
        keys.add(key)
        by_atom.setdefault(row.atom, []).append(row.atom_local)

    for atom in range(expected_atoms):
        if sorted(by_atom.get(atom, ())) != list(range(expected_per_atom)):
            raise ValueError(
                f"auxiliary atom-local indices are incomplete for atom {atom}"
            )
    return expected_total


def _validate_reference_product(data, naux, expected_nfreq):
    frequencies = tuple(sorted(set(data.frequency_ha.tolist())))
    if len(frequencies) != expected_nfreq:
        raise ValueError(
            f"frequency count expected {expected_nfreq}, got {len(frequencies)}"
        )
    occupied = tuple(sorted(set(data.occupied_state.tolist())))
    auxiliary = tuple(sorted(set(data.auxiliary_channel.tolist())))
    if auxiliary != tuple(range(naux)):
        raise ValueError(
            f"reference auxiliary channels expected 0..{naux - 1}, got {auxiliary}"
        )

    keys = list(
        zip(
            data.occupied_state.tolist(),
            data.auxiliary_channel.tolist(),
            data.frequency_ha.tolist(),
        )
    )
    expected_count = len(occupied) * naux * expected_nfreq
    if len(keys) != expected_count or len(set(keys)) != expected_count:
        raise ValueError(
            "reference Cartesian product over occupied state, auxiliary "
            "channel, and frequency is incomplete or duplicated"
        )
    return occupied, frequencies


def _validate_coulomb_whitening(data, status, raw_dimension):
    provenance = data.provenance
    if provenance.get("auxiliary_whitening") != "global_full_coulomb_v1":
        raise ValueError("target must declare global_full_coulomb_v1 whitening")
    raw = _integer(provenance.get("raw_auxiliary_dimension"), "raw auxiliary dimension")
    retained = _integer(
        provenance.get("whitened_auxiliary_rank"), "whitened auxiliary rank"
    )
    discarded = _integer(
        provenance.get("discarded_auxiliary_rank"), "discarded auxiliary rank"
    )
    if raw != raw_dimension or retained <= 0 or discarded < 0 or retained + discarded != raw:
        raise ValueError("Coulomb-whitening dimensions are inconsistent")

    threshold = _finite_float(
        provenance.get("coulomb_relative_threshold"), "Coulomb relative threshold"
    )
    max_error = _finite_float(
        provenance.get("coulomb_max_orthonormality_error"),
        "Coulomb orthonormality error",
    )
    if not 0.0 < threshold < 1.0 or not 0.0 <= max_error <= 1.0e-8:
        raise ValueError("Coulomb-whitening threshold or orthonormality error is invalid")
    eigenvalues = tuple(
        _finite_float(value, "Coulomb eigenvalue")
        for value in provenance.get("coulomb_eigenvalues", ())
    )
    if len(eigenvalues) != raw or eigenvalues != tuple(sorted(eigenvalues)):
        raise ValueError("complete sorted Coulomb eigenvalues are required")
    if eigenvalues[-1] <= 0.0:
        raise ValueError("Coulomb spectrum has no positive direction")
    cutoff = threshold * eigenvalues[-1]
    if sum(value > cutoff for value in eigenvalues) != retained:
        raise ValueError("Coulomb spectrum does not reproduce retained rank")

    transform_hash = provenance.get("coulomb_transform_sha256", "")
    if (
        not isinstance(transform_hash, str)
        or len(transform_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in transform_hash)
    ):
        raise ValueError("Coulomb transform SHA256 is invalid")
    expected_status = {
        "raw_auxiliary_dimension": str(raw),
        "whitened_auxiliary_rank": str(retained),
        "discarded_auxiliary_rank": str(discarded),
        "coulomb_transform_sha256": transform_hash,
        "response_channels": str(retained),
    }
    for key, expected in expected_status.items():
        if status.get(key) != expected:
            raise ValueError(f"status {key} does not match whitening provenance")
    if status.get("format") != "siab_v1" or status.get("target_file") != "sternheimer_matrix.dat":
        raise ValueError("status does not describe an SIAB-only target")
    return retained


def _validate_siab_contract(
    data, status, primitive_columns, expected_auxiliary_sha256
):
    target_hash = _sha256(
        data.provenance.get("auxiliary_basis_sha256"),
        "target auxiliary basis SHA256",
    )
    status_hash = _sha256(
        status.get("auxiliary_basis_sha256"),
        "status auxiliary basis SHA256",
    )
    if status_hash != target_hash:
        raise ValueError("status auxiliary basis SHA256 does not match target")
    if expected_auxiliary_sha256 is not None:
        expected_hash = _sha256(
            expected_auxiliary_sha256, "expected auxiliary basis SHA256"
        )
        if target_hash != expected_hash:
            raise ValueError("target auxiliary basis SHA256 does not match expected")

    representation = status.get("primitive_representation")
    if representation != "serial_reciprocal_pw_v1":
        raise ValueError("target does not use the required reciprocal PW primitive representation")
    if _integer(status.get("primitive_count"), "status primitive_count") != primitive_columns:
        raise ValueError("status primitive count does not match target columns")
    if _integer(
        status.get("primitive_reciprocal_count"),
        "status primitive_reciprocal_count",
    ) <= 0:
        raise ValueError("status primitive reciprocal count must be positive")

    estimated_bytes = _integer(
        status.get("estimated_dense_memory_bytes"),
        "status estimated_dense_memory_bytes",
    )
    slurm_bytes = _integer(
        status.get("slurm_memory_per_node_bytes"),
        "status slurm_memory_per_node_bytes",
    )
    if estimated_bytes <= 0 or slurm_bytes <= estimated_bytes:
        raise ValueError("status memory estimate is missing or exceeds the Slurm node limit")
    if status.get("memory_diagnostic") != "STERNHEIMER_SIAB_MEMORY.dat":
        raise ValueError("status does not identify the SIAB memory diagnostic")
    return target_hash, representation


def validate_response_target(
    data,
    channels,
    status,
    *,
    expected_atoms,
    primitive_lmax=4,
    radial_primitives=25,
    expected_nfreq=16,
    auxiliary_radial_counts=DEFAULT_AUXILIARY_RADIAL_COUNTS,
    expected_auxiliary_sha256=None,
):
    expected_atoms = _positive_int(expected_atoms, "expected_atoms")
    radial_primitives = _positive_int(radial_primitives, "radial_primitives")
    expected_nfreq = _positive_int(expected_nfreq, "expected_nfreq")
    if type(primitive_lmax) is not int or primitive_lmax < 0:
        raise ValueError("primitive_lmax must be a nonnegative integer")
    radial_counts = tuple(auxiliary_radial_counts)
    if not radial_counts or any(type(value) is not int or value <= 0 for value in radial_counts):
        raise ValueError("auxiliary_radial_counts must contain positive integers")

    if status.get("status") not in {"success", "completed"}:
        raise ValueError("Sternheimer status is not successful")
    if status.get("abfs_source") != "explicit_abfs":
        raise ValueError("Sternheimer target must use abfs_source explicit_abfs")
    if data.provenance.get("kernel") != "full_coulomb":
        raise ValueError("Sternheimer target must declare kernel full_coulomb")

    primitive_columns = _validate_primitives(
        data, expected_atoms, primitive_lmax, radial_primitives
    )
    auxiliary_channels = _validate_auxiliary_channels(
        channels, expected_atoms, radial_counts
    )
    if _integer(status.get("abfs_channels"), "status abfs_channels") != auxiliary_channels:
        raise ValueError("status ABFS channel count does not match diagnostic")
    response_channels = _validate_coulomb_whitening(
        data, status, auxiliary_channels
    )
    auxiliary_basis_sha256, primitive_representation = _validate_siab_contract(
        data, status, primitive_columns, expected_auxiliary_sha256
    )
    occupied, frequencies = _validate_reference_product(
        data, response_channels, expected_nfreq
    )
    return TargetValidationSummary(
        atoms=expected_atoms,
        primitive_columns=primitive_columns,
        auxiliary_channels=auxiliary_channels,
        response_channels=response_channels,
        occupied_states=occupied,
        frequencies=frequencies,
        references=data.q.shape[0],
        abfs_source=status["abfs_source"],
        auxiliary_basis_sha256=auxiliary_basis_sha256,
        primitive_representation=primitive_representation,
    )


def _parse_counts(value):
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("auxiliary counts must be comma-separated integers") from exc
    if not counts or any(item <= 0 for item in counts):
        raise argparse.ArgumentTypeError("auxiliary counts must be positive")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--channels", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--expected-atoms", required=True, type=int)
    parser.add_argument("--primitive-lmax", type=int, default=4)
    parser.add_argument("--radial-primitives", type=int, default=25)
    parser.add_argument("--expected-nfreq", type=int, default=16)
    parser.add_argument("--expected-auxiliary-sha256", required=True)
    parser.add_argument(
        "--auxiliary-radial-counts",
        type=_parse_counts,
        default=DEFAULT_AUXILIARY_RADIAL_COUNTS,
    )
    args = parser.parse_args()

    summary = validate_response_target(
        read_sternheimer(args.target),
        parse_abfs_channels(args.channels),
        parse_sternheimer_status(args.status),
        expected_atoms=args.expected_atoms,
        primitive_lmax=args.primitive_lmax,
        radial_primitives=args.radial_primitives,
        expected_nfreq=args.expected_nfreq,
        auxiliary_radial_counts=args.auxiliary_radial_counts,
        expected_auxiliary_sha256=args.expected_auxiliary_sha256,
    )
    print(json.dumps(summary.__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
