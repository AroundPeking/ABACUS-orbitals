"""Map shared SIAB radial coefficients into periodic Bloch-Bessel blocks."""

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Tuple

import torch

from periodic_galerkin_data import (
    PeriodicGalerkinKPoint,
    PeriodicGalerkinPrimitiveBlock,
)


@dataclass(frozen=True)
class PeriodicGalerkinCandidateColumn:
    element: str
    atom_index: int
    l: int
    m: int
    zeta: int


@dataclass(frozen=True)
class PeriodicGalerkinCandidateBasis:
    transform: torch.Tensor
    columns: Tuple[PeriodicGalerkinCandidateColumn, ...]


@dataclass(frozen=True)
class PeriodicGalerkinCandidateOperators:
    overlap: torch.Tensor
    hamiltonian_ha: torch.Tensor
    source: torch.Tensor
    occupied_projection: torch.Tensor
    columns: Tuple[PeriodicGalerkinCandidateColumn, ...]


@dataclass(frozen=True)
class PeriodicGalerkinBlockContractionCache:
    coefficient_profile: tuple
    groups: dict
    active: tuple
    overlap_blocks: dict
    hamiltonian_blocks: dict
    source_blocks: dict
    occupied_projection_blocks: dict


def _next_nonempty(lines, index):
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise ValueError("coefficient file ended unexpectedly")
    return lines[index].strip(), index + 1


def read_periodic_optimizer_coefficients(
    path,
    *,
    element,
    radial_rows,
    expected_nu,
):
    """Read one element from SIAB's native coefficient block.

    This lightweight reader intentionally has no dependency on the molecular
    campaign scripts so that periodic optimization remains usable with the
    Python 3.9 environment on the 66 server.
    """
    if not isinstance(element, str) or not element:
        raise ValueError("element must be nonempty")
    if type(radial_rows) is not int or radial_rows <= 0:
        raise ValueError("radial_rows must be a positive integer")
    try:
        expected_nu = tuple(expected_nu)
    except TypeError as error:
        raise ValueError("expected_nu must be a sequence") from error
    if (
        not expected_nu
        or any(type(count) is not int or count < 0 for count in expected_nu)
        or not any(expected_nu)
    ):
        raise ValueError("expected_nu must contain nonnegative counts and be nonempty")

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        index = next(
            position + 1
            for position, line in enumerate(lines)
            if line.strip() == "<Coefficient>"
        )
    except StopIteration as error:
        raise ValueError("missing <Coefficient> section") from error

    declared_total = None
    columns = {}
    closed = False
    while index < len(lines):
        line, index = _next_nonempty(lines, index)
        if line == "</Coefficient>":
            closed = True
            break
        if "Total number of radial orbitals" in line:
            try:
                declared_total = int(line.split()[0])
            except (IndexError, ValueError) as error:
                raise ValueError("invalid declared coefficient count") from error
            continue
        if not line.startswith("Type"):
            raise ValueError("unexpected coefficient row: " + line)

        label, index = _next_nonempty(lines, index)
        fields = label.split()
        if len(fields) != 3:
            raise ValueError("invalid coefficient label: " + label)
        try:
            l = int(fields[1])
            zeta = int(fields[2])
        except ValueError as error:
            raise ValueError("invalid coefficient label: " + label) from error
        if (
            fields[0] != element
            or l < 0
            or l >= len(expected_nu)
            or zeta <= 0
            or zeta > expected_nu[l]
        ):
            raise ValueError("unexpected coefficient label: " + label)
        key = (l, zeta)
        if key in columns:
            raise ValueError("duplicate coefficient column")

        values = []
        while len(values) < radial_rows:
            value_line, index = _next_nonempty(lines, index)
            if value_line == "</Coefficient>" or value_line.startswith("Type"):
                raise ValueError("incomplete coefficient column")
            try:
                values.extend(float(value) for value in value_line.split())
            except ValueError as error:
                raise ValueError("nonnumeric coefficient column") from error
        if len(values) != radial_rows or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError("invalid coefficient column")
        columns[key] = values

    if not closed:
        raise ValueError("missing </Coefficient> section")
    if declared_total is not None and declared_total != sum(expected_nu):
        raise ValueError("declared coefficient count does not match expected_nu")
    channels = []
    for l, count in enumerate(expected_nu):
        missing = [zeta for zeta in range(1, count + 1) if (l, zeta) not in columns]
        if missing:
            raise ValueError("missing coefficient column")
        if count:
            channel = torch.tensor(
                [columns[(l, zeta)] for zeta in range(1, count + 1)],
                dtype=torch.float64,
            ).transpose(0, 1).contiguous()
        else:
            channel = torch.empty((radial_rows, 0), dtype=torch.float64)
        channels.append(channel)
    return {element: channels}


def write_periodic_optimizer_coefficients(path, coefficients):
    """Write finite float64 coefficients in SIAB's native text format."""
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError("coefficients must be a nonempty dictionary")
    columns = []
    for element in sorted(coefficients):
        if not isinstance(element, str) or not element:
            raise ValueError("coefficient element must be nonempty")
        channels = coefficients[element]
        if not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError("coefficient channels must be a nonempty sequence")
        for l, channel in enumerate(channels):
            if (
                not isinstance(channel, torch.Tensor)
                or channel.device.type != "cpu"
                or channel.dtype != torch.float64
                or channel.ndim != 2
                or channel.shape[0] <= 0
                or not bool(torch.isfinite(channel).all())
            ):
                raise ValueError("coefficient channels must be finite CPU float64 matrices")
            for zeta in range(channel.shape[1]):
                columns.append((element, l, zeta + 1, channel[:, zeta]))
    if not columns:
        raise ValueError("coefficients define no radial orbital")

    output = [
        "<Coefficient>",
        "\t {} Total number of radial orbitals.".format(len(columns)),
    ]
    for element, l, zeta, column in columns:
        output.extend(
            (
                "\tType\tL\tZeta-Orbital",
                "\t  {} \t{}\t    {}".format(element, l, zeta),
            )
        )
        output.extend("\t {:.17e}".format(float(value)) for value in column)
    output.extend(
        (
            "</Coefficient>",
            "<Mkb>",
            "Left spillage = 0.0000000000e+00",
            "</Mkb>",
            "",
        )
    )
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(output), encoding="ascii")


def build_primitive_to_candidate(primitive_blocks, primitive_count, coefficients):
    """Expand element-shared radial coefficients over every atom and m block."""
    if not isinstance(primitive_blocks, tuple) or not primitive_blocks:
        raise ValueError("primitive_blocks must be a nonempty tuple")
    if not isinstance(primitive_count, int) or primitive_count <= 0:
        raise ValueError("primitive_count must be a positive integer")
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError("coefficients must be a nonempty element dictionary")

    columns = []
    labels = []
    expected_offset = 0
    for block in primitive_blocks:
        if not isinstance(block, PeriodicGalerkinPrimitiveBlock):
            raise ValueError("primitive_blocks contains an invalid block")
        if block.offset != expected_offset:
            raise ValueError("primitive blocks do not continuously cover primitive_count")
        expected_offset += block.n_primitive
        if block.element not in coefficients or not isinstance(coefficients[block.element], (list, tuple)):
            raise ValueError("coefficients are missing primitive-block element " + block.element)
        element_channels = coefficients[block.element]
        if block.l >= len(element_channels):
            raise ValueError("coefficients are missing primitive-block angular momentum")
        radial = element_channels[block.l]
        if not isinstance(radial, torch.Tensor) or radial.device.type != "cpu" or radial.ndim != 2:
            raise ValueError("radial coefficients must be rank-2 CPU tensors")
        if radial.dtype not in (torch.float64, torch.complex128):
            raise ValueError("radial coefficients must have float64 or complex128 dtype")
        if radial.shape[0] != block.n_primitive:
            raise ValueError("radial coefficient primitive count does not match ABACUS blocks")
        if not bool(torch.isfinite(radial).all()):
            raise ValueError("radial coefficients contain non-finite values")
        for zeta in range(radial.shape[1]):
            column = torch.zeros(primitive_count, dtype=torch.complex128)
            column[block.offset:block.offset + block.n_primitive] = radial[:, zeta].to(
                torch.complex128
            )
            columns.append(column)
            labels.append(PeriodicGalerkinCandidateColumn(
                element=block.element,
                atom_index=block.atom_index,
                l=block.l,
                m=block.m,
                zeta=zeta,
            ))
    if expected_offset != primitive_count:
        raise ValueError("primitive blocks do not cover primitive_count")
    if not columns:
        raise ValueError("radial coefficients produce an empty candidate basis")
    return PeriodicGalerkinCandidateBasis(
        transform=torch.stack(columns, dim=1),
        columns=tuple(labels),
    )


def _active_groups(primitive_blocks, coefficients):
    groups = {}
    active = []
    for block in primitive_blocks:
        radial = coefficients[block.element][block.l]
        if radial.shape[1] == 0:
            continue
        key = (block.element, block.l)
        group = groups.setdefault(key, [])
        index = len(group)
        group.append(block)
        active.append((block, key, index))
    return groups, active


def _coefficient_profile(coefficients):
    return tuple(
        (element, l, int(channel.shape[0]), int(channel.shape[1]))
        for element in sorted(coefficients)
        for l, channel in enumerate(coefficients[element])
    )


def _stack_operator_blocks(operator, row_blocks, column_blocks):
    return torch.stack(
        tuple(
            torch.stack(
                tuple(
                    operator[
                        row.offset:row.offset + row.n_primitive,
                        column.offset:column.offset + column.n_primitive,
                    ]
                    for column in column_blocks
                ),
                dim=0,
            )
            for row in row_blocks
        ),
        dim=0,
    )


def _stack_row_blocks(rows, blocks):
    return torch.stack(
        tuple(
            rows[..., block.offset:block.offset + block.n_primitive]
            for block in blocks
        ),
        dim=-2,
    )


def prepare_periodic_block_contraction_record(
    record,
    primitive_blocks,
    coefficients,
):
    """Cache coefficient-independent primitive slices for one k point."""
    if not isinstance(record, PeriodicGalerkinKPoint):
        raise ValueError("record must be a PeriodicGalerkinKPoint")
    profile = _coefficient_profile(coefficients)
    if record.block_contraction_cache is not None:
        if record.block_contraction_cache.coefficient_profile != profile:
            raise ValueError("block contraction cache coefficient profile mismatch")
        return record
    groups, active = _active_groups(primitive_blocks, coefficients)
    if not active:
        raise ValueError("candidate basis must be nonempty")
    groups = {key: tuple(blocks) for key, blocks in groups.items()}
    active = tuple(active)
    overlap_blocks = {}
    hamiltonian_blocks = {}
    for row_key, row_blocks in groups.items():
        for column_key, column_blocks in groups.items():
            key = (row_key, column_key)
            overlap_blocks[key] = _stack_operator_blocks(
                record.overlap,
                row_blocks,
                column_blocks,
            )
            hamiltonian_blocks[key] = _stack_operator_blocks(
                record.hamiltonian_ha,
                row_blocks,
                column_blocks,
            )
    cache = PeriodicGalerkinBlockContractionCache(
        coefficient_profile=profile,
        groups=groups,
        active=active,
        overlap_blocks=overlap_blocks,
        hamiltonian_blocks=hamiltonian_blocks,
        source_blocks={
            key: _stack_row_blocks(record.source, blocks)
            for key, blocks in groups.items()
        },
        occupied_projection_blocks={
            key: _stack_row_blocks(record.occupied_projection, blocks)
            for key, blocks in groups.items()
        },
    )
    return replace(record, block_contraction_cache=cache)


def _contract_operator(
    operator,
    groups,
    active,
    coefficients,
    *,
    prepared_blocks=None,
):
    grouped = {}
    for row_key, row_blocks in groups.items():
        row_radial = coefficients[row_key[0]][row_key[1]].to(operator.dtype)
        for column_key, column_blocks in groups.items():
            column_radial = coefficients[column_key[0]][column_key[1]].to(
                operator.dtype
            )
            blocks = (
                _stack_operator_blocks(operator, row_blocks, column_blocks)
                if prepared_blocks is None
                else prepared_blocks[(row_key, column_key)]
            )
            grouped[(row_key, column_key)] = torch.einsum(
                "rz,abrs,sw->azbw",
                row_radial.conj(),
                blocks,
                column_radial,
            )

    rows = []
    for _, row_key, row_index in active:
        rows.append(
            torch.cat(
                tuple(
                    grouped[(row_key, column_key)][
                        row_index, :, column_index, :
                    ]
                    for _, column_key, column_index in active
                ),
                dim=1,
            )
        )
    return torch.cat(tuple(rows), dim=0)


def _contract_rows(
    rows,
    groups,
    active,
    coefficients,
    *,
    prepared_blocks=None,
):
    grouped = {}
    for key, blocks in groups.items():
        radial = coefficients[key[0]][key[1]].to(rows.dtype)
        primitive = (
            _stack_row_blocks(rows, blocks)
            if prepared_blocks is None
            else prepared_blocks[key]
        )
        grouped[key] = torch.einsum("...ar,rz->...az", primitive, radial)
    return torch.cat(
        tuple(grouped[key][..., index, :] for _, key, index in active),
        dim=-1,
    )


def contract_periodic_candidate_operators(record, primitive_blocks, coefficients):
    """Contract periodic primitive operators without dense 1550-by-AO products."""
    if not isinstance(record, PeriodicGalerkinKPoint):
        raise ValueError("record must be a PeriodicGalerkinKPoint")
    primitive_count = record.overlap.shape[0]
    candidate = build_primitive_to_candidate(
        primitive_blocks,
        primitive_count,
        coefficients,
    )
    cache = record.block_contraction_cache
    if cache is None:
        groups, active = _active_groups(primitive_blocks, coefficients)
        if not active:
            raise ValueError("candidate basis must be nonempty")
    else:
        if not isinstance(cache, PeriodicGalerkinBlockContractionCache):
            raise ValueError("record has an invalid block contraction cache")
        if cache.coefficient_profile != _coefficient_profile(coefficients):
            raise ValueError("block contraction cache coefficient profile mismatch")
        groups = cache.groups
        active = cache.active
    return PeriodicGalerkinCandidateOperators(
        overlap=_contract_operator(
            record.overlap,
            groups,
            active,
            coefficients,
            prepared_blocks=None if cache is None else cache.overlap_blocks,
        ),
        hamiltonian_ha=_contract_operator(
            record.hamiltonian_ha,
            groups,
            active,
            coefficients,
            prepared_blocks=None if cache is None else cache.hamiltonian_blocks,
        ),
        source=_contract_rows(
            record.source,
            groups,
            active,
            coefficients,
            prepared_blocks=None if cache is None else cache.source_blocks,
        ),
        occupied_projection=_contract_rows(
            record.occupied_projection,
            groups,
            active,
            coefficients,
            prepared_blocks=(
                None if cache is None else cache.occupied_projection_blocks
            ),
        ),
        columns=candidate.columns,
    )
