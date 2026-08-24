"""Map shared SIAB radial coefficients into periodic Bloch-Bessel blocks."""

from dataclasses import dataclass
from typing import Tuple

import torch

from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock


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
