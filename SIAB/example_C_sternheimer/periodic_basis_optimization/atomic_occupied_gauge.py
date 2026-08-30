"""Align separately generated atomic source rows to a response-state gauge."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class NaoWavefunctions:
    eigenvalue_ry: torch.Tensor
    occupation: torch.Tensor
    coefficients: torch.Tensor


@dataclass(frozen=True)
class OccupiedGaugeResult:
    transform: torch.Tensor
    occupied_counts: tuple
    subspace_residuals: tuple
    unitarity_errors: tuple
    maximum_subspace_residual: float
    maximum_unitarity_error: float
    maximum_eigenvalue_difference_ry: float


def read_abacus_nao_wavefunctions(path):
    path = Path(path)
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 2:
        raise ValueError("NAO wavefunction file is truncated")
    try:
        number_bands = int(lines[0].split()[0])
        number_orbitals = int(lines[1].split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError("invalid NAO wavefunction header") from error
    if number_bands <= 0 or number_orbitals <= 0:
        raise ValueError("NAO wavefunction dimensions must be positive")

    index = 2
    eigenvalues = []
    occupations = []
    coefficients = []
    for expected_band in range(1, number_bands + 1):
        if index + 2 >= len(lines):
            raise ValueError("NAO wavefunction band header is truncated")
        try:
            band = int(lines[index].split()[0])
            eigenvalue = float(lines[index + 1].split()[0])
            occupation = float(lines[index + 2].split()[0])
        except (IndexError, ValueError) as error:
            raise ValueError("invalid NAO wavefunction band header") from error
        if band != expected_band:
            raise ValueError(
                f"NAO wavefunction band index {band} expected {expected_band}"
            )
        index += 3
        values = []
        while len(values) < number_orbitals:
            if index >= len(lines):
                raise ValueError("NAO wavefunction coefficients are truncated")
            try:
                values.extend(float(value) for value in lines[index].split())
            except ValueError as error:
                raise ValueError("invalid NAO wavefunction coefficient") from error
            index += 1
        if len(values) != number_orbitals:
            raise ValueError("NAO wavefunction band has extra coefficients")
        eigenvalues.append(eigenvalue)
        occupations.append(occupation)
        coefficients.append(values)
    if any(line.strip() for line in lines[index:]):
        raise ValueError("NAO wavefunction file has trailing content")

    eigenvalue_tensor = torch.tensor(eigenvalues, dtype=torch.float64)
    occupation_tensor = torch.tensor(occupations, dtype=torch.float64)
    coefficient_tensor = torch.tensor(coefficients, dtype=torch.float64).T
    for name, value in (
        ("eigenvalues", eigenvalue_tensor),
        ("occupations", occupation_tensor),
        ("coefficients", coefficient_tensor),
    ):
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"NAO wavefunction {name} must be finite")
    return NaoWavefunctions(
        eigenvalue_ry=eigenvalue_tensor,
        occupation=occupation_tensor,
        coefficients=coefficient_tensor,
    )


def derive_occupied_gauge(
    response_paths,
    source_paths,
    *,
    occupation_threshold=0.5,
    residual_tolerance=1.0e-8,
    unitarity_tolerance=1.0e-8,
    eigenvalue_tolerance_ry=1.0e-7,
):
    response_paths = tuple(Path(path) for path in response_paths)
    source_paths = tuple(Path(path) for path in source_paths)
    if not response_paths or len(response_paths) != len(source_paths):
        raise ValueError("response/source spin wavefunction files must match")
    for name, value in (
        ("occupation_threshold", occupation_threshold),
        ("residual_tolerance", residual_tolerance),
        ("unitarity_tolerance", unitarity_tolerance),
        ("eigenvalue_tolerance_ry", eigenvalue_tolerance_ry),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    transforms = []
    occupied_counts = []
    subspace_residuals = []
    unitarity_errors = []
    maximum_eigenvalue_difference = 0.0
    for spin_index, (response_path, source_path) in enumerate(
        zip(response_paths, source_paths), start=1
    ):
        response = read_abacus_nao_wavefunctions(response_path)
        source = read_abacus_nao_wavefunctions(source_path)
        if response.coefficients.shape != source.coefficients.shape:
            raise ValueError(f"spin {spin_index} wavefunction shapes differ")
        response_occupied = response.occupation > occupation_threshold
        source_occupied = source.occupation > occupation_threshold
        if not torch.equal(response_occupied, source_occupied):
            raise ValueError(f"spin {spin_index} occupied bands differ")
        count = int(torch.count_nonzero(response_occupied))
        if count == 0:
            raise ValueError(f"spin {spin_index} has no occupied bands")
        occupied_counts.append(count)
        response_eigenvalue = response.eigenvalue_ry[response_occupied]
        source_eigenvalue = source.eigenvalue_ry[source_occupied]
        eigenvalue_difference = float(
            torch.max(torch.abs(response_eigenvalue - source_eigenvalue))
        )
        maximum_eigenvalue_difference = max(
            maximum_eigenvalue_difference, eigenvalue_difference
        )
        if eigenvalue_difference > eigenvalue_tolerance_ry:
            raise ValueError(
                f"spin {spin_index} occupied eigenvalue difference "
                f"{eigenvalue_difference:.6g} Ry exceeds tolerance"
            )

        response_coefficients = response.coefficients[:, response_occupied]
        source_coefficients = source.coefficients[:, source_occupied]
        transform = torch.linalg.lstsq(
            response_coefficients,
            source_coefficients,
        ).solution
        residual = float(
            torch.linalg.vector_norm(
                response_coefficients @ transform - source_coefficients
            )
            / torch.linalg.vector_norm(source_coefficients)
        )
        identity = torch.eye(count, dtype=transform.dtype)
        unitarity_error = float(
            torch.linalg.vector_norm(transform.T @ transform - identity)
        )
        if residual > residual_tolerance:
            raise ValueError(
                f"spin {spin_index} occupied subspace residual "
                f"{residual:.6g} exceeds tolerance"
            )
        if unitarity_error > unitarity_tolerance:
            raise ValueError(
                f"spin {spin_index} occupied gauge is not unitary: "
                f"{unitarity_error:.6g}"
            )
        transforms.append(transform.to(torch.complex128))
        subspace_residuals.append(residual)
        unitarity_errors.append(unitarity_error)

    return OccupiedGaugeResult(
        transform=torch.block_diag(*transforms),
        occupied_counts=tuple(occupied_counts),
        subspace_residuals=tuple(subspace_residuals),
        unitarity_errors=tuple(unitarity_errors),
        maximum_subspace_residual=max(subspace_residuals),
        maximum_unitarity_error=max(unitarity_errors),
        maximum_eigenvalue_difference_ry=maximum_eigenvalue_difference,
    )


def rotate_source_rows_to_response_gauge(
    source_d,
    occupied_state,
    auxiliary_channel,
    response_to_source_transform,
):
    if source_d.ndim != 2:
        raise ValueError("source D must be a matrix")
    if occupied_state.ndim != 1 or auxiliary_channel.ndim != 1:
        raise ValueError("source metadata must be vectors")
    if source_d.shape[0] != occupied_state.shape[0] or source_d.shape[0] != (
        auxiliary_channel.shape[0]
    ):
        raise ValueError("source rows and metadata lengths differ")
    states = tuple(sorted(set(occupied_state.tolist())))
    channels = tuple(sorted(set(auxiliary_channel.tolist())))
    if states != tuple(range(len(states))):
        raise ValueError("occupied state IDs must be contiguous from zero")
    if response_to_source_transform.shape != (len(states), len(states)):
        raise ValueError("occupied gauge transform shape differs from states")
    rows = {}
    for row, key in enumerate(
        zip(occupied_state.tolist(), auxiliary_channel.tolist())
    ):
        if key in rows:
            raise ValueError(f"duplicate source key {key}")
        rows[key] = row
    expected = {(state, channel) for state in states for channel in channels}
    if set(rows) != expected:
        raise ValueError("source rows do not form an occupied/channel rectangle")

    aligned = source_d.clone()
    for channel in channels:
        channel_rows = [rows[(state, channel)] for state in states]
        aligned[channel_rows] = (
            response_to_source_transform.conj() @ source_d[channel_rows]
        )
    return aligned
