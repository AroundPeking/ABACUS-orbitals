from dataclasses import dataclass
import math
from typing import Tuple

import torch

from galerkin_sternheimer import evaluate_galerkin_response, evaluate_sos_response
from sternheimer_fixed_ao_data import SternheimerFixedAOData


@dataclass(frozen=True)
class FixedAOSidecarResponseResult:
    frequency_ha: torch.Tensor
    frequency_weight_ha: torch.Tensor
    galerkin_response_half: torch.Tensor
    galerkin_response: torch.Tensor
    sos_response_half: torch.Tensor
    sos_response: torch.Tensor
    galerkin_sos_relative_error: float
    galerkin_sos_max_abs_error: float
    eigenvalue_max_abs_error_ha: float
    overlap_condition_by_spin: Tuple[float, ...]


def evaluate_fixed_ao_sidecar(
    data,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
):
    if not isinstance(data, SternheimerFixedAOData):
        raise ValueError("data must be a SternheimerFixedAOData")

    galerkin_response_half_by_spin = []
    galerkin_response_by_spin = []
    sos_response_half_by_spin = []
    sos_response_by_spin = []
    calculated_eigenvalue = []
    overlap_condition_by_spin = []
    active_spin_count = 0
    for spin in range(data.hamiltonian_ha.shape[0]):
        energy, overlap_condition = _generalized_eigenvalues(
            data.overlap,
            data.hamiltonian_ha[spin],
            relative_rank_tolerance,
            condition_limit,
        )
        calculated_eigenvalue.append(energy)
        overlap_condition_by_spin.append(overlap_condition)
        if not bool(torch.any(data.occupation[spin] > 0.0)):
            zero = torch.zeros(
                (
                    data.frequency_ha.shape[0],
                    data.perturbation_ha.shape[0],
                    data.perturbation_ha.shape[0],
                ),
                dtype=torch.complex128,
                device="cpu",
            )
            galerkin_response_half_by_spin.append(zero)
            galerkin_response_by_spin.append(zero)
            sos_response_half_by_spin.append(zero)
            sos_response_by_spin.append(zero)
            continue
        active_spin_count += 1
        inputs = (
            data.overlap,
            data.hamiltonian_ha[spin],
            data.perturbation_ha,
            data.occupation[spin],
            data.frequency_ha,
        )
        galerkin = evaluate_galerkin_response(
            *inputs,
            relative_rank_tolerance=relative_rank_tolerance,
            condition_limit=condition_limit,
        )
        sos = evaluate_sos_response(
            *inputs,
            relative_rank_tolerance=relative_rank_tolerance,
            condition_limit=condition_limit,
        )
        galerkin_response_half_by_spin.append(galerkin.response_half)
        galerkin_response_by_spin.append(galerkin.response)
        sos_response_half_by_spin.append(sos.response_half)
        sos_response_by_spin.append(sos.response)

    if active_spin_count == 0:
        raise ValueError("at least one occupied spin channel is required")

    galerkin_response_half = torch.stack(
        tuple(galerkin_response_half_by_spin)
    ).sum(dim=0)
    galerkin_response = torch.stack(
        tuple(galerkin_response_by_spin)
    ).sum(dim=0)
    sos_response_half = torch.stack(tuple(sos_response_half_by_spin)).sum(dim=0)
    sos_response = torch.stack(tuple(sos_response_by_spin)).sum(dim=0)
    difference = galerkin_response - sos_response
    difference_norm = float(torch.linalg.vector_norm(difference))
    sos_norm = float(torch.linalg.vector_norm(sos_response))
    relative_error = (
        difference_norm / sos_norm
        if sos_norm > 0.0
        else (0.0 if difference_norm == 0.0 else math.inf)
    )
    maximum_absolute_error = float(torch.max(torch.abs(difference)))
    calculated_eigenvalue = torch.stack(tuple(calculated_eigenvalue))
    eigenvalue_error = float(
        torch.max(torch.abs(calculated_eigenvalue - data.eigenvalue_ha))
    )

    return FixedAOSidecarResponseResult(
        frequency_ha=data.frequency_ha,
        frequency_weight_ha=data.frequency_weight_ha,
        galerkin_response_half=galerkin_response_half,
        galerkin_response=galerkin_response,
        sos_response_half=sos_response_half,
        sos_response=sos_response,
        galerkin_sos_relative_error=relative_error,
        galerkin_sos_max_abs_error=maximum_absolute_error,
        eigenvalue_max_abs_error_ha=eigenvalue_error,
        overlap_condition_by_spin=tuple(overlap_condition_by_spin),
    )


def _generalized_eigenvalues(
    overlap,
    hamiltonian,
    relative_rank_tolerance,
    condition_limit,
):
    relative_rank_tolerance = _finite_positive(
        "relative_rank_tolerance", relative_rank_tolerance
    )
    if relative_rank_tolerance >= 1.0:
        raise ValueError("relative_rank_tolerance must be less than one")
    condition_limit = _finite_positive("condition_limit", condition_limit)
    if condition_limit < 1.0:
        raise ValueError("condition_limit must be at least one")

    overlap_eigenvalue, overlap_eigenvector = torch.linalg.eigh(overlap)
    maximum_overlap = torch.max(overlap_eigenvalue)
    threshold = relative_rank_tolerance * maximum_overlap
    if bool(torch.any(overlap_eigenvalue <= threshold)):
        raise RuntimeError("overlap is rank deficient")
    overlap_condition = float(maximum_overlap / torch.min(overlap_eigenvalue))
    if overlap_condition > condition_limit:
        raise RuntimeError("overlap condition number exceeds limit")
    lowdin = (
        overlap_eigenvector
        @ torch.diag(overlap_eigenvalue.rsqrt()).to(torch.complex128)
        @ overlap_eigenvector.mH
    )
    energy = torch.linalg.eigvalsh(lowdin.mH @ hamiltonian @ lowdin)
    return energy, overlap_condition


def _finite_positive(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value
