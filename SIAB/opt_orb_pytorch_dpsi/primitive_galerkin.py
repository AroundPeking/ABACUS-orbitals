from dataclasses import dataclass
from typing import Tuple

import torch

from galerkin_sternheimer import evaluate_galerkin_response


@dataclass(frozen=True)
class ContractedPrimitiveMatrices:
    overlap: torch.Tensor
    hamiltonian: torch.Tensor
    perturbation: torch.Tensor


@dataclass(frozen=True)
class PrimitiveGalerkinResult:
    contracted: ContractedPrimitiveMatrices
    frequency_ha: torch.Tensor
    response_half: torch.Tensor
    response: torch.Tensor
    active_spin_count: int
    overlap_condition_by_spin: Tuple[float, ...]


def contract_primitive_matrices(
    overlap,
    hamiltonian,
    perturbation,
    coefficients,
):
    """Contract primitive operators into one candidate AO basis."""
    _validate_primitive_inputs(
        overlap,
        hamiltonian,
        perturbation,
        coefficients,
    )
    coefficient_h = coefficients.mH
    candidate_overlap = coefficient_h @ overlap @ coefficients
    candidate_hamiltonian = torch.stack(
        tuple(coefficient_h @ value @ coefficients for value in hamiltonian)
    )
    candidate_perturbation = torch.stack(
        tuple(coefficient_h @ value @ coefficients for value in perturbation)
    )
    return ContractedPrimitiveMatrices(
        overlap=_hermitize(candidate_overlap),
        hamiltonian=_hermitize(candidate_hamiltonian),
        perturbation=_hermitize(candidate_perturbation),
    )


def evaluate_primitive_galerkin(
    overlap,
    hamiltonian,
    perturbation,
    coefficients,
    occupation,
    frequency_ha,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
):
    contracted = contract_primitive_matrices(
        overlap,
        hamiltonian,
        perturbation,
        coefficients,
    )
    _validate_response_inputs(contracted, occupation, frequency_ha)

    zero = torch.zeros(
        (
            frequency_ha.shape[0],
            contracted.perturbation.shape[0],
            contracted.perturbation.shape[0],
        ),
        dtype=torch.complex128,
        device="cpu",
    )
    response_half_by_spin = []
    response_by_spin = []
    overlap_condition_by_spin = []
    active_spin_count = 0
    empty_spin_condition = _condition_number(contracted.overlap)
    for spin in range(contracted.hamiltonian.shape[0]):
        if not bool(torch.any(occupation[spin] > 0.0)):
            response_half_by_spin.append(zero)
            response_by_spin.append(zero)
            overlap_condition_by_spin.append(empty_spin_condition)
            continue
        result = evaluate_galerkin_response(
            contracted.overlap,
            contracted.hamiltonian[spin],
            contracted.perturbation,
            occupation[spin],
            frequency_ha,
            relative_rank_tolerance=relative_rank_tolerance,
            condition_limit=condition_limit,
        )
        active_spin_count += 1
        response_half_by_spin.append(result.response_half)
        response_by_spin.append(result.response)
        overlap_condition_by_spin.append(result.overlap_condition)

    if active_spin_count == 0:
        raise ValueError("at least one occupied spin channel is required")
    return PrimitiveGalerkinResult(
        contracted=contracted,
        frequency_ha=frequency_ha,
        response_half=torch.stack(tuple(response_half_by_spin)).sum(dim=0),
        response=torch.stack(tuple(response_by_spin)).sum(dim=0),
        active_spin_count=active_spin_count,
        overlap_condition_by_spin=tuple(overlap_condition_by_spin),
    )


def _validate_primitive_inputs(
    overlap,
    hamiltonian,
    perturbation,
    coefficients,
):
    _validate_tensor("overlap", overlap, torch.complex128, 2)
    _validate_tensor("hamiltonian", hamiltonian, torch.complex128, 3)
    _validate_tensor("perturbation", perturbation, torch.complex128, 3)
    _validate_tensor("coefficients", coefficients, torch.complex128, 2)
    if overlap.shape[0] == 0 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap must be nonempty and square")
    n_primitive = overlap.shape[0]
    if hamiltonian.shape[0] == 0 or hamiltonian.shape[1:] != (
        n_primitive,
        n_primitive,
    ):
        raise ValueError(
            "hamiltonian shape must be (n_spin, n_primitive, n_primitive)"
        )
    if perturbation.shape[0] == 0 or perturbation.shape[1:] != (
        n_primitive,
        n_primitive,
    ):
        raise ValueError(
            "perturbation shape must be (n_auxiliary, n_primitive, n_primitive)"
        )
    if coefficients.shape[0] != n_primitive or coefficients.shape[1] == 0:
        raise ValueError(
            "coefficients shape must be (n_primitive, nonzero n_candidate)"
        )
    _require_hermitian("overlap", overlap)
    _require_hermitian("hamiltonian", hamiltonian)
    _require_hermitian("perturbation", perturbation)


def _validate_response_inputs(contracted, occupation, frequency_ha):
    _validate_tensor("occupation", occupation, torch.float64, 2)
    _validate_tensor("frequency_ha", frequency_ha, torch.float64, 1)
    expected = (
        contracted.hamiltonian.shape[0],
        contracted.overlap.shape[0],
    )
    if occupation.shape != expected:
        raise ValueError(
            f"occupation shape must be {expected}, got {tuple(occupation.shape)}"
        )
    if bool(torch.any(occupation < 0.0)):
        raise ValueError("occupation must be nonnegative")
    if frequency_ha.shape[0] == 0 or not bool(torch.all(frequency_ha > 0.0)):
        raise ValueError("frequency_ha must be nonempty and positive")


def _validate_tensor(name, value, dtype, rank):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")


def _require_hermitian(name, value):
    if not torch.allclose(value, value.mH, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{name} must be Hermitian")


def _hermitize(value):
    return 0.5 * (value + value.mH)


def _condition_number(value):
    return float(torch.linalg.cond(value).item())
