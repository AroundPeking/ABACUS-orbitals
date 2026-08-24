"""Efficient coefficient-space periodic Galerkin Sternheimer response."""

from dataclasses import dataclass
import math

import torch

from periodic_galerkin_basis import (
    PeriodicGalerkinCandidateOperators,
    build_primitive_to_candidate,
    contract_periodic_candidate_operators,
)
from periodic_galerkin_data import PeriodicGalerkinDataset


@dataclass(frozen=True)
class PeriodicGalerkinCoefficientResponseResult:
    response_half: torch.Tensor
    response: torch.Tensor
    relative_response_error: torch.Tensor
    minimum_occupied_capture: float
    maximum_overlap_condition: float
    minimum_candidate_rank: int


def _adjoint(value):
    return value.transpose(-2, -1).conj()


def _positive(name, value):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(name + " must be finite and positive")
    return float(value)


def _weighted_relative_error(value, reference, weights):
    weight = weights.reshape((weights.shape[0],) + (1,) * (value.ndim - 1))
    numerator = torch.sum(weight * torch.abs(value - reference) ** 2)
    denominator = torch.sum(weight * torch.abs(reference) ** 2)
    if float(denominator.detach()) == 0.0:
        return torch.sqrt(numerator)
    return torch.sqrt(numerator / denominator)


def evaluate_periodic_galerkin_coefficient_response(
    dataset,
    coefficients,
    *,
    contraction_backend="dense",
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
    occupied_capture_tolerance=1.0e-6,
):
    """Evaluate exact-Pi loss using radial block contractions.

    Unlike the capacity evaluator, this optimization path never reconstructs
    primitive-space dpsi or its Q diagnostic.  It preserves the exact response
    objective while avoiding dense primitive-to-AO products at every step.
    """
    if not isinstance(dataset, PeriodicGalerkinDataset):
        raise ValueError("dataset must be a PeriodicGalerkinDataset")
    if contraction_backend not in ("dense", "block"):
        raise ValueError("contraction_backend must be dense or block")
    relative_rank_tolerance = _positive(
        "relative_rank_tolerance", relative_rank_tolerance
    )
    condition_limit = _positive("condition_limit", condition_limit)
    occupied_capture_tolerance = _positive(
        "occupied_capture_tolerance", occupied_capture_tolerance
    )

    nfrequency = dataset.frequency_ha.shape[0]
    nauxiliary = dataset.whitened_auxiliary_rank
    response_half = torch.zeros(
        (nfrequency, nauxiliary, nauxiliary), dtype=torch.complex128
    )
    minimum_occupied_capture = math.inf
    maximum_overlap_condition = 1.0
    minimum_candidate_rank = dataset.primitive_count
    candidate_basis = None
    if contraction_backend == "dense":
        candidate_basis = build_primitive_to_candidate(
            dataset.primitive_blocks,
            dataset.primitive_count,
            coefficients,
        )

    for record in dataset.kpoints:
        if contraction_backend == "block":
            operators = contract_periodic_candidate_operators(
                record,
                dataset.primitive_blocks,
                coefficients,
            )
        else:
            transform = candidate_basis.transform
            operators = PeriodicGalerkinCandidateOperators(
                overlap=_adjoint(transform).matmul(record.overlap).matmul(transform),
                hamiltonian_ha=_adjoint(transform).matmul(
                    record.hamiltonian_ha
                ).matmul(transform),
                source=record.source.matmul(transform),
                occupied_projection=record.occupied_projection.matmul(transform),
                columns=candidate_basis.columns,
            )
        candidate_count = operators.overlap.shape[0]
        noccupied = record.occupation.shape[0]
        overlap_eigenvalue = torch.linalg.eigvalsh(operators.overlap.detach())
        maximum = torch.max(overlap_eigenvalue)
        if float(maximum.detach()) <= 0.0:
            raise RuntimeError("candidate overlap has no positive direction")
        threshold = relative_rank_tolerance * maximum
        retained = overlap_eigenvalue > threshold
        if bool(torch.any(~retained)):
            raise RuntimeError("candidate overlap is rank deficient")
        retained_eigenvalue = overlap_eigenvalue[retained]
        effective_count = int(retained_eigenvalue.shape[0])
        minimum_candidate_rank = min(minimum_candidate_rank, effective_count)
        condition = float((maximum / torch.min(retained_eigenvalue)).detach())
        if condition > condition_limit:
            raise RuntimeError("candidate overlap condition number exceeds limit")
        maximum_overlap_condition = max(maximum_overlap_condition, condition)

        candidate_identity = torch.eye(candidate_count, dtype=torch.complex128)
        try:
            cholesky = torch.linalg.cholesky(operators.overlap)
        except RuntimeError as error:
            raise RuntimeError("candidate overlap is not positive definite") from error
        lowdin = torch.linalg.solve(_adjoint(cholesky), candidate_identity)
        hamiltonian = _adjoint(lowdin).matmul(
            operators.hamiltonian_ha
        ).matmul(lowdin)
        source = operators.source.matmul(lowdin)
        occupied = operators.occupied_projection.matmul(lowdin)

        capture_matrix = occupied.matmul(_adjoint(occupied))
        capture = torch.linalg.eigvalsh(capture_matrix.detach())
        minimum_capture = float(torch.min(capture).detach())
        minimum_occupied_capture = min(minimum_occupied_capture, minimum_capture)
        if minimum_capture < 1.0 - occupied_capture_tolerance:
            raise RuntimeError("candidate basis does not capture the fixed occupied manifold")
        if effective_count <= noccupied:
            raise RuntimeError("candidate basis has no virtual complement")
        singular_value = torch.linalg.svdvals(occupied.detach())
        if singular_value.shape[0] != noccupied or bool(
            torch.any(
                singular_value
                <= relative_rank_tolerance * torch.max(singular_value)
            )
        ):
            raise RuntimeError("candidate fixed occupied manifold is rank deficient")

        occupied_projector = _adjoint(occupied).matmul(
            torch.linalg.solve(capture_matrix, occupied)
        )
        occupied_projector = 0.5 * (
            occupied_projector + _adjoint(occupied_projector)
        )
        identity = torch.eye(effective_count, dtype=torch.complex128)
        virtual_projector = identity - occupied_projector

        for ifrequency, frequency in enumerate(dataset.frequency_ha):
            for ib in range(noccupied):
                shifted = (
                    hamiltonian
                    - record.source_eigenvalue_ha[ib] * identity
                    + 1.0j * frequency * identity
                )
                system = (
                    virtual_projector.matmul(shifted).matmul(virtual_projector)
                    + occupied_projector
                )
                right_hand_side = -virtual_projector.matmul(_adjoint(source[ib]))
                response = torch.linalg.solve(system, right_hand_side)
                response = virtual_projector.matmul(response)
                response_candidate = lowdin.matmul(response)
                matrix_occupation = record.k_weight * record.occupation[ib]
                response_half[ifrequency] = response_half[ifrequency] + (
                    matrix_occupation
                    * operators.source[ib].matmul(response_candidate)
                )

    response = response_half + _adjoint(response_half)
    return PeriodicGalerkinCoefficientResponseResult(
        response_half=response_half,
        response=response,
        relative_response_error=_weighted_relative_error(
            response,
            dataset.reference_response,
            dataset.frequency_weights_ha,
        ),
        minimum_occupied_capture=minimum_occupied_capture,
        maximum_overlap_condition=maximum_overlap_condition,
        minimum_candidate_rank=minimum_candidate_rank,
    )
