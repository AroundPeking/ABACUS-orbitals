from dataclasses import dataclass
from typing import Tuple

import torch

from sternheimer_primitive_galerkin_data import (
    SternheimerPrimitiveGalerkinData,
)


@dataclass(frozen=True)
class PrimitiveProjectionResult:
    coefficients: torch.Tensor
    numerical_rank: int
    retained_condition: float
    rank_cutoff: float
    cross_overlap_relative_residual: float
    projected_overlap: torch.Tensor
    overlap_relative_residual: float
    projected_hamiltonian_ha: torch.Tensor
    hamiltonian_relative_residual: Tuple[float, ...]
    projected_perturbation_ha: torch.Tensor


def project_fixed_ao_to_primitives(
    data,
    *,
    relative_rank_tolerance=1.0e-10,
):
    if not isinstance(data, SternheimerPrimitiveGalerkinData):
        raise ValueError("data must be SternheimerPrimitiveGalerkinData")
    if (
        isinstance(relative_rank_tolerance, bool)
        or not isinstance(relative_rank_tolerance, (int, float))
        or not 0.0 < float(relative_rank_tolerance) < 1.0
    ):
        raise ValueError("relative_rank_tolerance must lie strictly between 0 and 1")

    overlap = _hermitize(data.overlap)
    eigenvalues, eigenvectors = torch.linalg.eigh(overlap)
    largest = float(torch.max(eigenvalues).item())
    if largest <= 0.0:
        raise RuntimeError("primitive overlap has no positive modes")
    cutoff = float(relative_rank_tolerance) * largest
    if float(torch.min(eigenvalues).item()) < -cutoff:
        raise RuntimeError("primitive overlap is materially indefinite")
    keep = eigenvalues > cutoff
    numerical_rank = int(torch.count_nonzero(keep).item())
    if numerical_rank == 0:
        raise RuntimeError("primitive overlap has numerical rank zero")

    retained_eigenvalues = eigenvalues[keep]
    retained_eigenvectors = eigenvectors[:, keep]
    pseudo_inverse = (
        retained_eigenvectors / retained_eigenvalues
    ) @ retained_eigenvectors.mH
    coefficients = pseudo_inverse @ data.primitive_ao_overlap

    projected_overlap = _hermitize(coefficients.mH @ overlap @ coefficients)
    projected_hamiltonian = _hermitize(
        torch.stack(
            tuple(
                coefficients.mH @ value @ coefficients
                for value in data.hamiltonian_ha
            )
        )
    )
    projected_perturbation = _hermitize(
        torch.stack(
            tuple(
                coefficients.mH @ value @ coefficients
                for value in data.perturbation_ha
            )
        )
    )
    cross_residual = overlap @ coefficients - data.primitive_ao_overlap

    return PrimitiveProjectionResult(
        coefficients=coefficients,
        numerical_rank=numerical_rank,
        retained_condition=float(
            (torch.max(retained_eigenvalues) / torch.min(retained_eigenvalues)).item()
        ),
        rank_cutoff=cutoff,
        cross_overlap_relative_residual=_relative_residual(
            cross_residual, data.primitive_ao_overlap
        ),
        projected_overlap=projected_overlap,
        overlap_relative_residual=_relative_residual(
            projected_overlap - data.fixed_ao_grid_overlap,
            data.fixed_ao_grid_overlap,
        ),
        projected_hamiltonian_ha=projected_hamiltonian,
        hamiltonian_relative_residual=tuple(
            _relative_residual(
                projected_hamiltonian[spin]
                - data.fixed_ao_grid_hamiltonian_ha[spin],
                data.fixed_ao_grid_hamiltonian_ha[spin],
            )
            for spin in range(projected_hamiltonian.shape[0])
        ),
        projected_perturbation_ha=projected_perturbation,
    )


def _relative_residual(difference, reference):
    denominator = max(
        float(torch.linalg.norm(reference).item()),
        torch.finfo(torch.float64).eps,
    )
    return float(torch.linalg.norm(difference).item()) / denominator


def _hermitize(value):
    return 0.5 * (value + value.mH)
