"""Direct and spectral gates for a response-orbital Galerkin sidecar."""

from dataclasses import dataclass
from typing import Tuple

import torch

from frozen_occupied_delta_st import (
    FrozenOccupiedDeltaSTResult,
    _generalized_eigensystem,
    evaluate_frozen_occupied_delta_st,
)
from response_only_virtual import (
    assemble_response_only_union,
    evaluate_response_only_sos,
    solve_response_only_virtual_eigensystem,
)


@dataclass(frozen=True)
class ResponseOrbitalGalerkinGateResult:
    direct: FrozenOccupiedDeltaSTResult
    spectral_response: torch.Tensor
    spectral_diagnostics: Tuple[dict, ...]
    spectral_direct_relative_frobenius: float
    spectral_direct_max_abs_difference: float
    response_dimension: int
    fixed_dimension: int


def evaluate_response_orbital_galerkin_gate(
    primitive,
    fixed_ao,
    *,
    relative_rank_tolerance=1.0e-8,
    condition_limit=1.0e12,
    eigenvalue_tolerance_ha=1.0e-8,
):
    """Evaluate all response AOs and compare direct and spectral forms of M."""
    if primitive.representation != "response_orbital_uniform_grid_gamma":
        raise ValueError("the response-orbital gate requires its uniform-grid representation")
    response_dimension = primitive.overlap.shape[0]
    coefficients = torch.eye(response_dimension, dtype=torch.complex128)
    direct = evaluate_frozen_occupied_delta_st(
        primitive,
        fixed_ao,
        coefficients,
        include_fixed_ao_virtual=True,
        relative_rank_tolerance=relative_rank_tolerance,
        condition_limit=condition_limit,
        eigenvalue_tolerance_ha=eigenvalue_tolerance_ha,
    )
    spectral_response, diagnostics = _evaluate_spectral_union(
        primitive,
        fixed_ao,
        coefficients,
        relative_rank_tolerance,
        condition_limit,
    )
    difference = spectral_response - direct.response
    direct_norm = torch.linalg.vector_norm(direct.response)
    if direct_norm <= 0.0:
        raise RuntimeError("the response-orbital gate produced a zero response")
    return ResponseOrbitalGalerkinGateResult(
        direct=direct,
        spectral_response=spectral_response,
        spectral_diagnostics=tuple(diagnostics),
        spectral_direct_relative_frobenius=float(
            torch.linalg.vector_norm(difference) / direct_norm
        ),
        spectral_direct_max_abs_difference=float(torch.max(torch.abs(difference))),
        response_dimension=response_dimension,
        fixed_dimension=fixed_ao.overlap.shape[0],
    )


def _evaluate_spectral_union(
    primitive,
    fixed_ao,
    response_coefficients,
    relative_rank_tolerance,
    condition_limit,
):
    coefficient_h = response_coefficients.mH
    response_overlap = _hermitize(
        coefficient_h @ primitive.overlap @ response_coefficients
    )
    response_fixed_overlap = coefficient_h @ primitive.primitive_ao_overlap
    response_hamiltonian = torch.stack(
        tuple(
            _hermitize(coefficient_h @ value @ response_coefficients)
            for value in primitive.hamiltonian_ha
        )
    )
    response_fixed_hamiltonian = torch.stack(
        tuple(
            coefficient_h @ value
            for value in primitive.primitive_ao_hamiltonian_ha
        )
    )
    response_perturbation = torch.stack(
        tuple(
            _hermitize(coefficient_h @ value @ response_coefficients)
            for value in primitive.perturbation_ha
        )
    )
    response_fixed_perturbation = torch.stack(
        tuple(
            coefficient_h @ value
            for value in primitive.primitive_ao_perturbation_ha
        )
    )

    total = torch.zeros(
        (
            primitive.frequency_ha.shape[0],
            len(primitive.channels),
            len(primitive.channels),
        ),
        dtype=torch.complex128,
    )
    diagnostics = []
    for spin in range(fixed_ao.hamiltonian_ha.shape[0]):
        occupied = fixed_ao.occupation[spin] > 0.0
        if not bool(torch.any(occupied)):
            continue
        if int(torch.count_nonzero(occupied)) != 1:
            raise ValueError(
                "the H2 response-orbital gate requires one occupied state per active spin"
            )
        energy, coefficient, _ = _generalized_eigensystem(
            fixed_ao.overlap,
            fixed_ao.hamiltonian_ha[spin],
            relative_rank_tolerance,
            condition_limit,
        )
        eigenvalue_error = float(
            torch.max(torch.abs(energy - fixed_ao.eigenvalue_ha[spin]))
        )
        occupied_coefficient = coefficient[:, occupied]
        grid_norm = torch.real(
            occupied_coefficient.mH
            @ primitive.fixed_ao_grid_overlap
            @ occupied_coefficient
        ).reshape(())
        if grid_norm <= 0.0:
            raise RuntimeError("fixed occupied state has non-positive grid norm")
        occupied_coefficient = occupied_coefficient / torch.sqrt(grid_norm)
        embedded_occupied = torch.cat(
            (
                occupied_coefficient,
                torch.zeros(
                    (response_coefficients.shape[1], 1),
                    dtype=torch.complex128,
                ),
            ),
            dim=0,
        )
        union = assemble_response_only_union(
            primitive.fixed_ao_grid_overlap,
            primitive.fixed_ao_grid_hamiltonian_ha[spin],
            fixed_ao.perturbation_ha,
            response_overlap,
            response_hamiltonian[spin],
            response_perturbation,
            response_fixed_overlap,
            response_fixed_hamiltonian[spin],
            response_fixed_perturbation,
        )
        eigensystem = solve_response_only_virtual_eigensystem(
            union.overlap,
            union.hamiltonian_ha,
            embedded_occupied,
            fixed_ao.eigenvalue_ha[spin, occupied],
            relative_rank_tolerance=relative_rank_tolerance,
            condition_limit=condition_limit,
        )
        response = evaluate_response_only_sos(
            eigensystem,
            union.perturbation_ha,
            fixed_ao.occupation[spin, occupied],
            primitive.frequency_ha,
        )
        total += response.response
        diagnostics.append(
            {
                "spin": spin,
                "fixed_ao_eigenvalue_max_abs_error_ha": eigenvalue_error,
                "occupied_grid_norm_before_normalization": float(grid_norm),
                "retained_virtual_rank": eigensystem.retained_virtual_rank,
                "dropped_trial_rank": eigensystem.dropped_trial_rank,
                "projected_overlap_condition": (
                    eigensystem.projected_overlap_condition
                ),
                "occupied_orthonormality_max_abs_error": (
                    eigensystem.occupied_orthonormality_max_abs_error
                ),
                "occupied_virtual_max_abs_overlap": (
                    eigensystem.occupied_virtual_max_abs_overlap
                ),
                "virtual_orthonormality_max_abs_error": (
                    eigensystem.virtual_orthonormality_max_abs_error
                ),
                "minimum_virtual_energy_ha": float(
                    eigensystem.virtual_energy_ha[0]
                ),
                "maximum_virtual_energy_ha": float(
                    eigensystem.virtual_energy_ha[-1]
                ),
            }
        )
    if not diagnostics:
        raise ValueError("the H2 response-orbital gate requires an active spin channel")
    return total, diagnostics


def _hermitize(value):
    return 0.5 * (value + value.mH)
