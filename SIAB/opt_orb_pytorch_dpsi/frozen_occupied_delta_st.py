"""Delta-ST response in a Bessel parent space with fixed LCAO occupied states."""

from dataclasses import dataclass
import math
from typing import Tuple

import torch

from sternheimer_fixed_ao_data import SternheimerFixedAOData
from sternheimer_primitive_galerkin_data import SternheimerPrimitiveGalerkinData


_PROTOCOL_PROVENANCE_KEYS = (
    "abacus_commit",
    "auxiliary_basis_sha256",
    "cell_bohr",
    "ecut_ry",
    "kernel",
    "orbital_sha256",
    "pseudopotential_sha256",
    "spin_convention",
)


@dataclass(frozen=True)
class FrozenOccupiedDeltaSTResult:
    frequency_ha: torch.Tensor
    response_half: torch.Tensor
    response: torch.Tensor
    active_spin_count: int
    retained_parent_rank_by_spin: Tuple[int, ...]
    dropped_parent_rank_by_spin: Tuple[int, ...]
    projected_overlap_condition_by_spin: Tuple[float, ...]
    fixed_ao_overlap_condition: float
    fixed_ao_eigenvalue_max_abs_error_ha: float


def evaluate_frozen_occupied_delta_st(
    primitive,
    fixed_ao,
    coefficients,
    *,
    include_fixed_ao_virtual=False,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
    eigenvalue_tolerance_ha=1.0e-8,
    active_spin_excluded_columns=(),
):
    """Solve projected Sternheimer equations without changing LCAO occupied states.

    When requested, the trial space is the union of the Bessel parent and the
    unoccupied fixed-LCAO eigenstates, matching the Delta-ST decomposition.
    """
    if type(include_fixed_ao_virtual) is not bool:
        raise ValueError("include_fixed_ao_virtual must be a boolean")
    relative_rank_tolerance, condition_limit, eigenvalue_tolerance_ha = (
        _validate_inputs(
            primitive,
            fixed_ao,
            coefficients,
            relative_rank_tolerance,
            condition_limit,
            eigenvalue_tolerance_ha,
        )
    )
    coefficient_h = coefficients.mH
    active_spin_excluded_columns = _validate_excluded_columns(
        active_spin_excluded_columns,
        coefficients.shape[1],
    )
    parent_overlap = _hermitize(coefficient_h @ primitive.overlap @ coefficients)
    parent_hamiltonian = torch.stack(
        tuple(
            _hermitize(coefficient_h @ value @ coefficients)
            for value in primitive.hamiltonian_ha
        )
    )
    parent_ao_overlap = coefficient_h @ primitive.primitive_ao_overlap
    parent_ao_hamiltonian = torch.stack(
        tuple(
            coefficient_h @ value
            for value in primitive.primitive_ao_hamiltonian_ha
        )
    )
    parent_ao_perturbation = torch.stack(
        tuple(
            coefficient_h @ value
            for value in primitive.primitive_ao_perturbation_ha
        )
    )

    ao_energy = []
    ao_coefficient = []
    fixed_ao_overlap_condition = None
    for spin in range(fixed_ao.hamiltonian_ha.shape[0]):
        energy, coefficient, overlap_condition = _generalized_eigensystem(
            fixed_ao.overlap,
            fixed_ao.hamiltonian_ha[spin],
            relative_rank_tolerance,
            condition_limit,
        )
        ao_energy.append(energy)
        ao_coefficient.append(coefficient)
        if fixed_ao_overlap_condition is None:
            fixed_ao_overlap_condition = overlap_condition
        elif overlap_condition != fixed_ao_overlap_condition:
            raise RuntimeError("fixed-AO overlap condition differs between spins")
    ao_energy = torch.stack(tuple(ao_energy))
    eigenvalue_error = float(
        torch.max(torch.abs(ao_energy - fixed_ao.eigenvalue_ha))
    )
    if eigenvalue_error > eigenvalue_tolerance_ha:
        raise RuntimeError(
            "fixed-AO generalized eigenvalues differ from ABACUS eigenvalues"
        )

    nfrequency = primitive.frequency_ha.shape[0]
    nauxiliary = len(primitive.channels)
    response_half = torch.zeros(
        (nfrequency, nauxiliary, nauxiliary),
        dtype=torch.complex128,
        device="cpu",
    )
    retained_rank = []
    dropped_rank = []
    projected_condition = []
    active_spin_count = 0

    for spin in range(primitive.hamiltonian_ha.shape[0]):
        occupied = fixed_ao.occupation[spin] > 0.0
        trial_overlap = parent_overlap
        trial_hamiltonian = parent_hamiltonian[spin]
        trial_ao_overlap = parent_ao_overlap
        trial_ao_hamiltonian = parent_ao_hamiltonian[spin]
        trial_ao_perturbation = parent_ao_perturbation
        excluded_count = 0
        if bool(torch.any(occupied)) and active_spin_excluded_columns:
            keep = tuple(
                index
                for index in range(parent_overlap.shape[0])
                if index not in active_spin_excluded_columns
            )
            (
                trial_overlap,
                trial_hamiltonian,
                trial_ao_overlap,
                trial_ao_hamiltonian,
                trial_ao_perturbation,
            ) = _select_trial_columns(
                parent_overlap,
                parent_hamiltonian[spin],
                parent_ao_overlap,
                parent_ao_hamiltonian[spin],
                parent_ao_perturbation,
                keep,
            )
            excluded_count = len(active_spin_excluded_columns)
        if include_fixed_ao_virtual and bool(torch.any(~occupied)):
            virtual_state = _orthonormalize_columns(
                ao_coefficient[spin][:, ~occupied],
                primitive.fixed_ao_grid_overlap,
                relative_rank_tolerance,
                condition_limit,
            )
            (
                trial_overlap,
                trial_hamiltonian,
                trial_ao_overlap,
                trial_ao_hamiltonian,
                trial_ao_perturbation,
            ) = _augment_with_fixed_ao_virtuals(
                trial_overlap,
                trial_hamiltonian,
                trial_ao_overlap,
                trial_ao_hamiltonian,
                trial_ao_perturbation,
                primitive.fixed_ao_grid_overlap,
                primitive.fixed_ao_grid_hamiltonian_ha[spin],
                fixed_ao.perturbation_ha,
                virtual_state,
            )
        if not bool(torch.any(occupied)):
            rank, dropped, condition = _positive_metric_rank(
                trial_overlap,
                relative_rank_tolerance,
                condition_limit,
            )
            retained_rank.append(rank)
            dropped_rank.append(dropped)
            projected_condition.append(condition)
            continue

        active_spin_count += 1
        occupied_coefficient = ao_coefficient[spin][:, occupied]
        occupied_state = _normalize_columns(
            occupied_coefficient,
            primitive.fixed_ao_grid_overlap,
        )
        occupied_projector = _orthonormalize_columns(
            occupied_state,
            primitive.fixed_ao_grid_overlap,
            relative_rank_tolerance,
            condition_limit,
        )

        overlap_parent_projector = trial_ao_overlap @ occupied_projector
        hamiltonian_parent_projector = (
            trial_ao_hamiltonian @ occupied_projector
        )
        projector_hamiltonian = _hermitize(
            occupied_projector.mH
            @ primitive.fixed_ao_grid_hamiltonian_ha[spin]
            @ occupied_projector
        )
        projected_overlap = _hermitize(
            trial_overlap
            - overlap_parent_projector @ overlap_parent_projector.mH
        )
        projected_hamiltonian = _hermitize(
            trial_hamiltonian
            - overlap_parent_projector @ hamiltonian_parent_projector.mH
            - hamiltonian_parent_projector @ overlap_parent_projector.mH
            + overlap_parent_projector
            @ projector_hamiltonian
            @ overlap_parent_projector.mH
        )
        transform, rank, dropped, condition = (
            _positive_metric_coordinate_transform(
                projected_overlap,
                relative_rank_tolerance,
                condition_limit,
            )
        )
        transformed_hamiltonian = _hermitize(
            transform.mH @ projected_hamiltonian @ transform
        )
        retained_rank.append(rank)
        dropped_rank.append(dropped + excluded_count)
        projected_condition.append(condition)
        identity = torch.eye(rank, dtype=torch.complex128, device="cpu")

        occupied_indices = torch.nonzero(occupied, as_tuple=False).flatten()
        for local_state, band_index in enumerate(occupied_indices):
            state = occupied_state[:, local_state]
            parent_perturbation_state = torch.stack(
                tuple(value @ state for value in trial_ao_perturbation)
            ).mT
            projector_perturbation_state = torch.stack(
                tuple(
                    occupied_projector.mH @ value @ state
                    for value in fixed_ao.perturbation_ha
                )
            ).mT
            right_hand_side = (
                parent_perturbation_state
                - overlap_parent_projector @ projector_perturbation_state
            )
            transformed_rhs = transform.mH @ right_hand_side
            energy = fixed_ao.eigenvalue_ha[spin, band_index]
            occupation = fixed_ao.occupation[spin, band_index]
            for frequency_index, frequency in enumerate(primitive.frequency_ha):
                shifted = (
                    transformed_hamiltonian
                    - energy * identity
                    + 1.0j * frequency * identity
                )
                response_coefficient = torch.linalg.solve(
                    shifted,
                    -transformed_rhs,
                )
                response_half[frequency_index] += occupation * (
                    transformed_rhs.mH @ response_coefficient
                )

    if active_spin_count == 0:
        raise ValueError("at least one occupied spin channel is required")
    response = _hermitize_without_half(response_half)
    return FrozenOccupiedDeltaSTResult(
        frequency_ha=primitive.frequency_ha,
        response_half=response_half,
        response=response,
        active_spin_count=active_spin_count,
        retained_parent_rank_by_spin=tuple(retained_rank),
        dropped_parent_rank_by_spin=tuple(dropped_rank),
        projected_overlap_condition_by_spin=tuple(projected_condition),
        fixed_ao_overlap_condition=fixed_ao_overlap_condition,
        fixed_ao_eigenvalue_max_abs_error_ha=eigenvalue_error,
    )


def _augment_with_fixed_ao_virtuals(
    parent_overlap,
    parent_hamiltonian,
    parent_ao_overlap,
    parent_ao_hamiltonian,
    parent_ao_perturbation,
    fixed_ao_grid_overlap,
    fixed_ao_grid_hamiltonian,
    fixed_ao_perturbation,
    virtual_state,
):
    parent_virtual_overlap = parent_ao_overlap @ virtual_state
    virtual_overlap = _hermitize(
        virtual_state.mH @ fixed_ao_grid_overlap @ virtual_state
    )
    trial_overlap = _hermitize(
        torch.cat(
            (
                torch.cat((parent_overlap, parent_virtual_overlap), dim=1),
                torch.cat((parent_virtual_overlap.mH, virtual_overlap), dim=1),
            ),
            dim=0,
        )
    )

    parent_virtual_hamiltonian = parent_ao_hamiltonian @ virtual_state
    virtual_hamiltonian = _hermitize(
        virtual_state.mH @ fixed_ao_grid_hamiltonian @ virtual_state
    )
    trial_hamiltonian = _hermitize(
        torch.cat(
            (
                torch.cat(
                    (parent_hamiltonian, parent_virtual_hamiltonian), dim=1
                ),
                torch.cat(
                    (parent_virtual_hamiltonian.mH, virtual_hamiltonian), dim=1
                ),
            ),
            dim=0,
        )
    )

    virtual_ao_overlap = virtual_state.mH @ fixed_ao_grid_overlap
    virtual_ao_hamiltonian = virtual_state.mH @ fixed_ao_grid_hamiltonian
    trial_ao_overlap = torch.cat(
        (parent_ao_overlap, virtual_ao_overlap), dim=0
    )
    trial_ao_hamiltonian = torch.cat(
        (parent_ao_hamiltonian, virtual_ao_hamiltonian), dim=0
    )
    trial_ao_perturbation = torch.stack(
        tuple(
            torch.cat(
                (parent_value, virtual_state.mH @ fixed_value), dim=0
            )
            for parent_value, fixed_value in zip(
                parent_ao_perturbation, fixed_ao_perturbation
            )
        )
    )
    return (
        trial_overlap,
        trial_hamiltonian,
        trial_ao_overlap,
        trial_ao_hamiltonian,
        trial_ao_perturbation,
    )


def _select_trial_columns(
    overlap,
    hamiltonian,
    ao_overlap,
    ao_hamiltonian,
    ao_perturbation,
    keep,
):
    index = torch.tensor(keep, dtype=torch.long, device=overlap.device)
    return (
        overlap.index_select(0, index).index_select(1, index),
        hamiltonian.index_select(0, index).index_select(1, index),
        ao_overlap.index_select(0, index),
        ao_hamiltonian.index_select(0, index),
        ao_perturbation.index_select(1, index),
    )


def _validate_inputs(
    primitive,
    fixed_ao,
    coefficients,
    relative_rank_tolerance,
    condition_limit,
    eigenvalue_tolerance_ha,
):
    if not isinstance(primitive, SternheimerPrimitiveGalerkinData):
        raise ValueError("primitive must be SternheimerPrimitiveGalerkinData")
    if not isinstance(fixed_ao, SternheimerFixedAOData):
        raise ValueError("fixed_ao must be SternheimerFixedAOData")
    if (
        primitive.primitive_ao_hamiltonian_ha is None
        or primitive.primitive_ao_perturbation_ha is None
    ):
        raise ValueError("exact frozen-occupied Delta-ST requires cross matrices")
    _require_tensor("coefficients", coefficients, torch.complex128, 2)
    if coefficients.shape[0] != primitive.overlap.shape[0] or coefficients.shape[1] == 0:
        raise ValueError(
            "coefficients shape must be (n_primitive, nonzero n_parent)"
        )
    if fixed_ao.channels != primitive.channels:
        raise ValueError("fixed-AO and primitive auxiliary channels differ")
    if not torch.equal(fixed_ao.frequency_ha, primitive.frequency_ha):
        raise ValueError("fixed-AO and primitive frequency grids differ")
    if not torch.equal(
        fixed_ao.frequency_weight_ha,
        primitive.frequency_weight_ha,
    ):
        raise ValueError("fixed-AO and primitive frequency weights differ")
    if not torch.equal(fixed_ao.occupation, primitive.occupation):
        raise ValueError("fixed-AO and primitive occupations differ")
    for key in _PROTOCOL_PROVENANCE_KEYS:
        if fixed_ao.provenance[key] != primitive.provenance[key]:
            raise ValueError(f"fixed-AO and primitive provenance differs: {key}")

    relative_rank_tolerance = _finite_positive(
        "relative_rank_tolerance", relative_rank_tolerance
    )
    if relative_rank_tolerance >= 1.0:
        raise ValueError("relative_rank_tolerance must be less than one")
    condition_limit = _finite_positive("condition_limit", condition_limit)
    if condition_limit < 1.0:
        raise ValueError("condition_limit must be at least one")
    eigenvalue_tolerance_ha = _finite_positive(
        "eigenvalue_tolerance_ha", eigenvalue_tolerance_ha
    )
    return relative_rank_tolerance, condition_limit, eigenvalue_tolerance_ha


def _validate_excluded_columns(value, dimension):
    if not isinstance(value, (tuple, list)):
        raise ValueError("active_spin_excluded_columns must be a tuple or list")
    result = []
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("active-spin excluded columns must be integers")
        if index < 0 or index >= dimension:
            raise ValueError("active-spin excluded column is outside the basis")
        if index in result:
            raise ValueError("active-spin excluded columns must be unique")
        result.append(index)
    if len(result) >= dimension:
        raise ValueError("active-spin exclusions cannot remove the full basis")
    return tuple(result)


def _generalized_eigensystem(overlap, hamiltonian, tolerance, condition_limit):
    transform, _, _, condition = _positive_metric_transform(
        overlap,
        tolerance,
        condition_limit,
        allow_dropped=False,
    )
    transformed_hamiltonian = _hermitize(transform.mH @ hamiltonian @ transform)
    energy, eigenvector = torch.linalg.eigh(transformed_hamiltonian)
    return energy, transform @ eigenvector, condition


def _normalize_columns(coefficients, metric):
    norms_squared = torch.real(
        torch.diagonal(coefficients.mH @ metric @ coefficients)
    )
    if bool(torch.any(norms_squared <= 0.0)):
        raise RuntimeError("occupied LCAO state has non-positive grid norm")
    return coefficients / torch.sqrt(norms_squared).reshape(1, -1)


def _orthonormalize_columns(
    coefficients,
    metric,
    relative_rank_tolerance,
    condition_limit,
):
    column_overlap = _hermitize(coefficients.mH @ metric @ coefficients)
    transform, _, _, _ = _positive_metric_transform(
        column_overlap,
        relative_rank_tolerance,
        condition_limit,
        allow_dropped=False,
    )
    return coefficients @ transform


def _positive_metric_rank(metric, tolerance, condition_limit):
    _, rank, dropped, condition = _positive_metric_transform(
        metric,
        tolerance,
        condition_limit,
    )
    return rank, dropped, condition


def _positive_metric_transform(
    metric,
    relative_rank_tolerance,
    condition_limit,
    *,
    allow_dropped=True,
):
    eigenvalue, eigenvector = torch.linalg.eigh(_hermitize(metric))
    maximum = torch.max(eigenvalue)
    if maximum <= 0.0:
        raise RuntimeError("metric has no positive eigenvalue")
    threshold = relative_rank_tolerance * maximum
    if bool(torch.any(eigenvalue < -threshold)):
        raise RuntimeError("metric has a materially negative eigenvalue")
    keep = eigenvalue > threshold
    rank = int(torch.count_nonzero(keep))
    dropped = int(eigenvalue.shape[0] - rank)
    if rank == 0:
        raise RuntimeError("metric has no retained direction")
    if not allow_dropped and dropped != 0:
        raise RuntimeError("metric is rank deficient")
    retained = eigenvalue[keep]
    condition = float(torch.max(retained) / torch.min(retained))
    if condition > condition_limit:
        raise RuntimeError("metric condition number exceeds limit")
    transform = eigenvector[:, keep] @ torch.diag(retained.rsqrt()).to(
        torch.complex128
    )
    return transform, rank, dropped, condition


def _positive_metric_coordinate_transform(
    metric,
    relative_rank_tolerance,
    condition_limit,
):
    """Whiten a fixed-rank metric without differentiating eigenvectors."""
    _, rank, dropped, condition = _positive_metric_transform(
        metric.detach(),
        relative_rank_tolerance,
        condition_limit,
    )
    indices = _pivoted_coordinate_indices(metric.detach(), rank)
    selection = torch.eye(
        metric.shape[0], dtype=metric.dtype, device=metric.device
    )[:, indices]
    coordinate_metric = _hermitize(selection.mH @ metric @ selection)
    cholesky = torch.linalg.cholesky(coordinate_metric)
    identity = torch.eye(rank, dtype=metric.dtype, device=metric.device)
    whitening = torch.linalg.solve_triangular(
        cholesky.mH,
        identity,
        upper=True,
    )
    return selection @ whitening, rank, dropped, condition


def _pivoted_coordinate_indices(metric, rank):
    diagonal = torch.real(torch.diagonal(metric)).clone()
    factor = torch.zeros(
        (metric.shape[0], rank), dtype=metric.dtype, device=metric.device
    )
    selected = []
    available = torch.ones(metric.shape[0], dtype=torch.bool, device=metric.device)
    scale = max(float(torch.max(diagonal)), 1.0)
    minimum_pivot = torch.finfo(torch.float64).eps * scale
    for column in range(rank):
        scores = torch.where(
            available,
            diagonal,
            torch.full_like(diagonal, -torch.inf),
        )
        pivot = int(torch.argmax(scores).item())
        pivot_value = float(scores[pivot])
        if not math.isfinite(pivot_value) or pivot_value <= minimum_pivot:
            raise RuntimeError("metric has no stable coordinate subset")
        selected.append(pivot)
        available[pivot] = False
        correction = (
            0.0
            if column == 0
            else factor[:, :column] @ factor[pivot, :column].conj()
        )
        factor[:, column] = (metric[:, pivot] - correction) / math.sqrt(
            pivot_value
        )
        diagonal = torch.clamp(
            diagonal - torch.abs(factor[:, column]) ** 2,
            min=0.0,
        )
    return torch.tensor(selected, dtype=torch.long, device=metric.device)


def _require_tensor(name, value, dtype, rank):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype or value.device.type != "cpu" or value.ndim != rank:
        raise ValueError(f"{name} has invalid dtype, device, or rank")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")


def _finite_positive(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _hermitize(value):
    return 0.5 * (value + value.mH)


def _hermitize_without_half(value):
    return value + value.mH
