"""Virtual eigensystem in the S-orthogonal complement of fixed occupied states."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ResponseOnlyVirtualEigensystem:
    energy_ha: torch.Tensor
    coefficient: torch.Tensor
    virtual_energy_ha: torch.Tensor
    virtual_coefficient: torch.Tensor
    retained_virtual_rank: int
    dropped_trial_rank: int
    projected_overlap_condition: float
    occupied_orthonormality_max_abs_error: float
    occupied_virtual_max_abs_overlap: float
    virtual_orthonormality_max_abs_error: float


@dataclass(frozen=True)
class ResponseOnlyResponse:
    frequency_ha: torch.Tensor
    response_half: torch.Tensor
    response: torch.Tensor


@dataclass(frozen=True)
class ResponseOnlyUnionMatrices:
    overlap: torch.Tensor
    hamiltonian_ha: torch.Tensor
    perturbation_ha: torch.Tensor
    fixed_dimension: int
    response_dimension: int


def assemble_response_only_union(
    fixed_overlap,
    fixed_hamiltonian_ha,
    fixed_perturbation_ha,
    response_overlap,
    response_hamiltonian_ha,
    response_perturbation_ha,
    response_fixed_overlap,
    response_fixed_hamiltonian_ha,
    response_fixed_perturbation_ha,
):
    """Assemble a same-metric fixed-AO plus response-AO operator union."""
    fixed_dimension, response_dimension, auxiliary_count = _validate_union_blocks(
        fixed_overlap,
        fixed_hamiltonian_ha,
        fixed_perturbation_ha,
        response_overlap,
        response_hamiltonian_ha,
        response_perturbation_ha,
        response_fixed_overlap,
        response_fixed_hamiltonian_ha,
        response_fixed_perturbation_ha,
    )
    overlap = _assemble_hermitian_blocks(
        fixed_overlap,
        response_overlap,
        response_fixed_overlap,
    )
    hamiltonian = _assemble_hermitian_blocks(
        fixed_hamiltonian_ha,
        response_hamiltonian_ha,
        response_fixed_hamiltonian_ha,
    )
    perturbation = torch.stack(
        tuple(
            _assemble_hermitian_blocks(fixed, response, cross)
            for fixed, response, cross in zip(
                fixed_perturbation_ha,
                response_perturbation_ha,
                response_fixed_perturbation_ha,
            )
        )
    )
    if perturbation.shape[0] != auxiliary_count:
        raise RuntimeError("assembled perturbation count changed")
    return ResponseOnlyUnionMatrices(
        overlap=overlap,
        hamiltonian_ha=hamiltonian,
        perturbation_ha=perturbation,
        fixed_dimension=fixed_dimension,
        response_dimension=response_dimension,
    )


def evaluate_response_only_sos(
    eigensystem,
    perturbation_ha,
    occupied_occupation,
    frequency_ha,
):
    """Evaluate the spectral form of the projected Sternheimer response."""
    if not isinstance(eigensystem, ResponseOnlyVirtualEigensystem):
        raise ValueError("eigensystem must be a ResponseOnlyVirtualEigensystem")
    _require_tensor("perturbation_ha", perturbation_ha, torch.complex128, 3)
    _require_tensor(
        "occupied_occupation", occupied_occupation, torch.float64, 1
    )
    _require_tensor("frequency_ha", frequency_ha, torch.float64, 1)
    dimension = eigensystem.coefficient.shape[0]
    if perturbation_ha.shape[0] == 0 or perturbation_ha.shape[1:] != (
        dimension,
        dimension,
    ):
        raise ValueError("perturbation_ha shape must be (n_auxiliary, n, n)")
    occupied_count = (
        eigensystem.energy_ha.shape[0]
        - eigensystem.virtual_energy_ha.shape[0]
    )
    if occupied_occupation.shape[0] != occupied_count:
        raise ValueError("occupied occupation count differs from the eigensystem")
    if bool(torch.any(occupied_occupation < 0.0)) or not bool(
        torch.any(occupied_occupation > 0.0)
    ):
        raise ValueError("occupied occupations must be nonnegative and nonempty")
    if frequency_ha.shape[0] == 0 or not bool(torch.all(frequency_ha > 0.0)):
        raise ValueError("frequency_ha must be nonempty and positive")
    _require_hermitian("perturbation_ha", perturbation_ha)

    occupied_coefficient = eigensystem.coefficient[:, :occupied_count]
    occupied_energy = eigensystem.energy_ha[:occupied_count]
    virtual_coefficient = eigensystem.virtual_coefficient
    virtual_energy = eigensystem.virtual_energy_ha
    response_half = []
    for frequency in frequency_ha:
        half = torch.zeros(
            (perturbation_ha.shape[0], perturbation_ha.shape[0]),
            dtype=torch.complex128,
            device="cpu",
        )
        for index in range(occupied_count):
            state = occupied_coefficient[:, index]
            perturbation_on_state = torch.matmul(perturbation_ha, state)
            coupling = virtual_coefficient.mH @ perturbation_on_state.mT
            denominator = (
                virtual_energy - occupied_energy[index] + 1.0j * frequency
            )
            response_coefficient = -coupling / denominator[:, None]
            response_state = virtual_coefficient @ response_coefficient
            half = half + occupied_occupation[index] * (
                perturbation_on_state.conj() @ response_state
            )
        response_half.append(half)
    response_half = torch.stack(tuple(response_half))
    return ResponseOnlyResponse(
        frequency_ha=frequency_ha,
        response_half=response_half,
        response=response_half + response_half.mH,
    )


def solve_response_only_virtual_eigensystem(
    overlap,
    hamiltonian_ha,
    occupied_coefficient,
    occupied_energy_ha,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
    occupied_orthonormality_tolerance=1.0e-10,
):
    """Keep fixed occupied states and diagonalize H only in their S complement."""
    (
        relative_rank_tolerance,
        condition_limit,
        occupied_orthonormality_tolerance,
    ) = _validate_inputs(
        overlap,
        hamiltonian_ha,
        occupied_coefficient,
        occupied_energy_ha,
        relative_rank_tolerance,
        condition_limit,
        occupied_orthonormality_tolerance,
    )

    occupied_overlap = _hermitize(
        occupied_coefficient.mH @ overlap @ occupied_coefficient
    )
    occupied_identity = torch.eye(
        occupied_coefficient.shape[1], dtype=torch.complex128, device="cpu"
    )
    occupied_error = float(
        torch.max(torch.abs(occupied_overlap - occupied_identity))
    )
    if occupied_error > occupied_orthonormality_tolerance:
        raise ValueError("occupied coefficients must be S-orthonormal")

    identity = torch.eye(overlap.shape[0], dtype=torch.complex128, device="cpu")
    projector = identity - occupied_coefficient @ occupied_coefficient.mH @ overlap
    projected_overlap = _hermitize(projector.mH @ overlap @ projector)
    whitener, retained_rank, dropped_rank, condition = _metric_whitener(
        projected_overlap,
        relative_rank_tolerance,
        condition_limit,
    )
    complement_basis = projector @ whitener
    projected_hamiltonian = _hermitize(
        complement_basis.mH @ hamiltonian_ha @ complement_basis
    )
    virtual_energy, virtual_rotation = torch.linalg.eigh(projected_hamiltonian)
    virtual_coefficient = complement_basis @ virtual_rotation

    occupied_virtual_overlap = (
        occupied_coefficient.mH @ overlap @ virtual_coefficient
    )
    virtual_overlap = _hermitize(
        virtual_coefficient.mH @ overlap @ virtual_coefficient
    )
    virtual_identity = torch.eye(
        retained_rank, dtype=torch.complex128, device="cpu"
    )
    occupied_virtual_error = float(torch.max(torch.abs(occupied_virtual_overlap)))
    virtual_error = float(torch.max(torch.abs(virtual_overlap - virtual_identity)))

    return ResponseOnlyVirtualEigensystem(
        energy_ha=torch.cat((occupied_energy_ha, virtual_energy)),
        coefficient=torch.cat((occupied_coefficient, virtual_coefficient), dim=1),
        virtual_energy_ha=virtual_energy,
        virtual_coefficient=virtual_coefficient,
        retained_virtual_rank=retained_rank,
        dropped_trial_rank=dropped_rank,
        projected_overlap_condition=condition,
        occupied_orthonormality_max_abs_error=occupied_error,
        occupied_virtual_max_abs_overlap=occupied_virtual_error,
        virtual_orthonormality_max_abs_error=virtual_error,
    )


def _validate_union_blocks(
    fixed_overlap,
    fixed_hamiltonian_ha,
    fixed_perturbation_ha,
    response_overlap,
    response_hamiltonian_ha,
    response_perturbation_ha,
    response_fixed_overlap,
    response_fixed_hamiltonian_ha,
    response_fixed_perturbation_ha,
):
    for name, value, rank in (
        ("fixed_overlap", fixed_overlap, 2),
        ("fixed_hamiltonian_ha", fixed_hamiltonian_ha, 2),
        ("fixed_perturbation_ha", fixed_perturbation_ha, 3),
        ("response_overlap", response_overlap, 2),
        ("response_hamiltonian_ha", response_hamiltonian_ha, 2),
        ("response_perturbation_ha", response_perturbation_ha, 3),
        ("response_fixed_overlap", response_fixed_overlap, 2),
        ("response_fixed_hamiltonian_ha", response_fixed_hamiltonian_ha, 2),
        ("response_fixed_perturbation_ha", response_fixed_perturbation_ha, 3),
    ):
        _require_tensor(name, value, torch.complex128, rank)
    fixed_dimension = fixed_overlap.shape[0]
    response_dimension = response_overlap.shape[0]
    auxiliary_count = fixed_perturbation_ha.shape[0]
    if fixed_dimension == 0 or fixed_overlap.shape != (
        fixed_dimension,
        fixed_dimension,
    ):
        raise ValueError("fixed_overlap must be nonempty and square")
    if response_dimension == 0 or response_overlap.shape != (
        response_dimension,
        response_dimension,
    ):
        raise ValueError("response_overlap must be nonempty and square")
    if fixed_hamiltonian_ha.shape != fixed_overlap.shape:
        raise ValueError("fixed Hamiltonian shape must match fixed overlap")
    if response_hamiltonian_ha.shape != response_overlap.shape:
        raise ValueError("response Hamiltonian shape must match response overlap")
    if auxiliary_count == 0 or fixed_perturbation_ha.shape[1:] != fixed_overlap.shape:
        raise ValueError("fixed perturbation shape is invalid")
    if response_perturbation_ha.shape != (
        auxiliary_count,
        response_dimension,
        response_dimension,
    ):
        raise ValueError("response perturbation shape is invalid")
    cross_shape = (response_dimension, fixed_dimension)
    if response_fixed_overlap.shape != cross_shape:
        raise ValueError("response-fixed overlap shape is invalid")
    if response_fixed_hamiltonian_ha.shape != cross_shape:
        raise ValueError("response-fixed Hamiltonian shape is invalid")
    if response_fixed_perturbation_ha.shape != (auxiliary_count,) + cross_shape:
        raise ValueError("response-fixed perturbation shape is invalid")
    for name, value in (
        ("fixed_overlap", fixed_overlap),
        ("fixed_hamiltonian_ha", fixed_hamiltonian_ha),
        ("fixed_perturbation_ha", fixed_perturbation_ha),
        ("response_overlap", response_overlap),
        ("response_hamiltonian_ha", response_hamiltonian_ha),
        ("response_perturbation_ha", response_perturbation_ha),
    ):
        _require_hermitian(name, value)
    return fixed_dimension, response_dimension, auxiliary_count


def _assemble_hermitian_blocks(fixed, response, response_fixed):
    return _hermitize(
        torch.cat(
            (
                torch.cat((fixed, response_fixed.mH), dim=1),
                torch.cat((response_fixed, response), dim=1),
            ),
            dim=0,
        )
    )


def _validate_inputs(
    overlap,
    hamiltonian_ha,
    occupied_coefficient,
    occupied_energy_ha,
    relative_rank_tolerance,
    condition_limit,
    occupied_orthonormality_tolerance,
):
    _require_tensor("overlap", overlap, torch.complex128, 2)
    _require_tensor("hamiltonian_ha", hamiltonian_ha, torch.complex128, 2)
    _require_tensor(
        "occupied_coefficient", occupied_coefficient, torch.complex128, 2
    )
    _require_tensor("occupied_energy_ha", occupied_energy_ha, torch.float64, 1)
    if overlap.shape[0] == 0 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap must be nonempty and square")
    if hamiltonian_ha.shape != overlap.shape:
        raise ValueError("hamiltonian_ha shape must match overlap")
    if occupied_coefficient.shape[0] != overlap.shape[0]:
        raise ValueError("occupied coefficient row count must match overlap")
    occupied_count = occupied_coefficient.shape[1]
    if occupied_count == 0 or occupied_count >= overlap.shape[0]:
        raise ValueError("occupied count must be positive and smaller than basis rank")
    if occupied_energy_ha.shape[0] != occupied_count:
        raise ValueError("occupied energy count must match occupied coefficients")
    _require_hermitian("overlap", overlap)
    _require_hermitian("hamiltonian_ha", hamiltonian_ha)

    relative_rank_tolerance = _finite_positive(
        "relative_rank_tolerance", relative_rank_tolerance
    )
    if relative_rank_tolerance >= 1.0:
        raise ValueError("relative_rank_tolerance must be less than one")
    condition_limit = _finite_positive("condition_limit", condition_limit)
    if condition_limit < 1.0:
        raise ValueError("condition_limit must be at least one")
    occupied_orthonormality_tolerance = _finite_positive(
        "occupied_orthonormality_tolerance",
        occupied_orthonormality_tolerance,
    )
    return (
        relative_rank_tolerance,
        condition_limit,
        occupied_orthonormality_tolerance,
    )


def _metric_whitener(metric, relative_rank_tolerance, condition_limit):
    eigenvalue, eigenvector = torch.linalg.eigh(_hermitize(metric))
    maximum = torch.max(eigenvalue)
    if maximum <= 0.0:
        raise RuntimeError("projected overlap has no positive direction")
    threshold = relative_rank_tolerance * maximum
    if bool(torch.any(eigenvalue < -threshold)):
        raise RuntimeError("projected overlap has a materially negative eigenvalue")
    keep = eigenvalue > threshold
    retained = eigenvalue[keep]
    rank = int(torch.count_nonzero(keep))
    dropped = int(eigenvalue.shape[0] - rank)
    condition = float(torch.max(retained) / torch.min(retained))
    if condition > condition_limit:
        raise RuntimeError("projected overlap condition number exceeds limit")
    whitener = eigenvector[:, keep] @ torch.diag(retained.rsqrt()).to(
        torch.complex128
    )
    return whitener, rank, dropped, condition


def _require_tensor(name, value, dtype, rank):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype or value.device.type != "cpu" or value.ndim != rank:
        raise ValueError(f"{name} has invalid dtype, device, or rank")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")


def _require_hermitian(name, value):
    if not torch.allclose(value, value.mH, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{name} must be Hermitian")


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
