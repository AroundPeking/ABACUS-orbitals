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
