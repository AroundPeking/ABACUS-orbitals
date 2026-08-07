from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FiniteAOResponseResult:
    energy: torch.Tensor
    occupation: torch.Tensor
    frequency_ha: torch.Tensor
    response_half: torch.Tensor
    response: torch.Tensor
    overlap_condition: float


def evaluate_galerkin_response(
    overlap,
    hamiltonian,
    perturbation,
    occupation,
    frequency_ha,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
):
    relative_rank_tolerance, condition_limit = _validate_inputs(
        overlap,
        hamiltonian,
        perturbation,
        occupation,
        frequency_ha,
        relative_rank_tolerance,
        condition_limit,
    )
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
    transformed_hamiltonian = lowdin.mH @ hamiltonian @ lowdin
    transformed_perturbation = lowdin.mH @ perturbation @ lowdin
    energy, eigenvector = torch.linalg.eigh(transformed_hamiltonian)

    occupied = occupation > 0.0
    occupied_eigenvector = eigenvector[:, occupied]
    occupied_projector = occupied_eigenvector @ occupied_eigenvector.mH
    identity = torch.eye(
        hamiltonian.shape[0],
        dtype=torch.complex128,
        device=hamiltonian.device,
    )
    virtual_projector = identity - occupied_projector

    response_half = []
    for frequency in frequency_ha:
        half = torch.zeros(
            (perturbation.shape[0], perturbation.shape[0]),
            dtype=torch.complex128,
            device=hamiltonian.device,
        )
        for occupied_index in torch.nonzero(occupied, as_tuple=False).flatten():
            state = eigenvector[:, occupied_index]
            perturbation_on_state = torch.matmul(
                transformed_perturbation,
                state,
            )
            shifted_hamiltonian = (
                transformed_hamiltonian
                - energy[occupied_index] * identity
                + 1.0j * frequency * identity
            )
            system = (
                virtual_projector
                @ shifted_hamiltonian
                @ virtual_projector
                + occupied_projector
            )
            right_hand_side = -virtual_projector @ perturbation_on_state.mT
            response_state = torch.linalg.solve(system, right_hand_side)
            response_state = virtual_projector @ response_state
            half = half + occupation[occupied_index] * (
                perturbation_on_state.conj() @ response_state
            )
        response_half.append(half)

    response_half = torch.stack(response_half)
    response = response_half + response_half.mH
    return FiniteAOResponseResult(
        energy=energy,
        occupation=occupation,
        frequency_ha=frequency_ha,
        response_half=response_half,
        response=response,
        overlap_condition=overlap_condition,
    )


def _validate_inputs(
    overlap,
    hamiltonian,
    perturbation,
    occupation,
    frequency_ha,
    relative_rank_tolerance,
    condition_limit,
):
    _validate_tensor("overlap", overlap, torch.complex128, 2)
    _validate_tensor("hamiltonian", hamiltonian, torch.complex128, 2)
    _validate_tensor("perturbation", perturbation, torch.complex128, 3)
    _validate_tensor("occupation", occupation, torch.float64, 1)
    _validate_tensor("frequency_ha", frequency_ha, torch.float64, 1)

    if overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap must be square")
    dimension = overlap.shape[0]
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError("hamiltonian shape must match overlap")
    if perturbation.shape[0] == 0 or perturbation.shape[1:] != (
        dimension,
        dimension,
    ):
        raise ValueError("perturbation shape must be (n_aux, n, n)")
    if occupation.shape[0] != dimension:
        raise ValueError("occupation length must match overlap")
    if frequency_ha.shape[0] == 0:
        raise ValueError("frequency_ha must be nonempty")

    for name, value in (
        ("overlap", overlap),
        ("hamiltonian", hamiltonian),
        ("perturbation", perturbation),
        ("occupation", occupation),
        ("frequency_ha", frequency_ha),
    ):
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")

    _require_hermitian("overlap", overlap)
    _require_hermitian("hamiltonian", hamiltonian)
    _require_hermitian("perturbation", perturbation)
    if bool(torch.any(occupation < 0.0)):
        raise ValueError("occupation must be nonnegative")
    occupied_count = int(torch.count_nonzero(occupation > 0.0))
    if occupied_count == 0:
        raise ValueError("at least one occupied state is required")
    if occupied_count == dimension:
        raise ValueError("at least one virtual state is required")
    if not bool(torch.all(frequency_ha > 0.0)):
        raise ValueError("frequency_ha must be positive")

    relative_rank_tolerance = _finite_positive(
        "relative_rank_tolerance",
        relative_rank_tolerance,
    )
    if relative_rank_tolerance >= 1.0:
        raise ValueError("relative_rank_tolerance must be less than one")
    condition_limit = _finite_positive("condition_limit", condition_limit)
    if condition_limit < 1.0:
        raise ValueError("condition_limit must be at least one")
    return relative_rank_tolerance, condition_limit


def _validate_tensor(name, value, dtype, rank):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")


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
