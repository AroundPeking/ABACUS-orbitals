"""Fixed-occupied Galerkin Sternheimer response for periodic SIAB bases."""

from dataclasses import dataclass, replace
import math
from typing import Tuple

import torch

from periodic_galerkin_data import PeriodicGalerkinDataset


@dataclass(frozen=True)
class PeriodicGalerkinResponseResult:
    response_half: torch.Tensor
    response: torch.Tensor
    projected_response: Tuple[torch.Tensor, ...]
    relative_response_error: torch.Tensor
    relative_projection_error: torch.Tensor
    minimum_occupied_capture: float
    maximum_overlap_condition: float
    minimum_candidate_rank: int


def _adjoint(value):
    return value.transpose(-2, -1).conj()


def _weighted_relative_error(value, reference, weights):
    leading = (weights.shape[0],) + (1,) * (value.ndim - 1)
    weight = weights.reshape(leading)
    numerator = torch.sum(weight * torch.abs(value - reference) ** 2)
    denominator = torch.sum(weight * torch.abs(reference) ** 2)
    if float(denominator.detach()) == 0.0:
        return torch.sqrt(numerator)
    return torch.sqrt(numerator / denominator)


def evaluate_periodic_galerkin_response(
    dataset,
    primitive_to_candidate,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
    occupied_capture_tolerance=1.0e-6,
):
    """Evaluate Pi in a candidate AO space while retaining ABACUS occupied states.

    ``primitive_to_candidate[j, a]`` defines candidate AO ``a`` as a linear
    combination of the ABACUS Bloch-Bessel primitive functions ``j``.  The
    occupied projector and source eigenvalues are never rediagonalized in the
    candidate space.
    """
    _validate_dataset(dataset)
    if not isinstance(primitive_to_candidate, torch.Tensor):
        raise ValueError("primitive_to_candidate must be a torch.Tensor")
    if primitive_to_candidate.device.type != "cpu" or primitive_to_candidate.ndim != 2:
        raise ValueError("primitive_to_candidate must be a rank-2 CPU tensor")
    if primitive_to_candidate.dtype not in (torch.float64, torch.complex128):
        raise ValueError("primitive_to_candidate must have float64 or complex128 dtype")
    if primitive_to_candidate.shape[0] != dataset.primitive_count:
        raise ValueError("primitive_to_candidate has the wrong primitive dimension")
    if primitive_to_candidate.shape[1] == 0:
        raise ValueError("candidate basis must be nonempty")
    if not bool(torch.isfinite(primitive_to_candidate).all()):
        raise ValueError("primitive_to_candidate contains non-finite values")
    relative_rank_tolerance, condition_limit, occupied_capture_tolerance = (
        _validate_tolerances(
            relative_rank_tolerance,
            condition_limit,
            occupied_capture_tolerance,
        )
    )

    return _evaluate_periodic_galerkin_response(
        dataset,
        primitive_to_candidate.to(torch.complex128),
        relative_rank_tolerance=relative_rank_tolerance,
        condition_limit=condition_limit,
        occupied_capture_tolerance=occupied_capture_tolerance,
        allow_rank_reduction=False,
    )


def evaluate_periodic_galerkin_mother_response(
    dataset,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
    occupied_capture_tolerance=1.0e-6,
):
    """Evaluate the complete primitive span after removing numerical nulls.

    This diagnostic establishes the best response available in the exported
    Bessel mother space.  Candidate-basis optimization remains strict and does
    not silently remove dependent candidate AOs.
    """
    _validate_dataset(dataset)
    relative_rank_tolerance, condition_limit, occupied_capture_tolerance = (
        _validate_tolerances(
            relative_rank_tolerance,
            condition_limit,
            occupied_capture_tolerance,
        )
    )
    return _evaluate_periodic_galerkin_response(
        dataset,
        None,
        relative_rank_tolerance=relative_rank_tolerance,
        condition_limit=condition_limit,
        occupied_capture_tolerance=occupied_capture_tolerance,
        allow_rank_reduction=True,
    )


def prepare_periodic_occupied_reference(
    dataset,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
):
    """Normalize occupied-capture gates to the finite Bessel mother space.

    The left normalization is invertible within the occupied manifold, so it
    leaves the occupied span and the Sternheimer virtual projector unchanged.
    """
    _validate_dataset(dataset)
    relative_rank_tolerance = _finite_positive(
        "relative_rank_tolerance", relative_rank_tolerance
    )
    condition_limit = _finite_positive("condition_limit", condition_limit)
    if relative_rank_tolerance >= 1.0 or condition_limit < 1.0:
        raise ValueError("occupied-reference tolerances are invalid")
    prepared = tuple(
        record.occupied_projection_normalization is not None
        for record in dataset.kpoints
    )
    if all(prepared):
        return dataset
    if any(prepared):
        raise ValueError("periodic dataset has partially prepared occupied references")

    records = []
    with torch.no_grad():
        for record in dataset.kpoints:
            overlap_eigenvalue, overlap_eigenvector = torch.linalg.eigh(
                record.overlap
            )
            maximum = torch.max(overlap_eigenvalue)
            if float(maximum) <= 0.0:
                raise RuntimeError("Bessel mother overlap has no positive direction")
            retained = overlap_eigenvalue > relative_rank_tolerance * maximum
            retained_eigenvalue = overlap_eigenvalue[retained]
            condition = float(maximum / torch.min(retained_eigenvalue))
            if condition > condition_limit:
                raise RuntimeError("Bessel mother overlap condition number exceeds limit")
            lowdin = overlap_eigenvector[:, retained].matmul(
                torch.diag(retained_eigenvalue.rsqrt()).to(torch.complex128)
            )
            occupied = record.occupied_projection.matmul(lowdin)
            capture = occupied.matmul(_adjoint(occupied))
            capture = 0.5 * (capture + _adjoint(capture))
            capture_eigenvalue, capture_eigenvector = torch.linalg.eigh(capture)
            capture_maximum = torch.max(capture_eigenvalue)
            if (
                capture_eigenvalue.numel() != record.occupation.numel()
                or float(capture_maximum) <= 0.0
                or bool(
                    torch.any(
                        capture_eigenvalue
                        <= relative_rank_tolerance * capture_maximum
                    )
                )
            ):
                raise RuntimeError(
                    "Bessel mother space does not capture the fixed occupied manifold"
                )
            if retained_eigenvalue.numel() <= record.occupation.numel():
                raise RuntimeError("Bessel mother space has no virtual complement")
            normalization = (
                capture_eigenvector
                .matmul(torch.diag(capture_eigenvalue.rsqrt()).to(torch.complex128))
                .matmul(_adjoint(capture_eigenvector))
            )
            records.append(
                replace(
                    record,
                    occupied_projection_normalization=normalization,
                )
            )
    return replace(dataset, kpoints=tuple(records))


def _evaluate_periodic_galerkin_response(
    dataset,
    transform,
    *,
    relative_rank_tolerance,
    condition_limit,
    occupied_capture_tolerance,
    allow_rank_reduction,
):
    nfrequency = dataset.frequency_ha.shape[0]
    nauxiliary = dataset.whitened_auxiliary_rank
    response_half = torch.zeros(
        (nfrequency, nauxiliary, nauxiliary), dtype=torch.complex128
    )
    projected_response = []
    minimum_occupied_capture = math.inf
    maximum_overlap_condition = 1.0
    minimum_candidate_rank = dataset.primitive_count

    for record in dataset.kpoints:
        noccupied = record.occupation.shape[0]
        if transform is None:
            candidate_count = dataset.primitive_count
            overlap = record.overlap
            hamiltonian = record.hamiltonian_ha
            source_candidate = record.source
            occupied_candidate = record.occupied_projection
        else:
            candidate_count = transform.shape[1]
            overlap = _adjoint(transform).matmul(record.overlap).matmul(transform)
            hamiltonian = _adjoint(transform).matmul(record.hamiltonian_ha).matmul(transform)
            source_candidate = record.source.matmul(transform)
            occupied_candidate = record.occupied_projection.matmul(transform)
        if record.occupied_projection_normalization is not None:
            occupied_candidate = record.occupied_projection_normalization.matmul(
                occupied_candidate
            )
        overlap_eigenvalue, overlap_eigenvector = torch.linalg.eigh(overlap)
        maximum = torch.max(overlap_eigenvalue)
        if float(maximum.detach()) <= 0.0:
            raise RuntimeError("candidate overlap has no positive direction")
        threshold = relative_rank_tolerance * maximum
        retained = overlap_eigenvalue > threshold
        if not allow_rank_reduction and bool(torch.any(~retained)):
            raise RuntimeError("candidate overlap is rank deficient")
        retained_eigenvalue = overlap_eigenvalue[retained]
        retained_eigenvector = overlap_eigenvector[:, retained]
        effective_count = int(retained_eigenvalue.shape[0])
        minimum_candidate_rank = min(minimum_candidate_rank, effective_count)
        condition = float((maximum / torch.min(retained_eigenvalue)).detach())
        if condition > condition_limit:
            raise RuntimeError("candidate overlap condition number exceeds limit")
        maximum_overlap_condition = max(maximum_overlap_condition, condition)

        lowdin = (
            retained_eigenvector
            @ torch.diag(retained_eigenvalue.rsqrt()).to(torch.complex128)
        )
        hamiltonian_orthonormal = _adjoint(lowdin).matmul(hamiltonian).matmul(lowdin)
        source_orthonormal = source_candidate.matmul(lowdin)
        occupied_orthonormal = occupied_candidate.matmul(lowdin)

        capture_matrix = occupied_orthonormal.matmul(_adjoint(occupied_orthonormal))
        capture = torch.linalg.eigvalsh(capture_matrix)
        minimum_capture = float(torch.min(capture).detach())
        minimum_occupied_capture = min(minimum_occupied_capture, minimum_capture)
        if minimum_capture < 1.0 - occupied_capture_tolerance:
            raise RuntimeError("candidate basis does not capture the fixed occupied manifold")
        if effective_count <= noccupied:
            raise RuntimeError("candidate basis has no virtual complement")
        singular_value = torch.linalg.svdvals(occupied_orthonormal)
        if singular_value.shape[0] != noccupied or bool(
            torch.any(singular_value <= relative_rank_tolerance * torch.max(singular_value))
        ):
            raise RuntimeError("candidate fixed occupied manifold is rank deficient")

        occupied_vectors = _adjoint(occupied_orthonormal)
        if allow_rank_reduction:
            complete_frame, _ = torch.linalg.qr(occupied_vectors, mode="complete")
            occupied_frame = complete_frame[:, :noccupied]
            virtual_frame = complete_frame[:, noccupied:]
        else:
            occupied_frame, _ = torch.linalg.qr(occupied_vectors, mode="reduced")
            virtual_frame = None
        occupied_projector = occupied_frame.matmul(_adjoint(occupied_frame))
        identity = torch.eye(effective_count, dtype=torch.complex128)
        virtual_projector = identity - occupied_projector
        if virtual_frame is not None:
            virtual_hamiltonian = (
                _adjoint(virtual_frame)
                .matmul(hamiltonian_orthonormal)
                .matmul(virtual_frame)
            )
            virtual_hamiltonian = 0.5 * (
                virtual_hamiltonian + _adjoint(virtual_hamiltonian)
            )
            virtual_eigenvalue, virtual_eigenvector = torch.linalg.eigh(
                virtual_hamiltonian
            )
            spectral_frame = virtual_frame.matmul(virtual_eigenvector)
        else:
            virtual_eigenvalue = None
            spectral_frame = None
        per_frequency_projection = []

        for ifrequency, frequency in enumerate(dataset.frequency_ha):
            band_projection = []
            for ib in range(noccupied):
                if spectral_frame is None:
                    shifted = (
                        hamiltonian_orthonormal
                        - record.source_eigenvalue_ha[ib] * identity
                        + 1.0j * frequency * identity
                    )
                    system = (
                        virtual_projector.matmul(shifted).matmul(virtual_projector)
                        + occupied_projector
                    )
                    right_hand_side = -virtual_projector.matmul(
                        _adjoint(source_orthonormal[ib])
                    )
                    response_orthonormal = torch.linalg.solve(
                        system, right_hand_side
                    )
                    response_orthonormal = virtual_projector.matmul(
                        response_orthonormal
                    )
                else:
                    right_hand_side = -_adjoint(spectral_frame).matmul(
                        _adjoint(source_orthonormal[ib])
                    )
                    denominator = (
                        virtual_eigenvalue
                        - record.source_eigenvalue_ha[ib]
                        + 1.0j * frequency
                    )
                    response_orthonormal = spectral_frame.matmul(
                        right_hand_side / denominator[:, None]
                    )
                response_candidate = lowdin.matmul(response_orthonormal)
                response_primitive = (
                    response_candidate
                    if transform is None
                    else transform.matmul(response_candidate)
                )

                matrix_occupation = record.k_weight * record.occupation[ib]
                response_half[ifrequency] = response_half[ifrequency] + matrix_occupation * (
                    record.source[ib].matmul(response_primitive)
                )
                band_projection.append(
                    _adjoint(response_primitive).matmul(record.overlap)
                )
            per_frequency_projection.append(torch.stack(band_projection))
        projected_response.append(torch.stack(per_frequency_projection))

    response = response_half + _adjoint(response_half)
    response_error = _weighted_relative_error(
        response, dataset.reference_response, dataset.frequency_weights_ha
    )

    projection_numerator = torch.zeros((), dtype=torch.float64)
    projection_denominator = torch.zeros((), dtype=torch.float64)
    for record, calculated in zip(dataset.kpoints, projected_response):
        weight = (
            dataset.frequency_weights_ha[:, None, None, None]
            * record.k_weight
            * record.occupation[None, :, None, None]
        )
        projection_numerator = projection_numerator + torch.sum(
            weight * torch.abs(calculated - record.reference_projection) ** 2
        )
        projection_denominator = projection_denominator + torch.sum(
            weight * torch.abs(record.reference_projection) ** 2
        )
    if float(projection_denominator.detach()) == 0.0:
        projection_error = torch.sqrt(projection_numerator)
    else:
        projection_error = torch.sqrt(projection_numerator / projection_denominator)

    return PeriodicGalerkinResponseResult(
        response_half=response_half,
        response=response,
        projected_response=tuple(projected_response),
        relative_response_error=response_error,
        relative_projection_error=projection_error,
        minimum_occupied_capture=minimum_occupied_capture,
        maximum_overlap_condition=maximum_overlap_condition,
        minimum_candidate_rank=minimum_candidate_rank,
    )


def _validate_dataset(dataset):
    if not isinstance(dataset, PeriodicGalerkinDataset):
        raise ValueError("dataset must be a PeriodicGalerkinDataset")


def _validate_tolerances(
    relative_rank_tolerance,
    condition_limit,
    occupied_capture_tolerance,
):
    relative_rank_tolerance = _finite_positive(
        "relative_rank_tolerance", relative_rank_tolerance
    )
    condition_limit = _finite_positive("condition_limit", condition_limit)
    occupied_capture_tolerance = _finite_positive(
        "occupied_capture_tolerance", occupied_capture_tolerance
    )
    if relative_rank_tolerance >= 1.0 or occupied_capture_tolerance >= 1.0:
        raise ValueError("relative tolerances must be less than one")
    if condition_limit < 1.0:
        raise ValueError("condition_limit must be at least one")
    return relative_rank_tolerance, condition_limit, occupied_capture_tolerance


def _finite_positive(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be positive and finite") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(name + " must be positive and finite")
    return value
