"""Differentiable compact-LCAO response against a grid Delta-ST reference."""

from dataclasses import dataclass
import math
from typing import Tuple

import torch

from delta_st_parent_space import (
    DeltaSTReference,
    FullCoulombMatrix,
    symmetric_response,
    validate_parent_space_protocol,
)
from frozen_occupied_delta_st import evaluate_frozen_occupied_delta_st
from projected_pi_optimization import ProjectedPiOptimizationResult
from sternheimer_fixed_ao_data import SternheimerFixedAOData
from sternheimer_primitive_galerkin_data import SternheimerPrimitiveGalerkinData
from sternheimer_spillage import assemble_orbital_coefficients


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
class DeltaSTCompressionFamilyResult:
    loss: torch.Tensor
    frequency_ha: torch.Tensor
    frequency_weight: torch.Tensor
    frequency_loss: torch.Tensor
    candidate_pi: torch.Tensor
    reference_pi: torch.Tensor
    candidate_response_m: torch.Tensor
    reference_rank: int
    max_candidate_condition: float
    retained_rank_by_spin: Tuple[int, ...]
    dropped_rank_by_spin: Tuple[int, ...]


@dataclass(frozen=True)
class AtomicOccupiedAnchor:
    occupied_band_index: int
    omitted_original_s_zeta: int
    fixed_ao_coefficients: Tuple[float, ...]
    maximum_off_s_coefficient: float
    eigenvalue_max_abs_error_ha: float


class FrozenOccupiedDeltaSTCompression:
    """Evaluate a compact Bessel-contracted LCAO response."""

    def __init__(
        self,
        reference,
        primitive,
        fixed_ao,
        coulomb,
        *,
        family_name="H",
        eigenvalue_threshold=0.0,
        relative_rank_tolerance=1.0e-12,
        condition_limit=1.0e12,
        eigenvalue_tolerance_ha=1.0e-8,
        active_spin_excluded_columns=(),
        include_fixed_ao_virtual=False,
    ):
        validate_parent_space_protocol(reference, primitive, coulomb)
        _validate_fixed_ao_protocol(primitive, fixed_ao)
        if not isinstance(family_name, str) or not family_name:
            raise ValueError("family_name must be nonempty")
        self.reference = reference
        self.primitive = primitive
        self.fixed_ao = fixed_ao
        self.coulomb = coulomb
        self.family_name = family_name
        self.eigenvalue_threshold = _finite_nonnegative(
            "eigenvalue_threshold", eigenvalue_threshold
        )
        self.relative_rank_tolerance = _finite_positive_less_than_one(
            "relative_rank_tolerance", relative_rank_tolerance
        )
        self.condition_limit = _finite_at_least_one(
            "condition_limit", condition_limit
        )
        self.eigenvalue_tolerance_ha = _finite_positive(
            "eigenvalue_tolerance_ha", eigenvalue_tolerance_ha
        )
        self.active_spin_excluded_columns = tuple(active_spin_excluded_columns)
        if type(include_fixed_ao_virtual) is not bool:
            raise ValueError("include_fixed_ao_virtual must be a boolean")
        self.include_fixed_ao_virtual = include_fixed_ao_virtual

        self._reference_pi, self._coulomb_transform = symmetric_response(
            coulomb.matrix,
            reference.response_m,
            self.eigenvalue_threshold,
        )
        self._reference_frequency_norm = torch.sum(
            torch.abs(self._reference_pi) ** 2,
            dim=(1, 2),
        ).real
        self._weighted_reference_norm = torch.dot(
            reference.frequency_weight_ha,
            self._reference_frequency_norm,
        )
        if not bool(torch.isfinite(self._weighted_reference_norm)) or not bool(
            self._weighted_reference_norm > 0.0
        ):
            raise RuntimeError(
                "weighted grid Delta-ST response norm must be positive and finite"
            )

    def evaluate_matrix(self, coefficients):
        _require_coefficient_matrix(coefficients, self.primitive.overlap.shape[0])
        candidate = evaluate_frozen_occupied_delta_st(
            self.primitive,
            self.fixed_ao,
            coefficients,
            relative_rank_tolerance=self.relative_rank_tolerance,
            condition_limit=self.condition_limit,
            eigenvalue_tolerance_ha=self.eigenvalue_tolerance_ha,
            active_spin_excluded_columns=self.active_spin_excluded_columns,
            include_fixed_ao_virtual=self.include_fixed_ao_virtual,
        )
        candidate_pi, transform = symmetric_response(
            self.coulomb.matrix,
            candidate.response,
            self.eigenvalue_threshold,
        )
        if transform != self._coulomb_transform:
            raise RuntimeError("candidate and reference Coulomb transforms differ")

        frequency_error = torch.sum(
            torch.abs(candidate_pi - self._reference_pi) ** 2,
            dim=(1, 2),
        ).real
        if not bool(torch.all(torch.isfinite(frequency_error))):
            raise RuntimeError("Delta-ST compression error must be finite")
        frequency_loss = frequency_error / self._reference_frequency_norm
        loss = torch.dot(
            self.reference.frequency_weight_ha,
            frequency_error,
        ) / self._weighted_reference_norm
        if not bool(torch.isfinite(loss)) or bool(loss < 0.0):
            raise RuntimeError("Delta-ST compression loss must be finite and nonnegative")
        maximum_condition = max(candidate.projected_overlap_condition_by_spin)
        return DeltaSTCompressionFamilyResult(
            loss=loss,
            frequency_ha=self.reference.frequency_ha,
            frequency_weight=self.reference.frequency_weight_ha,
            frequency_loss=frequency_loss,
            candidate_pi=candidate_pi,
            reference_pi=self._reference_pi,
            candidate_response_m=candidate.response,
            reference_rank=self._coulomb_transform.retained_rank,
            max_candidate_condition=maximum_condition,
            retained_rank_by_spin=candidate.retained_parent_rank_by_spin,
            dropped_rank_by_spin=candidate.dropped_parent_rank_by_spin,
        )

    def evaluate(self, radial_coefficients):
        coefficients, _ = assemble_orbital_coefficients(
            self.primitive, radial_coefficients
        )
        family = self.evaluate_matrix(coefficients)
        return ProjectedPiOptimizationResult(
            loss=family.loss,
            max_condition=family.max_candidate_condition,
            frequency_ha=family.frequency_ha,
            frequency_loss=family.frequency_loss,
            family_results={self.family_name: family},
        )


def anchor_atomic_occupied_radial(
    primitive,
    fixed_ao,
    radial_coefficients,
    *,
    element,
    spin=0,
    symmetry_tolerance=1.0e-8,
    eigenvalue_tolerance_ha=1.0e-8,
):
    """Rotate an atomic s basis so one frozen radial column is occupied."""
    _validate_fixed_ao_protocol(primitive, fixed_ao)
    if not isinstance(element, str) or not element:
        raise ValueError("element must be nonempty")
    if isinstance(spin, bool) or not isinstance(spin, int):
        raise ValueError("spin must be an integer")
    if spin < 0 or spin >= fixed_ao.hamiltonian_ha.shape[0]:
        raise ValueError("spin is outside the fixed-AO spin range")
    symmetry_tolerance = _finite_positive(
        "symmetry_tolerance", symmetry_tolerance
    )
    eigenvalue_tolerance_ha = _finite_positive(
        "eigenvalue_tolerance_ha", eigenvalue_tolerance_ha
    )

    coefficient_matrix, labels = assemble_orbital_coefficients(
        primitive, radial_coefficients
    )
    if coefficient_matrix.shape[1] != fixed_ao.overlap.shape[0]:
        raise ValueError("initial radial basis does not match the fixed-AO dimension")
    atoms = {(label.element, label.atom_index) for label in labels}
    if atoms != {(element, 0)}:
        raise ValueError("atomic occupied anchoring requires one atom at index zero")

    energy, eigenvector = _fixed_ao_eigensystem(
        fixed_ao.overlap, fixed_ao.hamiltonian_ha[spin]
    )
    eigenvalue_error = float(
        torch.max(torch.abs(energy - fixed_ao.eigenvalue_ha[spin]))
    )
    if eigenvalue_error > eigenvalue_tolerance_ha:
        raise RuntimeError("fixed-AO eigenvalues differ during occupied anchoring")
    occupied = torch.nonzero(
        fixed_ao.occupation[spin] > 0.0, as_tuple=False
    ).flatten()
    if occupied.numel() != 1:
        raise ValueError("atomic occupied anchoring requires one occupied state")
    occupied_band = int(occupied[0].item())

    s_indices = tuple(
        index
        for index, label in enumerate(labels)
        if label.element == element
        and label.atom_index == 0
        and label.l == 0
        and label.m == 0
    )
    radial_s = radial_coefficients[element][0]
    if len(s_indices) != radial_s.shape[1]:
        raise ValueError("atomic s labels do not match radial s columns")
    off_s_indices = tuple(
        index for index in range(len(labels)) if index not in s_indices
    )
    off_s_maximum = (
        0.0
        if not off_s_indices
        else float(
            torch.max(
                torch.abs(eigenvector[list(off_s_indices), occupied_band])
            )
        )
    )
    if off_s_maximum > symmetry_tolerance:
        raise RuntimeError("atomic occupied state is not an s state")

    weights = eigenvector[list(s_indices), occupied_band]
    omitted = int(torch.argmax(torch.abs(weights)).item())
    phase = weights[omitted] / torch.abs(weights[omitted])
    weights = weights / phase
    imaginary_maximum = float(torch.max(torch.abs(torch.imag(weights))))
    if imaginary_maximum > symmetry_tolerance:
        raise RuntimeError("atomic occupied s coefficients are not real up to phase")
    weights_real = torch.real(weights).to(torch.float64)
    occupied_radial = radial_s @ weights_real
    complement = tuple(
        radial_s[:, index] for index in range(radial_s.shape[1]) if index != omitted
    )

    anchored = _clone_radial_coefficients(radial_coefficients)
    anchored[element][0] = torch.stack(
        (occupied_radial,) + complement,
        dim=1,
    )
    return anchored, AtomicOccupiedAnchor(
        occupied_band_index=occupied_band,
        omitted_original_s_zeta=omitted + 1,
        fixed_ao_coefficients=tuple(float(value) for value in weights_real),
        maximum_off_s_coefficient=off_s_maximum,
        eigenvalue_max_abs_error_ha=eigenvalue_error,
    )


def _validate_fixed_ao_protocol(primitive, fixed_ao):
    if not isinstance(primitive, SternheimerPrimitiveGalerkinData):
        raise ValueError("primitive must be SternheimerPrimitiveGalerkinData")
    if not isinstance(fixed_ao, SternheimerFixedAOData):
        raise ValueError("fixed_ao must be SternheimerFixedAOData")
    if fixed_ao.channels != primitive.channels:
        raise ValueError("fixed-AO and primitive auxiliary channels differ")
    if not torch.equal(fixed_ao.frequency_ha, primitive.frequency_ha):
        raise ValueError("fixed-AO and primitive frequency grids differ")
    if not torch.equal(
        fixed_ao.frequency_weight_ha, primitive.frequency_weight_ha
    ):
        raise ValueError("fixed-AO and primitive frequency weights differ")
    if not torch.equal(fixed_ao.occupation, primitive.occupation):
        raise ValueError("fixed-AO and primitive occupations differ")
    for key in _PROTOCOL_PROVENANCE_KEYS:
        if fixed_ao.provenance[key] != primitive.provenance[key]:
            raise ValueError(f"fixed-AO and primitive provenance differs: {key}")


def _fixed_ao_eigensystem(overlap, hamiltonian):
    cholesky = torch.linalg.cholesky(overlap)
    identity = torch.eye(
        overlap.shape[0], dtype=overlap.dtype, device=overlap.device
    )
    transform = torch.linalg.solve_triangular(
        cholesky.mH,
        identity,
        upper=True,
    )
    transformed_hamiltonian = transform.mH @ hamiltonian @ transform
    transformed_hamiltonian = 0.5 * (
        transformed_hamiltonian + transformed_hamiltonian.mH
    )
    energy, eigenvector = torch.linalg.eigh(transformed_hamiltonian)
    return energy, transform @ eigenvector


def _clone_radial_coefficients(coefficients):
    result = {}
    for element, by_l in coefficients.items():
        if isinstance(by_l, dict):
            result[element] = {
                l: matrix.clone() for l, matrix in by_l.items()
            }
        else:
            result[element] = [matrix.clone() for matrix in by_l]
    return result


def _require_coefficient_matrix(value, nprimitive):
    if not isinstance(value, torch.Tensor):
        raise ValueError("coefficients must be a torch.Tensor")
    if (
        value.dtype != torch.complex128
        or value.device.type != "cpu"
        or value.ndim != 2
    ):
        raise ValueError("coefficients must be a CPU complex128 matrix")
    if value.shape[0] != nprimitive or value.shape[1] == 0:
        raise ValueError(
            "coefficients shape must be (n_primitive, nonzero n_candidate)"
        )
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("coefficients must contain only finite values")


def _finite_nonnegative(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _finite_positive(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _finite_positive_less_than_one(name, value):
    value = _finite_positive(name, value)
    if value >= 1.0:
        raise ValueError(f"{name} must be less than one")
    return value


def _finite_at_least_one(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and at least one") from exc
    if not math.isfinite(value) or value < 1.0:
        raise ValueError(f"{name} must be finite and at least one")
    return value
