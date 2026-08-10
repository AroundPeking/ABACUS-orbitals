from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch

from sternheimer_data import PrimitiveBlock
from sternheimer_fixed_ao_data import AuxiliaryChannel


_PROVENANCE_KEYS = (
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
class SternheimerPrimitiveGalerkinData:
    format_version: int
    representation: str
    energy_unit: str
    blocks: Tuple[PrimitiveBlock, ...]
    channels: Tuple[AuxiliaryChannel, ...]
    occupation: torch.Tensor
    overlap: torch.Tensor
    hamiltonian_ha: torch.Tensor
    perturbation_ha: torch.Tensor
    primitive_ao_overlap: torch.Tensor
    fixed_ao_grid_overlap: torch.Tensor
    fixed_ao_grid_hamiltonian_ha: torch.Tensor
    frequency_ha: torch.Tensor
    frequency_weight_ha: torch.Tensor
    provenance: Dict[str, object]
    primitive_ao_hamiltonian_ha: Optional[torch.Tensor] = None
    primitive_ao_perturbation_ha: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.format_version != 1:
            raise ValueError(f"format_version must be 1, got {self.format_version}")
        if self.representation not in (
            "bessel_primitive_uniform_grid_gamma",
            "response_orbital_uniform_grid_gamma",
        ):
            raise ValueError(
                "representation must be bessel_primitive_uniform_grid_gamma "
                "or response_orbital_uniform_grid_gamma"
            )
        if self.energy_unit != "Ha":
            raise ValueError("energy_unit must be Ha")
        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise ValueError("blocks must be a nonempty tuple")
        if not isinstance(self.channels, tuple) or not self.channels:
            raise ValueError("channels must be a nonempty tuple")
        for index, block in enumerate(self.blocks):
            if not isinstance(block, PrimitiveBlock):
                raise ValueError(f"blocks[{index}] must be a PrimitiveBlock")
        for index, channel in enumerate(self.channels):
            if not isinstance(channel, AuxiliaryChannel):
                raise ValueError(f"channels[{index}] must be an AuxiliaryChannel")
            if channel.channel_index != index:
                raise ValueError(
                    f"channels[{index}].channel_index expected {index}, "
                    f"got {channel.channel_index}"
                )

        self._validate_tensor("occupation", self.occupation, torch.float64, 2)
        self._validate_tensor("overlap", self.overlap, torch.complex128, 2)
        self._validate_tensor(
            "hamiltonian_ha", self.hamiltonian_ha, torch.complex128, 3
        )
        self._validate_tensor(
            "perturbation_ha", self.perturbation_ha, torch.complex128, 3
        )
        self._validate_tensor(
            "primitive_ao_overlap",
            self.primitive_ao_overlap,
            torch.complex128,
            2,
        )
        self._validate_tensor(
            "fixed_ao_grid_overlap",
            self.fixed_ao_grid_overlap,
            torch.complex128,
            2,
        )
        self._validate_tensor(
            "fixed_ao_grid_hamiltonian_ha",
            self.fixed_ao_grid_hamiltonian_ha,
            torch.complex128,
            3,
        )
        self._validate_tensor(
            "frequency_ha", self.frequency_ha, torch.float64, 1
        )
        self._validate_tensor(
            "frequency_weight_ha", self.frequency_weight_ha, torch.float64, 1
        )

        if self.overlap.shape[0] == 0 or self.overlap.shape[0] != self.overlap.shape[1]:
            raise ValueError("overlap must be nonempty and square")
        n_primitive = self.overlap.shape[0]
        if self.primitive_ao_overlap.shape[0] != n_primitive:
            raise ValueError(
                "primitive_ao_overlap row count must match the primitive dimension"
            )
        n_fixed_ao = self.primitive_ao_overlap.shape[1]
        if n_fixed_ao == 0:
            raise ValueError("primitive_ao_overlap must contain fixed AO columns")
        if self.fixed_ao_grid_overlap.shape != (n_fixed_ao, n_fixed_ao):
            raise ValueError(
                "fixed_ao_grid_overlap shape must be (n_fixed_ao, n_fixed_ao)"
            )
        n_spin = self.occupation.shape[0]
        if n_spin == 0 or self.occupation.shape[1] != n_fixed_ao:
            raise ValueError("occupation shape must be (n_spin, n_fixed_ao)")
        if self.hamiltonian_ha.shape != (n_spin, n_primitive, n_primitive):
            raise ValueError(
                "hamiltonian_ha shape must be (n_spin, n_primitive, n_primitive)"
            )
        if self.fixed_ao_grid_hamiltonian_ha.shape != (
            n_spin,
            n_fixed_ao,
            n_fixed_ao,
        ):
            raise ValueError(
                "fixed_ao_grid_hamiltonian_ha shape must be "
                "(n_spin, n_fixed_ao, n_fixed_ao)"
            )
        if self.perturbation_ha.shape != (
            len(self.channels),
            n_primitive,
            n_primitive,
        ):
            raise ValueError(
                "perturbation_ha shape must be "
                "(n_auxiliary, n_primitive, n_primitive)"
            )
        has_cross_hamiltonian = self.primitive_ao_hamiltonian_ha is not None
        has_cross_perturbation = self.primitive_ao_perturbation_ha is not None
        if has_cross_hamiltonian != has_cross_perturbation:
            raise ValueError(
                "primitive-to-AO Hamiltonian and perturbation data must appear together"
            )
        if has_cross_hamiltonian:
            self._validate_tensor(
                "primitive_ao_hamiltonian_ha",
                self.primitive_ao_hamiltonian_ha,
                torch.complex128,
                3,
            )
            self._validate_tensor(
                "primitive_ao_perturbation_ha",
                self.primitive_ao_perturbation_ha,
                torch.complex128,
                3,
            )
            if self.primitive_ao_hamiltonian_ha.shape != (
                n_spin,
                n_primitive,
                n_fixed_ao,
            ):
                raise ValueError(
                    "primitive_ao_hamiltonian_ha shape must be "
                    "(n_spin, n_primitive, n_fixed_ao)"
                )
            if self.primitive_ao_perturbation_ha.shape != (
                len(self.channels),
                n_primitive,
                n_fixed_ao,
            ):
                raise ValueError(
                    "primitive_ao_perturbation_ha shape must be "
                    "(n_auxiliary, n_primitive, n_fixed_ao)"
                )
        self._validate_block_coverage(n_primitive)

        if bool(torch.any(self.occupation < 0.0)):
            raise ValueError("occupation must be nonnegative")
        if self.frequency_ha.shape[0] == 0:
            raise ValueError("at least one frequency is required")
        if self.frequency_weight_ha.shape != self.frequency_ha.shape:
            raise ValueError("frequency_weight_ha shape must match frequency_ha")
        if not bool(torch.all(self.frequency_ha > 0.0)):
            raise ValueError("frequency_ha must be positive")
        if self.frequency_ha.shape[0] > 1 and not bool(
            torch.all(self.frequency_ha[1:] > self.frequency_ha[:-1])
        ):
            raise ValueError("frequency_ha must be strictly increasing")
        if bool(torch.any(self.frequency_weight_ha < 0.0)):
            raise ValueError("frequency_weight_ha must be nonnegative")

        self._require_hermitian("OVERLAP_S", self.overlap)
        self._require_hermitian("HAMILTONIAN_H", self.hamiltonian_ha)
        self._require_hermitian("PERTURBATION_V", self.perturbation_ha)
        self._require_hermitian(
            "FIXED_AO_GRID_OVERLAP", self.fixed_ao_grid_overlap
        )
        self._require_hermitian(
            "FIXED_AO_GRID_HAMILTONIAN", self.fixed_ao_grid_hamiltonian_ha
        )

        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a JSON object")
        for key in _PROVENANCE_KEYS:
            if key not in self.provenance:
                raise ValueError(f"missing provenance key: {key}")
        self._validate_cell(self.provenance["cell_bohr"])

    def _validate_block_coverage(self, n_primitive):
        expected_offset = 0
        for block in self.blocks:
            if block.offset != expected_offset:
                raise ValueError(
                    "primitive blocks must be contiguous in increasing offset order"
                )
            expected_offset += block.n_primitive
        if expected_offset != n_primitive:
            raise ValueError("primitive blocks do not cover the primitive dimension")

    @staticmethod
    def _validate_tensor(field, value, dtype, rank):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{field} must be a torch.Tensor")
        if value.dtype != dtype:
            raise ValueError(f"{field} must have dtype {dtype}")
        if value.device.type != "cpu":
            raise ValueError(f"{field} must be on CPU")
        if value.ndim != rank:
            raise ValueError(f"{field} must have rank {rank}, got {value.ndim}")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{field} must contain only finite values")

    @staticmethod
    def _require_hermitian(field, value):
        if not torch.allclose(value, value.mH, atol=1.0e-10, rtol=0.0):
            raise ValueError(f"{field} is not Hermitian")

    @staticmethod
    def _validate_cell(cell):
        if not isinstance(cell, (list, tuple)) or len(cell) != 9:
            raise ValueError(
                "provenance cell_bohr must contain nine row-major lattice components"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in cell
        ):
            raise ValueError("provenance cell_bohr components must be finite numbers")
        a, b, c, d, e, f, g, h, i = (float(value) for value in cell)
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (
            d * h - e * g
        )
        if not math.isfinite(determinant) or determinant == 0.0:
            raise ValueError("provenance cell_bohr lattice must be nonsingular")
