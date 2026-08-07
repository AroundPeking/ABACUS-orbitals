from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch


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
class AuxiliaryChannel:
    channel_index: int
    atom_index: int
    l: int
    radial_index: int
    m: int
    label: str

    def __post_init__(self):
        for field in ("channel_index", "atom_index", "l", "radial_index", "m"):
            if type(getattr(self, field)) is not int:
                raise ValueError(f"{field} must be an integer")
        if self.channel_index < 0:
            raise ValueError("channel_index must be nonnegative")
        if self.atom_index < 0:
            raise ValueError("atom_index must be nonnegative")
        if self.l < 0:
            raise ValueError("l must be nonnegative")
        if self.radial_index < 0:
            raise ValueError("radial_index must be nonnegative")
        if not -self.l <= self.m <= self.l:
            raise ValueError("m must satisfy -l <= m <= l")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be nonempty")


@dataclass(frozen=True)
class SternheimerFixedAOData:
    format_version: int
    representation: str
    energy_unit: str
    channels: Tuple[AuxiliaryChannel, ...]
    eigenvalue_ha: torch.Tensor
    occupation: torch.Tensor
    overlap: torch.Tensor
    hamiltonian_ha: torch.Tensor
    perturbation_ha: torch.Tensor
    frequency_ha: torch.Tensor
    frequency_weight_ha: torch.Tensor
    provenance: Dict[str, object]

    def __post_init__(self):
        if self.format_version != 1:
            raise ValueError(f"format_version must be 1, got {self.format_version}")
        if self.representation != "fixed_lcao_gamma":
            raise ValueError("representation must be fixed_lcao_gamma")
        if self.energy_unit != "Ha":
            raise ValueError("energy_unit must be Ha")
        if not isinstance(self.channels, tuple):
            raise ValueError("channels must be a tuple")
        for index, channel in enumerate(self.channels):
            if not isinstance(channel, AuxiliaryChannel):
                raise ValueError(f"channels[{index}] must be an AuxiliaryChannel")
            if channel.channel_index != index:
                raise ValueError(
                    f"channels[{index}].channel_index expected {index}, "
                    f"got {channel.channel_index}"
                )

        for field in ("eigenvalue_ha", "occupation"):
            self._validate_tensor(field, getattr(self, field), torch.float64, 2)
        for field in ("frequency_ha", "frequency_weight_ha"):
            self._validate_tensor(field, getattr(self, field), torch.float64, 1)
        self._validate_tensor("overlap", self.overlap, torch.complex128, 2)
        self._validate_tensor(
            "hamiltonian_ha", self.hamiltonian_ha, torch.complex128, 3
        )
        self._validate_tensor(
            "perturbation_ha", self.perturbation_ha, torch.complex128, 3
        )
        for field in (
            "eigenvalue_ha",
            "occupation",
            "overlap",
            "hamiltonian_ha",
            "perturbation_ha",
            "frequency_ha",
            "frequency_weight_ha",
        ):
            if not bool(torch.all(torch.isfinite(getattr(self, field)))):
                raise ValueError(f"{field} must contain only finite values")

        if self.overlap.shape[0] == 0 or self.overlap.shape[0] != self.overlap.shape[1]:
            raise ValueError("overlap must be nonempty and square")
        n_basis = self.overlap.shape[0]
        if self.eigenvalue_ha.shape != self.occupation.shape:
            raise ValueError("occupation shape must match eigenvalue_ha")
        n_spin = self.eigenvalue_ha.shape[0]
        if n_spin == 0 or self.eigenvalue_ha.shape[1] != n_basis:
            raise ValueError("eigenvalue_ha shape must be (n_spin, n_basis)")
        if self.hamiltonian_ha.shape != (n_spin, n_basis, n_basis):
            raise ValueError("hamiltonian_ha shape must be (n_spin, n_basis, n_basis)")
        if self.perturbation_ha.shape != (len(self.channels), n_basis, n_basis):
            raise ValueError("perturbation_ha shape must be (n_auxiliary, n_basis, n_basis)")
        if len(self.channels) == 0:
            raise ValueError("at least one auxiliary channel is required")
        if self.frequency_ha.shape[0] == 0:
            raise ValueError("at least one frequency is required")
        if self.frequency_weight_ha.shape != self.frequency_ha.shape:
            raise ValueError("frequency_weight_ha shape must match frequency_ha")
        if bool(torch.any(self.occupation < 0.0)):
            raise ValueError("occupation must be nonnegative")
        if not bool(torch.all(self.frequency_ha > 0.0)):
            raise ValueError("frequency_ha must be positive")
        if bool(torch.any(self.frequency_weight_ha < 0.0)):
            raise ValueError("frequency_weight_ha must be nonnegative")
        self._require_hermitian("OVERLAP_S", self.overlap)
        self._require_hermitian("HAMILTONIAN_H", self.hamiltonian_ha)
        self._require_hermitian("PERTURBATION_V", self.perturbation_ha)

        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a JSON object")
        for key in _PROVENANCE_KEYS:
            if key not in self.provenance:
                raise ValueError(f"missing provenance key: {key}")
        self._validate_cell(self.provenance["cell_bohr"])

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
