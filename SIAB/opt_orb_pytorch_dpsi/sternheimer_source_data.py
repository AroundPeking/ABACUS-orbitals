from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch

from sternheimer_data import PrimitiveBlock


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
class SternheimerSourceData:
    format_version: int
    grid_volume_bohr3: float
    blocks: Tuple[PrimitiveBlock, ...]
    occupied_state: torch.Tensor
    auxiliary_channel: torch.Tensor
    occupation: torch.Tensor
    norm: torch.Tensor
    d: torch.Tensor
    overlap: torch.Tensor
    provenance: Dict[str, object]

    def __post_init__(self):
        if self.format_version != 1:
            raise ValueError(
                f"format_version must be 1, got {self.format_version}"
            )
        try:
            volume = float(self.grid_volume_bohr3)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "grid_volume_bohr3 must be a positive number"
            ) from exc
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError("grid_volume_bohr3 must be positive")
        if not isinstance(self.blocks, tuple):
            raise ValueError("blocks must be a tuple")

        self._validate_tensor(
            "occupied_state", self.occupied_state, torch.int64, rank=1
        )
        self._validate_tensor(
            "auxiliary_channel", self.auxiliary_channel, torch.int64, rank=1
        )
        self._validate_tensor(
            "occupation", self.occupation, torch.float64, rank=1
        )
        self._validate_tensor("norm", self.norm, torch.float64, rank=1)
        self._validate_tensor("d", self.d, torch.complex128, rank=2)
        self._validate_tensor(
            "overlap", self.overlap, torch.complex128, rank=2
        )
        for field in ("occupation", "norm", "d", "overlap"):
            if not bool(torch.all(torch.isfinite(getattr(self, field)))):
                raise ValueError(f"{field} must contain only finite values")

        n_source = self.occupied_state.shape[0]
        for field in ("auxiliary_channel", "occupation", "norm"):
            size = getattr(self, field).shape[0]
            if size != n_source:
                raise ValueError(
                    f"{field} length {size} does not match "
                    f"occupied_state length {n_source}"
                )
        if self.d.shape[0] != n_source:
            raise ValueError(
                f"d row count {self.d.shape[0]} does not match "
                f"source count {n_source}"
            )
        if not bool(torch.all(self.occupied_state >= 0)):
            raise ValueError(
                "occupied_state must contain only nonnegative values"
            )
        if not bool(torch.all(self.auxiliary_channel >= 0)):
            raise ValueError(
                "auxiliary_channel must contain only nonnegative values"
            )
        keys = list(
            zip(
                self.occupied_state.tolist(),
                self.auxiliary_channel.tolist(),
            )
        )
        if len(keys) != len(set(keys)):
            duplicate = next(key for key in keys if keys.count(key) > 1)
            raise ValueError(f"duplicate source key: {duplicate}")

        if not bool(torch.all(self.occupation > 0.0)):
            raise ValueError("occupation must contain only positive values")
        if not bool(torch.all(self.norm > 0.0)):
            raise ValueError("norm must contain only positive values")
        if self.overlap.shape[0] != self.overlap.shape[1]:
            raise ValueError(
                f"overlap must be square, got shape {tuple(self.overlap.shape)}"
            )

        n_primitive = self.d.shape[1]
        if self.overlap.shape[0] != n_primitive:
            raise ValueError(
                f"overlap dimension {self.overlap.shape[0]} does not match "
                f"d primitive dimension {n_primitive}"
            )
        self._validate_block_coverage(n_primitive)
        if not torch.allclose(
            self.overlap,
            self.overlap.conj().transpose(0, 1),
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("OVERLAP_S is not Hermitian (overlap field)")

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
            raise ValueError(
                "provenance cell_bohr components must be finite numbers"
            )
        a, b, c, d, e, f, g, h, i = (float(value) for value in cell)
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (
            d * h - e * g
        )
        if not math.isfinite(determinant) or determinant == 0.0:
            raise ValueError(
                "provenance cell_bohr lattice must be nonsingular"
            )

    def _validate_block_coverage(self, n_primitive):
        expected_offset = 0
        for index, block in enumerate(self.blocks):
            if not isinstance(block, PrimitiveBlock):
                raise ValueError(
                    f"blocks[{index}] must be a PrimitiveBlock"
                )
            if block.offset != expected_offset:
                raise ValueError(
                    f"blocks[{index}].offset expected {expected_offset}, "
                    f"got {block.offset}"
                )
            expected_offset += block.n_primitive
        if expected_offset != n_primitive:
            raise ValueError(
                f"blocks cover {expected_offset} primitives, expected {n_primitive}"
            )
