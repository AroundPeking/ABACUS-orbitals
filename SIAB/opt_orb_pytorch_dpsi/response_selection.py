from dataclasses import dataclass
import math

from sternheimer_spillage import evaluate_spillage_for_columns


_TARGET_ROLES = frozenset({"physical", "ghost"})


@dataclass(frozen=True)
class ResponseTargetFamily:
    name: str
    data: tuple
    role: str
    real_atom_index: int = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("target family name must be nonempty")
        data = tuple(self.data)
        if not data:
            raise ValueError("target family data must be nonempty")
        object.__setattr__(self, "data", data)
        if self.role not in _TARGET_ROLES:
            raise ValueError("target family role must be physical or ghost")
        if self.role == "physical":
            if self.real_atom_index is not None:
                raise ValueError(
                    "physical target family must not set real_atom_index"
                )
            return
        if type(self.real_atom_index) is not int or self.real_atom_index < 0:
            raise ValueError(
                "ghost target family requires a nonnegative real_atom_index"
            )
        for value in data:
            if not any(
                block.atom_index == self.real_atom_index
                for block in value.blocks
            ):
                raise ValueError("real_atom_index maps to no primitive blocks")


def _sum_residual(data_items, coefficients, include):
    total = 0.0
    for data in data_items:
        residual = float(
            evaluate_spillage_for_columns(
                data, coefficients, include
            ).weighted_residual.item()
        )
        if not math.isfinite(residual) or residual < 0.0:
            raise RuntimeError(
                "target-family residual must be finite and nonnegative"
            )
        total += residual
    return total


def _positive_dzp_residual(value):
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("fixed-DZP family residual must be positive")
    return value


def normalized_family_loss(family, current, fixed_dzp):
    if not isinstance(family, ResponseTargetFamily):
        raise TypeError("family must be a ResponseTargetFamily")
    if family.role != "physical":
        raise ValueError("normalized physical loss requires a physical family")
    include_all = lambda label: True
    numerator = _sum_residual(family.data, current, include_all)
    denominator = _positive_dzp_residual(
        _sum_residual(family.data, fixed_dzp, include_all)
    )
    return numerator / denominator


def borrowing_gap(family, current, fixed_dzp):
    if not isinstance(family, ResponseTargetFamily):
        raise TypeError("family must be a ResponseTargetFamily")
    if family.role != "ghost":
        raise ValueError("borrowing gap requires a ghost family")

    own = lambda label: label.atom_index == family.real_atom_index
    all_centers = lambda label: True
    denominator = _positive_dzp_residual(
        _sum_residual(family.data, fixed_dzp, own)
    )
    own_residual = _sum_residual(family.data, current, own)
    all_residual = _sum_residual(family.data, current, all_centers)
    difference = own_residual - all_residual
    tolerance = 1.0e-10 * max(
        abs(own_residual), abs(all_residual), denominator, 1.0
    )
    if difference < -tolerance:
        raise RuntimeError(
            "all-center residual exceeds own-center residual; "
            "check fragment/ghost target construction"
        )
    return max(difference, 0.0) / denominator
