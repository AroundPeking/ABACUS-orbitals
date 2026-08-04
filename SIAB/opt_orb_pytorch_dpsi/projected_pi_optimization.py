from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch

from projected_pi import NormalizedPhysicalFamilyProjectedPi


_PHYSICAL_FAMILIES = ("H", "H2")


@dataclass(frozen=True)
class ProjectedPiOptimizationResult:
    loss: torch.Tensor
    max_condition: float
    frequency_ha: torch.Tensor
    frequency_loss: torch.Tensor
    family_results: dict
    sensitivity_alpha: float | None = None
    family_power: int | None = None

    @property
    def lowest_frequency_ha(self):
        return self.frequency_ha[0]

    @property
    def lowest_frequency_loss(self):
        return self.frequency_loss[0]


class NormalizedPhysicalFamilyProjectedPiOptimization:
    def __init__(
        self,
        *named_pairs,
        relative_rank_tolerance=1.0e-12,
        condition_limit=1.0e12,
        sensitivity_alpha=None,
        family_power=None,
    ):
        items = _normalize_named_pairs(named_pairs)
        names = tuple(name for name, _ in items)
        if (
            len(items) != len(_PHYSICAL_FAMILIES)
            or len(set(names)) != len(names)
            or set(names) != set(_PHYSICAL_FAMILIES)
        ):
            raise ValueError(
                "projected-Pi optimization requires exactly one H and one H2 pair"
            )
        pair_by_name = dict(items)
        ordered = tuple(
            (name, pair_by_name[name]) for name in _PHYSICAL_FAMILIES
        )
        self._condition_limit = _positive_condition_limit(condition_limit)
        if sensitivity_alpha is None:
            if family_power is not None:
                raise ValueError(
                    "family_power requires an RPA-sensitive projected-Pi objective"
                )
            self._family_power = None
        else:
            self._family_power = _fourth_order_family_power(family_power)
        if sensitivity_alpha is None:
            self._family = NormalizedPhysicalFamilyProjectedPi(
                ordered,
                relative_rank_tolerance=relative_rank_tolerance,
                condition_limit=self._condition_limit,
            )
        else:
            self._family = NormalizedPhysicalFamilyProjectedPi(
                ordered,
                relative_rank_tolerance=relative_rank_tolerance,
                condition_limit=self._condition_limit,
                sensitivity_alpha=sensitivity_alpha,
            )

    def evaluate(self, coefficients):
        family = self._family.evaluate(coefficients)
        if not bool(torch.isfinite(family.loss)):
            raise RuntimeError("projected-Pi optimization loss must be finite")
        if (
            not math.isfinite(family.max_candidate_condition)
            or family.max_candidate_condition > self._condition_limit
        ):
            raise RuntimeError(
                "projected-Pi candidate overlap condition number exceeds limit"
            )

        h = family.results["H"]
        h2 = family.results["H2"]
        if not torch.equal(h.frequency_ha, h2.frequency_ha):
            raise ValueError("H and H2 projected-Pi frequency grids differ")
        if not torch.equal(h.frequency_weight, h2.frequency_weight):
            raise ValueError("H and H2 projected-Pi frequency weights differ")
        if self._family_power is None:
            loss = family.loss
            sensitivity_alpha = None
        else:
            family_losses = torch.stack(
                tuple(family.results[name].loss for name in _PHYSICAL_FAMILIES)
            )
            loss = torch.linalg.vector_norm(family_losses, ord=4)
            sensitivity_alpha = h.sensitivity_alpha
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("projected-Pi optimization loss must be finite")
        frequency_loss = (h.frequency_loss + h2.frequency_loss) / 2.0
        if not bool(torch.all(torch.isfinite(frequency_loss))):
            raise RuntimeError(
                "projected-Pi optimization frequency loss must be finite"
            )
        return ProjectedPiOptimizationResult(
            loss=loss,
            max_condition=family.max_candidate_condition,
            frequency_ha=h.frequency_ha,
            frequency_loss=frequency_loss,
            family_results=family.results,
            sensitivity_alpha=sensitivity_alpha,
            family_power=self._family_power,
        )


def _normalize_named_pairs(named_pairs):
    if len(named_pairs) == 1 and isinstance(named_pairs[0], Mapping):
        items = tuple(named_pairs[0].items())
    elif len(named_pairs) == 1 and not _is_named_pair(named_pairs[0]):
        try:
            items = tuple(named_pairs[0])
        except TypeError as exc:
            raise ValueError(
                "projected-Pi families must be (name, pair) values"
            ) from exc
    else:
        items = tuple(named_pairs)
    if any(not _is_named_pair(item) for item in items):
        raise ValueError("projected-Pi families must be (name, pair) values")
    return items


def _is_named_pair(value):
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and isinstance(value[0], str)
    )


def _positive_condition_limit(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("condition_limit must be finite and at least 1") from exc
    if not math.isfinite(value) or value < 1.0:
        raise ValueError("condition_limit must be finite and at least 1")
    return value


def _fourth_order_family_power(value):
    if isinstance(value, bool) or not isinstance(value, int) or value != 4:
        raise ValueError("family_power must be exactly 4")
    return value
