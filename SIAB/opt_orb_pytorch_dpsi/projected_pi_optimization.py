from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch

from projected_pi import NormalizedPhysicalFamilyProjectedPi


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
            len(items) != 2
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError(
                "projected-Pi optimization requires exactly two unique family names"
            )
        self._family_names = names
        ordered = items
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

        first = family.results[self._family_names[0]]
        second = family.results[self._family_names[1]]
        if not torch.equal(first.frequency_ha, second.frequency_ha):
            raise ValueError("projected-Pi family frequency grids differ")
        if not torch.equal(first.frequency_weight, second.frequency_weight):
            raise ValueError("projected-Pi family frequency weights differ")
        if self._family_power is None:
            loss = family.loss
            sensitivity_alpha = None
        else:
            family_losses = torch.stack(
                tuple(family.results[name].loss for name in self._family_names)
            )
            loss = torch.linalg.vector_norm(family_losses, ord=4)
            sensitivity_alpha = first.sensitivity_alpha
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("projected-Pi optimization loss must be finite")
        frequency_loss = (first.frequency_loss + second.frequency_loss) / 2.0
        if not bool(torch.all(torch.isfinite(frequency_loss))):
            raise RuntimeError(
                "projected-Pi optimization frequency loss must be finite"
            )
        return ProjectedPiOptimizationResult(
            loss=loss,
            max_condition=family.max_candidate_condition,
            frequency_ha=first.frequency_ha,
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
