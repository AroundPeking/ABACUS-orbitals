import math
from collections.abc import Mapping

import torch

from response_selection import ResponseTargetFamily
from sternheimer_spillage import (
    OrbitalColumn,
    SternheimerLossResult,
    SternheimerSpillage,
    assemble_orbital_coefficients,
    evaluate_spillage_for_columns,
)


def _fixed_columns(data, coefficients, fixed_specs):
    _, labels = assemble_orbital_coefficients(data, coefficients)
    selected = []
    for spec in fixed_specs:
        if not isinstance(spec, Mapping) or set(spec) != {
            "element",
            "l",
            "zeta",
        }:
            raise ValueError("fixed orbital spec requires element, l, and zeta")
        matches = tuple(
            label
            for label in labels
            if label.element == spec["element"]
            and label.l == spec["l"]
            and label.zeta == spec["zeta"]
        )
        if not matches:
            raise ValueError(f"fixed orbital spec {dict(spec)!r} maps to no columns")
        selected.extend(matches)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("fixed orbital specs must map to unique columns")
    if any(not isinstance(value, OrbitalColumn) for value in selected):
        raise TypeError("fixed orbital expansion produced an invalid column")
    selected_set = set(selected)
    return tuple(label for label in labels if label in selected_set)


class NormalizedPhysicalFamilySpillage:
    """Sum physical family residuals normalized to their fixed-DZP values."""

    def __init__(
        self,
        families,
        c0,
        fixed_dzp,
        fixed_specs,
        condition_limit=1.0e12,
    ):
        families = tuple(families)
        fixed_specs = tuple(fixed_specs)
        if not families or any(
            not isinstance(value, ResponseTargetFamily)
            or value.role != "physical"
            for value in families
        ):
            raise ValueError("optimizer loss requires physical target families")

        self._families = []
        for family in families:
            evaluators = []
            denominator = None
            for data in family.data:
                fixed_columns = _fixed_columns(data, c0, fixed_specs)
                evaluator = SternheimerSpillage(
                    data,
                    c0,
                    fixed_columns,
                    condition_limit=condition_limit,
                )
                evaluators.append(evaluator)
                fixed_result = evaluate_spillage_for_columns(
                    data,
                    fixed_dzp,
                    include=lambda label, selected=set(fixed_columns): (
                        label in selected
                    ),
                    condition_limit=condition_limit,
                )
                denominator = (
                    fixed_result.weighted_residual.detach().clone()
                    if denominator is None
                    else denominator
                    + fixed_result.weighted_residual.detach().clone()
                )
            denominator_value = float(denominator.item())
            if (
                not math.isfinite(denominator_value)
                or denominator_value <= 0.0
            ):
                raise RuntimeError("fixed-DZP family residual must be positive")
            self._families.append((family.name, tuple(evaluators), denominator))

    def evaluate(self, coefficients):
        total = None
        max_condition = 0.0
        for _, evaluators, denominator in self._families:
            residual = None
            for evaluator in evaluators:
                value = evaluator.evaluate(coefficients)
                residual = (
                    value.weighted_residual
                    if residual is None
                    else residual + value.weighted_residual
                )
                max_condition = max(max_condition, value.max_condition)
            family_loss = residual / denominator
            total = family_loss if total is None else total + family_loss

        if total is None or not bool(torch.isfinite(total)):
            raise RuntimeError("normalized physical-family loss must be finite")
        return SternheimerLossResult(
            loss=total,
            weighted_residual=total,
            weighted_norm=torch.ones_like(total),
            max_condition=max_condition,
        )
