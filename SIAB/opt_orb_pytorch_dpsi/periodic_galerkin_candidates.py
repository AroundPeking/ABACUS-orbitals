"""Reusable candidate directions from named periodic Galerkin loss families."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch

from periodic_galerkin_fit import (
    _assemble,
    _global_pi_loss,
    _prepare_block_contraction_caches,
    _retract_variables,
    _validate_inputs,
)
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


@dataclass(frozen=True)
class PeriodicGalerkinFamilyGradientResult:
    coefficients: dict
    family_order: tuple
    family_losses: dict
    gradients: dict
    normalized_gradients: dict
    gradient_norms: dict
    gradient_cosines: dict
    minimum_occupied_capture: float
    maximum_overlap_condition: float
    prepared_datasets: tuple | None = None
    dataset_families: tuple | None = None
    additional_family_evaluators: dict | None = None
    occupied_capture_tolerance: float = 1.0e-6


@dataclass(frozen=True)
class PeriodicGalerkinParetoCandidate:
    weight: float
    trust_radius: float
    coefficients: dict
    coefficients_sha256: str


@dataclass(frozen=True)
class PeriodicGalerkinSingleFamilyCandidate:
    family: str
    trust_radius: float
    coefficients: dict
    coefficients_sha256: str


def _clone_blocks(values):
    return {
        element: [channel.detach().clone() for channel in channels]
        for element, channels in values.items()
    }


def _validate_matching_blocks(coefficients, gradient):
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError("coefficients must be a nonempty dictionary")
    if not isinstance(gradient, dict) or set(gradient) != set(coefficients):
        raise ValueError("gradient elements must match coefficients")
    for element, channels in coefficients.items():
        if len(gradient[element]) != len(channels):
            raise ValueError("gradient channels must match coefficients")
        for coefficient, value in zip(channels, gradient[element]):
            if (
                not isinstance(coefficient, torch.Tensor)
                or not isinstance(value, torch.Tensor)
                or coefficient.shape != value.shape
                or coefficient.dtype != torch.float64
                or value.dtype != torch.float64
                or coefficient.device.type != "cpu"
                or value.device.type != "cpu"
                or not bool(torch.isfinite(coefficient).all())
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    "coefficient and gradient blocks must be matching finite CPU float64 matrices"
                )


def project_fixed_prefix_tangent(coefficients, *, fixed_nu, gradient):
    """Project a coefficient gradient onto the fixed-prefix Stiefel tangent."""
    _validate_matching_blocks(coefficients, gradient)
    if not isinstance(fixed_nu, dict) or set(fixed_nu) != set(coefficients):
        raise ValueError("fixed_nu elements must match coefficients")
    projected = {}
    for element, channels in coefficients.items():
        try:
            counts = tuple(fixed_nu[element])
        except TypeError as error:
            raise ValueError("fixed_nu channels must be a sequence") from error
        if len(counts) != len(channels):
            raise ValueError("fixed_nu must define every angular channel")
        projected[element] = []
        for coefficient, value, count in zip(
            channels,
            gradient[element],
            counts,
        ):
            if type(count) is not int or count < 0 or count > coefficient.shape[1]:
                raise ValueError("fixed_nu exceeds a coefficient channel")
            fixed = coefficient[:, :count]
            variable = coefficient[:, count:]
            variable_gradient = value[:, count:].clone()
            if variable.shape[1]:
                if fixed.shape[1]:
                    gram = fixed.transpose(0, 1).matmul(fixed)
                    projection = torch.linalg.solve(
                        gram,
                        fixed.transpose(0, 1).matmul(variable_gradient),
                    )
                    variable_gradient = variable_gradient - fixed.matmul(projection)
                symmetric = 0.5 * (
                    variable.transpose(0, 1).matmul(variable_gradient)
                    + variable_gradient.transpose(0, 1).matmul(variable)
                )
                variable_gradient = variable_gradient - variable.matmul(symmetric)
            projected[element].append(
                torch.cat(
                    (
                        torch.zeros_like(fixed),
                        variable_gradient,
                    ),
                    dim=1,
                )
            )
    return projected


def _gradient_inner(left, right):
    _validate_matching_blocks(left, right)
    return sum(
        torch.sum(left[element][index] * right[element][index])
        for element in left
        for index in range(len(left[element]))
    )


def normalize_gradient(gradient, *, family):
    if not isinstance(family, str) or not family.strip():
        raise ValueError("family must be nonempty")
    _validate_matching_blocks(gradient, gradient)
    norm_squared = _gradient_inner(gradient, gradient)
    norm = math.sqrt(float(norm_squared.detach()))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"{family} gradient norm is zero or non-finite")
    return (
        {
            element: [channel / norm for channel in channels]
            for element, channels in gradient.items()
        },
        norm,
    )


def _retract_candidate(coefficients, fixed_nu, direction, trust_radius):
    candidate = {}
    for element, channels in coefficients.items():
        candidate[element] = []
        for channel, step, fixed_count in zip(
            channels,
            direction[element],
            fixed_nu[element],
        ):
            fixed = channel[:, :fixed_count].clone()
            variable = channel[:, fixed_count:] + trust_radius * step[:, fixed_count:]
            if variable.shape[1]:
                if fixed.shape[1]:
                    gram = fixed.transpose(0, 1).matmul(fixed)
                    projection = torch.linalg.solve(
                        gram,
                        fixed.transpose(0, 1).matmul(variable),
                    )
                    variable = variable - fixed.matmul(projection)
                singular_values = torch.linalg.svdvals(variable)
                if (
                    singular_values.shape[0] != variable.shape[1]
                    or float(torch.min(singular_values))
                    <= 1.0e-12 * float(torch.max(singular_values))
                ):
                    raise RuntimeError("Pareto candidate radial channel is rank deficient")
                frame, triangular = torch.linalg.qr(variable, mode="reduced")
                diagonal = torch.diagonal(triangular)
                sign = torch.where(
                    diagonal < 0.0,
                    -torch.ones_like(diagonal),
                    torch.ones_like(diagonal),
                )
                variable = frame * sign
            candidate[element].append(torch.cat((fixed, variable), dim=1))
    return candidate


def coefficient_sha256(coefficients):
    _validate_matching_blocks(coefficients, coefficients)
    digest = hashlib.sha256()
    digest.update(b"SIAB_PERIODIC_GALERKIN_COEFFICIENTS_V1\0")
    for element in sorted(coefficients):
        digest.update(element.encode("ascii") + b"\0")
        for angular_momentum, channel in enumerate(coefficients[element]):
            digest.update(
                f"{angular_momentum}:{channel.shape[0]}:{channel.shape[1]}\0".encode(
                    "ascii"
                )
            )
            digest.update(channel.detach().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_single_family_candidate(
    result,
    *,
    fixed_nu,
    family,
    trust_radius=0.01,
):
    """Build one deterministic trust-region descent for a named family."""
    if not isinstance(result, PeriodicGalerkinFamilyGradientResult):
        raise ValueError("result must be a family-gradient result")
    if not isinstance(family, str) or family not in result.family_order:
        raise ValueError("family must name one evaluated family")
    if (
        not isinstance(trust_radius, (int, float))
        or isinstance(trust_radius, bool)
        or not math.isfinite(trust_radius)
        or trust_radius <= 0.0
    ):
        raise ValueError("trust_radius must be finite and positive")
    trust_radius = float(trust_radius)
    direction = {
        element: [-channel for channel in channels]
        for element, channels in result.normalized_gradients[family].items()
    }
    coefficients = _retract_candidate(
        result.coefficients,
        fixed_nu,
        direction,
        trust_radius,
    )
    return PeriodicGalerkinSingleFamilyCandidate(
        family=family,
        trust_radius=trust_radius,
        coefficients=_clone_blocks(coefficients),
        coefficients_sha256=coefficient_sha256(coefficients),
    )


def build_pareto_candidate_bank(
    result,
    *,
    fixed_nu,
    family_pair,
    weights=(0.25, 0.5, 0.75),
    trust_radius=0.01,
):
    """Build a deterministic two-family Pareto bank at one trust radius."""
    if not isinstance(result, PeriodicGalerkinFamilyGradientResult):
        raise ValueError("result must be a family-gradient result")
    try:
        left, right = tuple(family_pair)
    except (TypeError, ValueError) as error:
        raise ValueError("family_pair must contain exactly two families") from error
    if (
        left == right
        or left not in result.family_order
        or right not in result.family_order
    ):
        raise ValueError("family_pair must name two distinct evaluated families")
    try:
        weights = tuple(float(weight) for weight in weights)
    except (TypeError, ValueError) as error:
        raise ValueError("weights must be finite values between zero and one") from error
    if (
        not weights
        or weights != tuple(sorted(set(weights)))
        or any(
            not math.isfinite(weight) or weight <= 0.0 or weight >= 1.0
            for weight in weights
        )
    ):
        raise ValueError("weights must be unique increasing values between zero and one")
    if (
        not isinstance(trust_radius, (int, float))
        or isinstance(trust_radius, bool)
        or not math.isfinite(trust_radius)
        or trust_radius <= 0.0
    ):
        raise ValueError("trust_radius must be finite and positive")
    trust_radius = float(trust_radius)

    candidates = []
    for weight in weights:
        combined = {
            element: [
                weight * left_channel + (1.0 - weight) * right_channel
                for left_channel, right_channel in zip(
                    result.normalized_gradients[left][element],
                    result.normalized_gradients[right][element],
                )
            ]
            for element in result.coefficients
        }
        descent, _ = normalize_gradient(
            combined,
            family=f"{left}:{right}:{weight:.6f}",
        )
        descent = {
            element: [-channel for channel in channels]
            for element, channels in descent.items()
        }
        coefficients = _retract_candidate(
            result.coefficients,
            fixed_nu,
            descent,
            trust_radius,
        )
        candidates.append(
            PeriodicGalerkinParetoCandidate(
                weight=weight,
                trust_radius=trust_radius,
                coefficients=_clone_blocks(coefficients),
                coefficients_sha256=coefficient_sha256(coefficients),
            )
        )
    return tuple(candidates)


def assess_family_tradeoff(
    baseline,
    candidate,
    *,
    maximum_relative_degradation,
):
    if (
        not isinstance(baseline, dict)
        or not baseline
        or not isinstance(candidate, dict)
        or set(candidate) != set(baseline)
    ):
        raise ValueError("candidate family losses must match a nonempty baseline")
    if (
        not isinstance(maximum_relative_degradation, (int, float))
        or isinstance(maximum_relative_degradation, bool)
        or not math.isfinite(maximum_relative_degradation)
        or maximum_relative_degradation < 0.0
    ):
        raise ValueError("maximum_relative_degradation must be finite and nonnegative")
    maximum_relative_degradation = float(maximum_relative_degradation)
    ratios = {}
    improved = []
    degraded = []
    for family in baseline:
        base = float(baseline[family])
        value = float(candidate[family])
        if (
            not math.isfinite(base)
            or not math.isfinite(value)
            or base <= 0.0
            or value < 0.0
        ):
            raise ValueError("family losses must be finite with positive baselines")
        ratio = value / base
        ratios[family] = ratio
        if value < base:
            improved.append(family)
        if ratio > 1.0 + maximum_relative_degradation:
            degraded.append(family)
    reasons = []
    if not improved:
        reasons.append("no_family_improved")
    if degraded:
        reasons.append("family_degradation_exceeds_limit")
    return {
        "gate": "pass" if not reasons else "fail",
        "relative_family_losses": ratios,
        "improved_families": improved,
        "degraded_families": degraded,
        "failure_reasons": reasons,
    }


def evaluate_candidate_family_losses(result, coefficients):
    """Evaluate a trial using the prepared family problem from a gradient audit."""
    if not isinstance(result, PeriodicGalerkinFamilyGradientResult):
        raise ValueError("result must be a family-gradient result")
    if result.prepared_datasets is None:
        raise ValueError("family-gradient result has no prepared evaluation problem")
    _validate_matching_blocks(result.coefficients, coefficients)
    loss_options = {}
    if result.dataset_families is not None:
        loss_options["dataset_families"] = result.dataset_families
    if result.additional_family_evaluators:
        loss_options["additional_family_evaluators"] = (
            result.additional_family_evaluators
        )
    loss, capture, condition, family_losses = _global_pi_loss(
        result.prepared_datasets,
        coefficients,
        occupied_capture_tolerance=result.occupied_capture_tolerance,
        **loss_options,
    )
    return {
        "loss": float(loss.detach()),
        "family_losses": {
            family: float(value.detach()) for family, value in family_losses.items()
        },
        "minimum_occupied_capture": capture,
        "maximum_overlap_condition": condition,
    }


def evaluate_family_gradients(
    datasets,
    initial,
    *,
    fixed_nu,
    dataset_families=None,
    additional_family_evaluators=None,
    occupied_capture_tolerance=1.0e-6,
    block_cache_workers=1,
):
    """Evaluate one tangent gradient for each independently normalized family."""
    fixed, variable, parameters = _validate_inputs(datasets, initial, fixed_nu)
    datasets = tuple(
        prepare_periodic_occupied_reference(dataset) for dataset in datasets
    )
    _retract_variables(fixed, variable)
    coefficients = _assemble(fixed, variable)
    datasets = _prepare_block_contraction_caches(
        datasets,
        coefficients,
        block_cache_workers,
    )
    loss_options = {}
    if dataset_families is not None:
        loss_options["dataset_families"] = tuple(dataset_families)
    if additional_family_evaluators:
        loss_options["additional_family_evaluators"] = dict(
            additional_family_evaluators
        )
    _, minimum_capture, maximum_condition, family_losses = _global_pi_loss(
        datasets,
        coefficients,
        occupied_capture_tolerance=occupied_capture_tolerance,
        **loss_options,
    )
    family_order = tuple(family_losses)
    if not family_order:
        raise RuntimeError("Galerkin candidate generation has no loss family")

    gradients = {}
    normalized = {}
    norms = {}
    for position, family in enumerate(family_order):
        parameter_gradients = torch.autograd.grad(
            family_losses[family],
            parameters,
            retain_graph=position + 1 < len(family_order),
        )
        iterator = iter(parameter_gradients)
        raw = {}
        for element, channels in coefficients.items():
            raw[element] = []
            for channel, fixed_count in zip(channels, fixed_nu[element]):
                variable_count = channel.shape[1] - fixed_count
                variable_gradient = (
                    next(iterator)
                    if variable_count
                    else torch.empty(
                        (channel.shape[0], 0),
                        dtype=torch.float64,
                    )
                )
                raw[element].append(
                    torch.cat(
                        (
                            torch.zeros(
                                (channel.shape[0], fixed_count),
                                dtype=torch.float64,
                            ),
                            variable_gradient,
                        ),
                        dim=1,
                    )
                )
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("family gradient count does not match variables")
        tangent = project_fixed_prefix_tangent(
            coefficients,
            fixed_nu=fixed_nu,
            gradient=raw,
        )
        unit, norm = normalize_gradient(tangent, family=family)
        gradients[family] = tangent
        normalized[family] = unit
        norms[family] = norm

    cosines = {}
    for left_index, left in enumerate(family_order):
        for right in family_order[left_index + 1 :]:
            cosines[f"{left}:{right}"] = float(
                _gradient_inner(normalized[left], normalized[right]).detach()
            )
    return PeriodicGalerkinFamilyGradientResult(
        coefficients=_clone_blocks(coefficients),
        family_order=family_order,
        family_losses={
            family: float(value.detach()) for family, value in family_losses.items()
        },
        gradients={
            family: _clone_blocks(value) for family, value in gradients.items()
        },
        normalized_gradients={
            family: _clone_blocks(value) for family, value in normalized.items()
        },
        gradient_norms=norms,
        gradient_cosines=cosines,
        minimum_occupied_capture=minimum_capture,
        maximum_overlap_condition=maximum_condition,
        prepared_datasets=datasets,
        dataset_families=(
            tuple(dataset_families) if dataset_families is not None else None
        ),
        additional_family_evaluators=(
            dict(additional_family_evaluators)
            if additional_family_evaluators
            else None
        ),
        occupied_capture_tolerance=float(occupied_capture_tolerance),
    )
