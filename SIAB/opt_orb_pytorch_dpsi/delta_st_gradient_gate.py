"""Bounded one-step gradient gate for Delta-ST response compression."""

from dataclasses import dataclass
import hashlib
import math
import pathlib
from typing import Mapping, Tuple

import torch

from freeze_orbitals import validate_freeze_orbitals, zero_frozen_gradients


@dataclass(frozen=True)
class DeltaSTGradientGateResult:
    initial_result: object
    accepted_result: object
    coefficients: Mapping[str, object]
    initial_loss: float
    accepted_loss: float
    accepted_step: float
    raw_fixed_gradient_norm: float
    masked_fixed_gradient_norm: float
    variable_gradient_norm: float


@dataclass(frozen=True)
class DeltaSTOptimizationRecord:
    iteration: int
    loss: float
    accepted_step: float
    relative_loss_reduction: float
    raw_fixed_gradient_norm: float
    masked_fixed_gradient_norm: float
    variable_gradient_norm: float
    maximum_frequency_loss: float
    maximum_condition: float
    retained_rank_by_spin: Tuple[int, ...]
    dropped_rank_by_spin: Tuple[int, ...]


@dataclass(frozen=True)
class DeltaSTOptimizationResult:
    initial_result: object
    final_result: object
    coefficients: Mapping[str, object]
    initial_loss: float
    final_loss: float
    stop_reason: str
    history: Tuple[DeltaSTOptimizationRecord, ...]


def require_file_sha256(path, expected):
    """Return a file SHA256 and optionally require an exact expected value."""
    path = pathlib.Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot read file for SHA256: {path}") from exc
    actual = digest.hexdigest()
    if expected is not None:
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("expected SHA256 must contain 64 lowercase hex digits")
        if actual != expected:
            raise ValueError(
                f"file SHA256 differs from the physical protocol: {path}"
            )
    return actual


def run_delta_st_gradient_gate(
    objective,
    coefficients,
    freeze_specs,
    *,
    step_sizes=(0.02, 0.01, 0.005, 0.002, 0.001),
):
    """Accept one response-lowering step while preserving frozen columns."""
    if not hasattr(objective, "evaluate"):
        raise TypeError("objective must provide evaluate(coefficients)")
    steps = _validate_step_sizes(step_sizes)
    frozen = validate_freeze_orbitals(freeze_specs, coefficients)
    matrices = tuple(_coefficient_matrices(coefficients))
    if not matrices:
        raise ValueError("coefficients must contain at least one matrix")

    for _, _, matrix in matrices:
        matrix.grad = None
    initial_result = objective.evaluate(coefficients)
    initial_loss = _finite_scalar_loss(initial_result.loss, "initial loss")
    initial_result.loss.backward()

    raw_fixed_squared = 0.0
    variable_squared = 0.0
    for element, l, matrix in matrices:
        if matrix.grad is None or matrix.grad.shape != matrix.shape:
            raise RuntimeError(f"missing gradient for {element}/{l}")
        if not bool(torch.all(torch.isfinite(matrix.grad))):
            raise RuntimeError(f"non-finite gradient for {element}/{l}")
        for zeta in range(matrix.shape[1]):
            norm_squared = float(torch.sum(matrix.grad[:, zeta] ** 2))
            if (element, l, zeta) in frozen:
                raw_fixed_squared += norm_squared
            else:
                variable_squared += norm_squared
    variable_norm = math.sqrt(variable_squared)
    if variable_norm == 0.0:
        raise RuntimeError("variable Delta-ST gradient norm is zero")

    zero_frozen_gradients(coefficients, frozen)
    masked_fixed_squared = sum(
        float(torch.sum(coefficients[element][l].grad[:, zeta] ** 2))
        for element, l, zeta in frozen
    )
    masked_fixed_norm = math.sqrt(masked_fixed_squared)
    if masked_fixed_norm != 0.0:
        raise RuntimeError("frozen Delta-ST gradients are not exactly zero")

    initial_coefficients = _clone_coefficients(coefficients)
    gradients = {
        (element, l): matrix.grad.detach().clone()
        for element, l, matrix in matrices
    }
    for step in steps:
        trial = _clone_coefficients(initial_coefficients)
        with torch.no_grad():
            for element, l, matrix in _coefficient_matrices(trial):
                matrix.sub_(step * gradients[(element, l)])
                for frozen_element, frozen_l, zeta in frozen:
                    if frozen_element == element and frozen_l == l:
                        matrix[:, zeta].copy_(
                            initial_coefficients[element][l][:, zeta]
                        )
        _require_frozen_equal(trial, initial_coefficients, frozen)
        accepted_result = objective.evaluate(trial)
        accepted_loss = _finite_scalar_loss(
            accepted_result.loss, f"trial loss at step {step}"
        )
        if accepted_loss < initial_loss:
            return DeltaSTGradientGateResult(
                initial_result=initial_result,
                accepted_result=accepted_result,
                coefficients=trial,
                initial_loss=initial_loss,
                accepted_loss=accepted_loss,
                accepted_step=step,
                raw_fixed_gradient_norm=math.sqrt(raw_fixed_squared),
                masked_fixed_gradient_norm=masked_fixed_norm,
                variable_gradient_norm=variable_norm,
            )
    raise RuntimeError("no bounded gradient step lowers the Delta-ST response loss")


def run_delta_st_response_optimization(
    objective,
    coefficients,
    freeze_specs,
    *,
    max_steps=100,
    initial_step=0.2,
    maximum_step=2.0,
    backtracking_factor=0.5,
    step_growth=1.5,
    armijo_factor=1.0e-4,
    maximum_line_search_steps=16,
    gradient_tolerance=1.0e-8,
    relative_loss_tolerance=1.0e-9,
    relative_loss_patience=5,
):
    """Minimize the full-frequency response loss with frozen AO columns."""
    if not hasattr(objective, "evaluate"):
        raise TypeError("objective must provide evaluate(coefficients)")
    max_steps = _positive_integer("max_steps", max_steps)
    maximum_line_search_steps = _positive_integer(
        "maximum_line_search_steps", maximum_line_search_steps
    )
    relative_loss_patience = _positive_integer(
        "relative_loss_patience", relative_loss_patience
    )
    initial_step = _finite_positive("initial_step", initial_step)
    maximum_step = _finite_positive("maximum_step", maximum_step)
    if maximum_step < initial_step:
        raise ValueError("maximum_step must be at least initial_step")
    backtracking_factor = _finite_open_unit_interval(
        "backtracking_factor", backtracking_factor
    )
    armijo_factor = _finite_open_unit_interval("armijo_factor", armijo_factor)
    step_growth = _finite_positive("step_growth", step_growth)
    if step_growth < 1.0:
        raise ValueError("step_growth must be at least one")
    gradient_tolerance = _finite_nonnegative(
        "gradient_tolerance", gradient_tolerance
    )
    relative_loss_tolerance = _finite_nonnegative(
        "relative_loss_tolerance", relative_loss_tolerance
    )

    frozen = validate_freeze_orbitals(freeze_specs, coefficients)
    fixed_reference = _clone_coefficients(coefficients)
    current = _clone_coefficients(coefficients, requires_grad=True)
    if not tuple(_coefficient_matrices(current)):
        raise ValueError("coefficients must contain at least one matrix")

    history = []
    initial_result = None
    initial_loss = None
    previous_loss = None
    accepted_step = 0.0
    small_reduction_count = 0
    trial_step = initial_step
    stop_reason = "max_steps"

    for iteration in range(max_steps + 1):
        matrices = tuple(_coefficient_matrices(current))
        for _, _, matrix in matrices:
            matrix.grad = None
        result = objective.evaluate(current)
        loss = _finite_scalar_loss(result.loss, f"loss at iteration {iteration}")
        result.loss.backward()
        raw_fixed_norm, variable_norm = _gradient_norms(matrices, frozen)
        if variable_norm < 0.0:
            raise RuntimeError("variable Delta-ST gradient norm is invalid")
        zero_frozen_gradients(current, frozen)
        masked_fixed_norm = _fixed_gradient_norm(current, frozen)
        if masked_fixed_norm != 0.0:
            raise RuntimeError("frozen Delta-ST gradients are not exactly zero")

        if initial_result is None:
            initial_result = result
            initial_loss = loss
        relative_reduction = (
            0.0
            if previous_loss is None
            else (previous_loss - loss) / max(previous_loss, 1.0e-300)
        )
        history.append(
            _optimization_record(
                iteration,
                result,
                loss,
                accepted_step,
                relative_reduction,
                raw_fixed_norm,
                masked_fixed_norm,
                variable_norm,
            )
        )

        if variable_norm <= gradient_tolerance:
            stop_reason = "gradient_tolerance"
            break
        if iteration == max_steps:
            break
        if previous_loss is not None:
            if relative_reduction <= relative_loss_tolerance:
                small_reduction_count += 1
            else:
                small_reduction_count = 0
            if small_reduction_count >= relative_loss_patience:
                stop_reason = "relative_loss_tolerance"
                break

        gradients = {
            (element, l): matrix.grad.detach().clone()
            for element, l, matrix in matrices
        }
        gradient_squared = variable_norm * variable_norm
        accepted = None
        candidate_step = trial_step
        for _ in range(maximum_line_search_steps):
            trial = _clone_coefficients(current)
            with torch.no_grad():
                for element, l, matrix in _coefficient_matrices(trial):
                    matrix.sub_(candidate_step * gradients[(element, l)])
                _restore_frozen_columns(trial, fixed_reference, frozen)
            _require_frozen_equal(trial, fixed_reference, frozen)
            trial_result = objective.evaluate(trial)
            candidate_loss = _finite_scalar_loss(
                trial_result.loss,
                f"trial loss at iteration {iteration + 1}",
            )
            armijo_bound = loss - armijo_factor * candidate_step * gradient_squared
            if candidate_loss <= armijo_bound and candidate_loss < loss:
                accepted = (trial, candidate_loss, candidate_step)
                break
            candidate_step *= backtracking_factor
        if accepted is None:
            stop_reason = "line_search_failed"
            break

        trial, _, accepted_step = accepted
        previous_loss = loss
        current = _clone_coefficients(trial, requires_grad=True)
        trial_step = min(maximum_step, accepted_step * step_growth)

    final_coefficients = _clone_coefficients(current)
    _require_frozen_equal(final_coefficients, fixed_reference, frozen)
    return DeltaSTOptimizationResult(
        initial_result=initial_result,
        final_result=result,
        coefficients=final_coefficients,
        initial_loss=initial_loss,
        final_loss=history[-1].loss,
        stop_reason=stop_reason,
        history=tuple(history),
    )


def _coefficient_matrices(coefficients):
    if not isinstance(coefficients, Mapping):
        raise TypeError("coefficients must be a mapping")
    for element, by_l in coefficients.items():
        channels = by_l.items() if isinstance(by_l, Mapping) else enumerate(by_l)
        for l, matrix in channels:
            if not isinstance(matrix, torch.Tensor):
                raise TypeError(f"coefficient {element}/{l} must be a tensor")
            if matrix.dtype != torch.float64 or matrix.ndim != 2:
                raise ValueError(
                    f"coefficient {element}/{l} must be a float64 matrix"
                )
            yield element, int(l), matrix


def _clone_coefficients(coefficients, *, requires_grad=False):
    result = {}
    for element, by_l in coefficients.items():
        if isinstance(by_l, Mapping):
            result[element] = {
                l: matrix.detach().clone().requires_grad_(requires_grad)
                for l, matrix in by_l.items()
            }
        else:
            result[element] = [
                matrix.detach().clone().requires_grad_(requires_grad)
                for matrix in by_l
            ]
    return result


def _gradient_norms(matrices, frozen):
    raw_fixed_squared = 0.0
    variable_squared = 0.0
    for element, l, matrix in matrices:
        if matrix.grad is None or matrix.grad.shape != matrix.shape:
            raise RuntimeError(f"missing gradient for {element}/{l}")
        if not bool(torch.all(torch.isfinite(matrix.grad))):
            raise RuntimeError(f"non-finite gradient for {element}/{l}")
        for zeta in range(matrix.shape[1]):
            norm_squared = float(torch.sum(matrix.grad[:, zeta] ** 2))
            if (element, l, zeta) in frozen:
                raw_fixed_squared += norm_squared
            else:
                variable_squared += norm_squared
    return math.sqrt(raw_fixed_squared), math.sqrt(variable_squared)


def _fixed_gradient_norm(coefficients, frozen):
    return math.sqrt(
        sum(
            float(torch.sum(coefficients[element][l].grad[:, zeta] ** 2))
            for element, l, zeta in frozen
        )
    )


def _restore_frozen_columns(candidate, reference, frozen):
    for element, l, zeta in frozen:
        candidate[element][l][:, zeta].copy_(reference[element][l][:, zeta])


def _optimization_record(
    iteration,
    result,
    loss,
    accepted_step,
    relative_loss_reduction,
    raw_fixed_gradient_norm,
    masked_fixed_gradient_norm,
    variable_gradient_norm,
):
    try:
        families = tuple(result.family_results.values())
    except AttributeError as exc:
        raise RuntimeError("optimization result must provide family_results") from exc
    if len(families) != 1:
        raise RuntimeError("response compression optimization requires one family")
    family = families[0]
    maximum_frequency_loss = float(torch.max(result.frequency_loss))
    maximum_condition = float(result.max_condition)
    if (
        not math.isfinite(maximum_frequency_loss)
        or maximum_frequency_loss < 0.0
        or not math.isfinite(maximum_condition)
        or maximum_condition < 1.0
    ):
        raise RuntimeError("invalid response-compression diagnostics")
    return DeltaSTOptimizationRecord(
        iteration=iteration,
        loss=loss,
        accepted_step=accepted_step,
        relative_loss_reduction=relative_loss_reduction,
        raw_fixed_gradient_norm=raw_fixed_gradient_norm,
        masked_fixed_gradient_norm=masked_fixed_gradient_norm,
        variable_gradient_norm=variable_gradient_norm,
        maximum_frequency_loss=maximum_frequency_loss,
        maximum_condition=maximum_condition,
        retained_rank_by_spin=tuple(family.retained_rank_by_spin),
        dropped_rank_by_spin=tuple(family.dropped_rank_by_spin),
    )


def _require_frozen_equal(candidate, reference, frozen):
    for element, l, zeta in frozen:
        if not torch.equal(
            candidate[element][l][:, zeta], reference[element][l][:, zeta]
        ):
            raise RuntimeError(f"frozen coefficient changed: {element}/{l}/{zeta + 1}")


def _validate_step_sizes(values):
    try:
        steps = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("step_sizes must contain positive finite values") from exc
    if not steps or any(not math.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError("step_sizes must contain positive finite values")
    return steps


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _finite_positive(name, value):
    value = _finite_nonnegative(name, value)
    if value == 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_open_unit_interval(name, value):
    value = _finite_positive(name, value)
    if value >= 1.0:
        raise ValueError(f"{name} must be less than one")
    return value


def _finite_scalar_loss(value, name):
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise RuntimeError(f"{name} must be a scalar tensor")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{name} must be finite and nonnegative")
    return result
