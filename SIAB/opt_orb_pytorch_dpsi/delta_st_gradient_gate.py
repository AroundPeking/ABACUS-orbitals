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


def _clone_coefficients(coefficients):
    result = {}
    for element, by_l in coefficients.items():
        if isinstance(by_l, Mapping):
            result[element] = {
                l: matrix.detach().clone() for l, matrix in by_l.items()
            }
        else:
            result[element] = [matrix.detach().clone() for matrix in by_l]
    return result


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


def _finite_scalar_loss(value, name):
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise RuntimeError(f"{name} must be a scalar tensor")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{name} must be finite and nonnegative")
    return result
