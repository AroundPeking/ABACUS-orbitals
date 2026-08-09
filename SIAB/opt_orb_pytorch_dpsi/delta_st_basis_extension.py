"""Deterministic radial-shell extension for Delta-ST response compression."""

from dataclasses import dataclass
import math
from typing import Mapping, Tuple

import torch


@dataclass(frozen=True)
class DeltaSTBasisExtensionResult:
    coefficients: Mapping[str, object]
    element: str
    l: int
    selected_mode: int
    initial_loss: float
    selected_loss: float
    candidate_losses: Tuple[float, ...]
    radial_metric_condition: float
    maximum_magnetic_metric_relative_deviation: float
    maximum_metric_orthogonality: float
    metric_normalization_error: float


def select_metric_complement_shell(
    primitive,
    objective,
    coefficients,
    *,
    element,
    l,
    metric_tolerance=1.0e-10,
):
    """Append the metric-complement direction with the lowest immediate loss."""
    if not hasattr(objective, "evaluate"):
        raise TypeError("objective must provide evaluate(coefficients)")
    if not isinstance(element, str) or not element:
        raise ValueError("element must be nonempty")
    if isinstance(l, bool) or not isinstance(l, int) or l < 0:
        raise ValueError("l must be a nonnegative integer")
    try:
        metric_tolerance = float(metric_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric_tolerance must be positive and finite") from exc
    if not math.isfinite(metric_tolerance) or metric_tolerance <= 0.0:
        raise ValueError("metric_tolerance must be positive and finite")

    current = _radial_channel(coefficients, element, l)
    metric, magnetic_metric_deviation = _common_radial_metric(
        primitive,
        element,
        l,
        metric_tolerance,
    )
    if current.shape[0] != metric.shape[0]:
        raise ValueError("radial coefficient rows do not match the primitive metric")
    if current.shape[1] >= current.shape[0]:
        raise RuntimeError("radial channel has no metric complement")

    cholesky = torch.linalg.cholesky(metric)
    whitened = cholesky.mT @ current
    q, _ = torch.linalg.qr(whitened, mode="complete")
    complement = q[:, current.shape[1] :]
    directions = torch.linalg.solve_triangular(
        cholesky.mT,
        complement,
        upper=True,
    )

    initial_loss = _finite_loss(objective.evaluate(coefficients).loss, "initial loss")
    candidates = []
    losses = []
    for mode in range(directions.shape[1]):
        candidate = _clone_coefficients(coefficients)
        candidate[element][l] = torch.cat(
            (candidate[element][l], directions[:, mode : mode + 1]),
            dim=1,
        )
        loss = _finite_loss(
            objective.evaluate(candidate).loss,
            f"candidate loss for l={l} mode={mode}",
        )
        candidates.append(candidate)
        losses.append(loss)
    selected_mode = min(range(len(losses)), key=lambda index: (losses[index], index))
    selected_direction = directions[:, selected_mode : selected_mode + 1]
    orthogonality = current.mT @ metric @ selected_direction
    normalization = selected_direction.mT @ metric @ selected_direction
    condition = float(torch.linalg.cond(metric))
    if not math.isfinite(condition):
        raise RuntimeError("radial metric condition number is not finite")
    return DeltaSTBasisExtensionResult(
        coefficients=candidates[selected_mode],
        element=element,
        l=l,
        selected_mode=selected_mode,
        initial_loss=initial_loss,
        selected_loss=losses[selected_mode],
        candidate_losses=tuple(losses),
        radial_metric_condition=condition,
        maximum_magnetic_metric_relative_deviation=magnetic_metric_deviation,
        maximum_metric_orthogonality=(
            0.0 if orthogonality.numel() == 0 else float(torch.max(torch.abs(orthogonality)))
        ),
        metric_normalization_error=float(
            torch.max(torch.abs(normalization - torch.eye(1, dtype=torch.float64)))
        ),
    )


def _common_radial_metric(primitive, element, l, tolerance):
    blocks = tuple(
        block
        for block in primitive.blocks
        if block.element == element and block.atom_index == 0 and block.l == l
    )
    if not blocks:
        raise ValueError(f"primitive data has no {element} l={l} radial block")
    metrics = []
    for block in blocks:
        row = slice(block.offset, block.offset + block.n_primitive)
        value = primitive.overlap[row, row]
        value = 0.5 * (value + value.mH)
        scale = max(float(torch.max(torch.abs(value))), 1.0)
        if float(torch.max(torch.abs(torch.imag(value)))) > tolerance * scale:
            raise RuntimeError("atomic radial metric is materially complex")
        metrics.append(torch.real(value).to(torch.float64))
    metric = torch.mean(torch.stack(metrics, dim=0), dim=0)
    scale = float(torch.max(torch.abs(metric)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("atomic radial metric scale is not positive and finite")
    maximum_relative_deviation = max(
        float(torch.max(torch.abs(value - metric))) / scale for value in metrics
    )
    return metric, maximum_relative_deviation


def _radial_channel(coefficients, element, l):
    try:
        value = coefficients[element][l]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"coefficients are missing {element} l={l}") from exc
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or value.ndim != 2
        or value.device.type != "cpu"
    ):
        raise ValueError("radial coefficients must be CPU float64 matrices")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("radial coefficients must be finite")
    return value


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


def _finite_loss(value, name):
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise RuntimeError(f"{name} must be a scalar tensor")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{name} must be finite and nonnegative")
    return result
