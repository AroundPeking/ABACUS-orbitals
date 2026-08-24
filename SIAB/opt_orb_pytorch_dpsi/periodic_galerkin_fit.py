"""Optimize compact SIAB radial subspaces against exact periodic Pi."""

from dataclasses import dataclass
import math

import torch

from periodic_galerkin_data import PeriodicGalerkinDataset
from periodic_galerkin_optimization import (
    evaluate_periodic_galerkin_coefficient_response,
)


@dataclass(frozen=True)
class PeriodicGalerkinFitResult:
    coefficients: dict
    history: tuple
    initial_loss: float
    best_loss: float
    best_step: int
    steps_completed: int
    stop_reason: str


def _finite_positive(name, value):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(name + " must be finite and positive")
    return float(value)


def _validate_inputs(datasets, initial, fixed_nu):
    if not isinstance(datasets, tuple) or not datasets:
        raise ValueError("datasets must be a nonempty tuple")
    if any(not isinstance(dataset, PeriodicGalerkinDataset) for dataset in datasets):
        raise ValueError("datasets contain an invalid periodic dataset")
    if not isinstance(initial, dict) or not initial:
        raise ValueError("initial coefficients must be a nonempty dictionary")
    if not isinstance(fixed_nu, dict) or set(fixed_nu) != set(initial):
        raise ValueError("fixed_nu elements must match initial coefficients")

    fixed = {}
    variable = {}
    parameters = []
    for element, channels in initial.items():
        if not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError("initial coefficient channels must be nonempty")
        try:
            counts = tuple(fixed_nu[element])
        except TypeError as error:
            raise ValueError("fixed_nu channels must be a sequence") from error
        if len(counts) != len(channels):
            raise ValueError("fixed_nu must define every angular channel")
        fixed[element] = []
        variable[element] = []
        for l, (channel, count) in enumerate(zip(channels, counts)):
            if (
                not isinstance(channel, torch.Tensor)
                or channel.device.type != "cpu"
                or channel.dtype != torch.float64
                or channel.ndim != 2
                or channel.shape[0] <= 0
                or not bool(torch.isfinite(channel).all())
            ):
                raise ValueError("initial channels must be finite CPU float64 matrices")
            if type(count) is not int or count < 0 or count > channel.shape[1]:
                raise ValueError("fixed_nu exceeds a candidate channel")
            fixed_channel = channel[:, :count].detach().clone()
            variable_channel = torch.nn.Parameter(
                channel[:, count:].detach().clone()
            )
            fixed[element].append(fixed_channel)
            variable[element].append(variable_channel)
            if variable_channel.shape[1]:
                if variable_channel.shape[1] + count > channel.shape[0]:
                    raise ValueError("candidate radial channel exceeds primitive rank")
                parameters.append(variable_channel)
    if not parameters:
        raise ValueError("fixed_nu leaves no variable radial orbital")
    return fixed, variable, parameters


def _assemble(fixed, variable):
    return {
        element: [
            torch.cat((fixed_channel, variable_channel), dim=1)
            for fixed_channel, variable_channel in zip(
                fixed[element], variable[element]
            )
        ]
        for element in fixed
    }


def _retract_variables(fixed, variable, rank_tolerance=1.0e-12):
    with torch.no_grad():
        for element in fixed:
            for fixed_channel, parameter in zip(fixed[element], variable[element]):
                if parameter.shape[1] == 0:
                    continue
                value = parameter
                if fixed_channel.shape[1]:
                    gram = fixed_channel.transpose(0, 1).matmul(fixed_channel)
                    projection = torch.linalg.solve(
                        gram,
                        fixed_channel.transpose(0, 1).matmul(value),
                    )
                    value = value - fixed_channel.matmul(projection)
                singular_value = torch.linalg.svdvals(value)
                if (
                    singular_value.shape[0] != parameter.shape[1]
                    or float(torch.min(singular_value))
                    <= rank_tolerance * float(torch.max(singular_value))
                ):
                    raise RuntimeError("variable radial channel is rank deficient")
                frame, triangular = torch.linalg.qr(value, mode="reduced")
                diagonal = torch.diagonal(triangular)
                sign = torch.where(
                    diagonal < 0.0,
                    -torch.ones_like(diagonal),
                    torch.ones_like(diagonal),
                )
                parameter.copy_(frame * sign)


def _global_pi_loss(datasets, coefficients):
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    minimum_capture = math.inf
    maximum_condition = 1.0
    for dataset in datasets:
        result = evaluate_periodic_galerkin_coefficient_response(
            dataset,
            coefficients,
            contraction_backend="dense",
        )
        weight = dataset.frequency_weights_ha[:, None, None]
        q_weight = dataset.q_weight
        numerator = numerator + q_weight * torch.sum(
            weight * torch.abs(result.response - dataset.reference_response) ** 2
        )
        denominator = denominator + q_weight * torch.sum(
            weight * torch.abs(dataset.reference_response) ** 2
        )
        minimum_capture = min(minimum_capture, result.minimum_occupied_capture)
        maximum_condition = max(maximum_condition, result.maximum_overlap_condition)
    if float(denominator.detach()) == 0.0:
        raise RuntimeError("periodic exact Pi has zero norm")
    return numerator / denominator, minimum_capture, maximum_condition


def _clone_coefficients(coefficients):
    return {
        element: [channel.detach().clone() for channel in channels]
        for element, channels in coefficients.items()
    }


def optimize_periodic_galerkin_basis(
    datasets,
    initial,
    *,
    fixed_nu,
    learning_rate=1.0e-3,
    max_steps=2000,
    minimum_steps=200,
    plateau_patience=300,
    plateau_relative_improvement=1.0e-6,
    progress_callback=None,
):
    """Optimize only the nonfixed radial columns and retain the best subspace."""
    learning_rate = _finite_positive("learning_rate", learning_rate)
    plateau_relative_improvement = _finite_positive(
        "plateau_relative_improvement", plateau_relative_improvement
    )
    if (
        type(max_steps) is not int
        or type(minimum_steps) is not int
        or type(plateau_patience) is not int
        or max_steps <= 0
        or minimum_steps < 0
        or minimum_steps > max_steps
        or plateau_patience <= 0
    ):
        raise ValueError("optimization step and plateau controls are invalid")
    fixed, variable, parameters = _validate_inputs(datasets, initial, fixed_nu)
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    _retract_variables(fixed, variable)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    history = []
    initial_loss = None
    best_loss = math.inf
    best_step = 0
    best_coefficients = None
    significant_loss = math.inf
    last_significant_step = 0
    stop_reason = "maximum_steps"

    for step in range(max_steps + 1):
        coefficients = _assemble(fixed, variable)
        loss, minimum_capture, maximum_condition = _global_pi_loss(
            datasets,
            coefficients,
        )
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError("periodic Pi loss is non-finite")
        if initial_loss is None:
            initial_loss = loss_value
            significant_loss = loss_value
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
            best_coefficients = _clone_coefficients(coefficients)
        if loss_value < significant_loss * (1.0 - plateau_relative_improvement):
            significant_loss = loss_value
            last_significant_step = step
        history.append(
            {
                "step": step,
                "loss": loss_value,
                "relative_pi_error": math.sqrt(loss_value),
                "minimum_occupied_capture": minimum_capture,
                "maximum_overlap_condition": maximum_condition,
            }
        )
        if progress_callback is not None:
            progress_callback(history[-1])

        if step == max_steps:
            break
        if (
            step >= minimum_steps
            and step - last_significant_step >= plateau_patience
        ):
            stop_reason = "plateau"
            break
        optimizer.zero_grad()
        loss.backward()
        for parameter in parameters:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("periodic Pi gradient is missing or non-finite")
        optimizer.step()
        _retract_variables(fixed, variable)

    if best_coefficients is None:
        raise RuntimeError("periodic basis optimization produced no accepted step")
    return PeriodicGalerkinFitResult(
        coefficients=best_coefficients,
        history=tuple(history),
        initial_loss=initial_loss,
        best_loss=best_loss,
        best_step=best_step,
        steps_completed=history[-1]["step"],
        stop_reason=stop_reason,
    )
