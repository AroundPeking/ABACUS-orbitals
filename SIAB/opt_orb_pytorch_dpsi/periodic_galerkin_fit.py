"""Optimize compact SIAB radial subspaces against exact periodic Pi."""

import copy
from dataclasses import dataclass
import math

import torch

from periodic_galerkin_data import PeriodicGalerkinDataset
from periodic_galerkin_optimization import (
    evaluate_periodic_galerkin_coefficient_response,
)
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


@dataclass(frozen=True)
class PeriodicGalerkinFitResult:
    coefficients: dict
    history: tuple
    initial_loss: float
    best_loss: float
    best_step: int
    steps_completed: int
    stop_reason: str
    total_backtracks: int
    final_learning_rate: float
    initial_minimum_occupied_capture: float
    occupied_capture_floor: float


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


def _global_pi_loss(
    datasets,
    coefficients,
    *,
    occupied_capture_tolerance=1.0e-6,
):
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    minimum_capture = math.inf
    maximum_condition = 1.0
    for dataset in datasets:
        result = evaluate_periodic_galerkin_coefficient_response(
            dataset,
            coefficients,
            contraction_backend="block",
            occupied_capture_tolerance=occupied_capture_tolerance,
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
    maximum_backtracks=20,
    occupied_capture_degradation_tolerance=1.0e-8,
    progress_callback=None,
):
    """Optimize only the nonfixed radial columns and retain the best subspace."""
    learning_rate = _finite_positive("learning_rate", learning_rate)
    plateau_relative_improvement = _finite_positive(
        "plateau_relative_improvement", plateau_relative_improvement
    )
    if (
        not isinstance(occupied_capture_degradation_tolerance, (int, float))
        or isinstance(occupied_capture_degradation_tolerance, bool)
        or not math.isfinite(occupied_capture_degradation_tolerance)
        or occupied_capture_degradation_tolerance < 0.0
        or occupied_capture_degradation_tolerance >= 1.0
    ):
        raise ValueError(
            "occupied_capture_degradation_tolerance must be finite in [0, 1)"
        )
    occupied_capture_degradation_tolerance = float(
        occupied_capture_degradation_tolerance
    )
    if (
        type(max_steps) is not int
        or type(minimum_steps) is not int
        or type(plateau_patience) is not int
        or max_steps <= 0
        or minimum_steps < 0
        or minimum_steps > max_steps
        or plateau_patience <= 0
        or type(maximum_backtracks) is not int
        or maximum_backtracks < 0
    ):
        raise ValueError("optimization step and plateau controls are invalid")
    fixed, variable, parameters = _validate_inputs(datasets, initial, fixed_nu)
    datasets = tuple(prepare_periodic_occupied_reference(dataset) for dataset in datasets)
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
    pending_evaluation = None
    backtracks_from_previous_step = 0
    total_backtracks = 0
    initial_minimum_occupied_capture = None
    occupied_capture_floor = None
    occupied_capture_tolerance = None

    for step in range(max_steps + 1):
        coefficients = _assemble(fixed, variable)
        if pending_evaluation is None:
            loss, minimum_capture, maximum_condition = _global_pi_loss(
                datasets,
                coefficients,
                occupied_capture_tolerance=(
                    1.0 - 1.0e-12
                    if occupied_capture_tolerance is None
                    else occupied_capture_tolerance
                ),
            )
        else:
            loss, minimum_capture, maximum_condition = pending_evaluation
            pending_evaluation = None
        if initial_minimum_occupied_capture is None:
            initial_minimum_occupied_capture = minimum_capture
            occupied_capture_floor = max(
                0.0,
                minimum_capture - occupied_capture_degradation_tolerance,
            )
            occupied_capture_tolerance = min(
                1.0 - 1.0e-15,
                max(1.0e-15, 1.0 - occupied_capture_floor),
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
                "occupied_capture_floor": occupied_capture_floor,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "backtracks_from_previous_step": backtracks_from_previous_step,
            }
        )
        backtracks_from_previous_step = 0
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
        parameter_snapshot = [parameter.detach().clone() for parameter in parameters]
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        base_learning_rates = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        accepted = False
        for backtrack in range(maximum_backtracks + 1):
            if backtrack:
                with torch.no_grad():
                    for parameter, snapshot in zip(parameters, parameter_snapshot):
                        parameter.copy_(snapshot)
                optimizer.load_state_dict(optimizer_snapshot)
                for group, base_rate in zip(
                    optimizer.param_groups, base_learning_rates
                ):
                    group["lr"] = base_rate * (0.5 ** backtrack)
            optimizer.step()
            try:
                _retract_variables(fixed, variable)
                trial_coefficients = _assemble(fixed, variable)
                pending_evaluation = _global_pi_loss(
                    datasets,
                    trial_coefficients,
                    occupied_capture_tolerance=occupied_capture_tolerance,
                )
            except RuntimeError as error:
                if str(error) != (
                    "candidate basis does not capture the fixed occupied manifold"
                ):
                    raise
                continue
            accepted = True
            backtracks_from_previous_step = backtrack
            total_backtracks += backtrack
            break
        if not accepted:
            with torch.no_grad():
                for parameter, snapshot in zip(parameters, parameter_snapshot):
                    parameter.copy_(snapshot)
            optimizer.load_state_dict(optimizer_snapshot)
            stop_reason = "occupied_capture_boundary"
            break

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
        total_backtracks=total_backtracks,
        final_learning_rate=float(optimizer.param_groups[0]["lr"]),
        initial_minimum_occupied_capture=initial_minimum_occupied_capture,
        occupied_capture_floor=occupied_capture_floor,
    )
