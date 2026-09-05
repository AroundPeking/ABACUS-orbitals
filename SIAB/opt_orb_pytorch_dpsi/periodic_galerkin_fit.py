"""Optimize compact SIAB radial subspaces against exact periodic Pi."""

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import time

import torch

from periodic_galerkin_data import PeriodicGalerkinDataset
from periodic_galerkin_basis import (
    build_primitive_to_candidate,
    prepare_periodic_block_contraction_record,
)
from periodic_galerkin_optimization import (
    evaluate_periodic_galerkin_coefficient_response,
)
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference
from periodic_galerkin_rpa import periodic_rpa_objective


@dataclass(frozen=True)
class PeriodicGalerkinFitResult:
    coefficients: dict
    history: tuple
    initial_loss: float
    initial_family_losses: dict
    best_loss: float
    best_family_losses: dict
    best_step: int
    steps_completed: int
    stop_reason: str
    total_backtracks: int
    final_learning_rate: float
    occupied_capture_reference: str
    reference_minimum_occupied_capture: float
    initial_minimum_occupied_capture: float
    occupied_capture_floor: float
    objective: str = "pi"
    objective_weights: dict = None


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
    dataset_families=None,
    additional_family_evaluators=None,
):
    if dataset_families is None:
        dataset_families = ("periodic",) * len(datasets)
    else:
        dataset_families = tuple(dataset_families)
    if len(dataset_families) != len(datasets):
        raise ValueError("dataset_families must match datasets")
    if any(not isinstance(name, str) or not name.strip() for name in dataset_families):
        raise ValueError("each family name must be nonempty")
    if additional_family_evaluators is None:
        additional_family_evaluators = {}
    elif not isinstance(additional_family_evaluators, dict):
        raise ValueError("additional_family_evaluators must be a dictionary")
    if any(
        not isinstance(name, str) or not name.strip()
        for name in additional_family_evaluators
    ):
        raise ValueError("each family name must be nonempty")
    if set(dataset_families) & set(additional_family_evaluators):
        raise ValueError("each family name must identify exactly one loss family")

    numerators = {}
    denominators = {}
    minimum_capture = math.inf
    maximum_condition = 1.0
    for dataset, family in zip(datasets, dataset_families):
        result = evaluate_periodic_galerkin_coefficient_response(
            dataset,
            coefficients,
            contraction_backend="block",
            occupied_capture_tolerance=occupied_capture_tolerance,
        )
        weight = dataset.frequency_weights_ha[:, None, None]
        q_weight = dataset.q_weight
        numerator = q_weight * torch.sum(
            weight * torch.abs(result.response - dataset.reference_response) ** 2
        )
        denominator = q_weight * torch.sum(
            weight * torch.abs(dataset.reference_response) ** 2
        )
        numerators[family] = numerators.get(
            family, torch.zeros((), dtype=torch.float64)
        ) + numerator
        denominators[family] = denominators.get(
            family, torch.zeros((), dtype=torch.float64)
        ) + denominator
        minimum_capture = min(minimum_capture, result.minimum_occupied_capture)
        maximum_condition = max(maximum_condition, result.maximum_overlap_condition)

    family_losses = {}
    for family in dict.fromkeys(dataset_families):
        denominator = denominators[family]
        if float(denominator.detach()) == 0.0:
            raise RuntimeError("periodic exact Pi has zero norm")
        family_losses[family] = numerators[family] / denominator
    for family, evaluator in additional_family_evaluators.items():
        if not hasattr(evaluator, "evaluate") or not callable(evaluator.evaluate):
            raise ValueError("additional family evaluator must define evaluate")
        result = evaluator.evaluate(coefficients)
        loss = getattr(result, "loss", None)
        condition = getattr(result, "max_candidate_condition", None)
        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not bool(torch.isfinite(loss))
        ):
            raise RuntimeError("additional family loss must be a finite scalar")
        if (
            not isinstance(condition, (int, float))
            or isinstance(condition, bool)
            or not math.isfinite(condition)
            or condition < 1.0
        ):
            raise RuntimeError("additional family condition must be finite and at least one")
        family_losses[family] = loss
        maximum_condition = max(maximum_condition, float(condition))
    if not family_losses:
        raise RuntimeError("joint response objective has no physical family")
    loss = torch.stack(tuple(family_losses.values())).mean()
    return loss, minimum_capture, maximum_condition, family_losses


def _clone_coefficients(coefficients):
    return {
        element: [channel.detach().clone() for channel in channels]
        for element, channels in coefficients.items()
    }


def _global_rpa_loss(datasets, coefficients, *, occupied_capture_tolerance, weights):
    start = time.perf_counter()
    responses = []
    minimum_capture, maximum_condition = math.inf, 1.0
    for dataset in datasets:
        result = evaluate_periodic_galerkin_coefficient_response(
            dataset, coefficients, contraction_backend="block",
            occupied_capture_tolerance=occupied_capture_tolerance,
        )
        responses.append(result.response)
        minimum_capture = min(minimum_capture, result.minimum_occupied_capture)
        maximum_condition = max(maximum_condition, result.maximum_overlap_condition)
    objective = periodic_rpa_objective(datasets, tuple(responses), **weights)
    diagnostics = {
        key: float(getattr(objective, key).detach())
        for key in ("pi_relative_squared_error", "trace_log_relative_squared_error",
                    "energy_relative_squared_error", "candidate_energy_ha", "reference_energy_ha")
    }
    diagnostics.update(
        q_weight_coverage=objective.q_weight_coverage,
        complete_q_weight=objective.complete_q_weight,
        evaluation_seconds=time.perf_counter() - start,
        per_q=[{
            "selected_iq": record.selected_iq,
            "q_weight": record.q_weight,
            "frequency_ha": record.frequency_ha.tolist(),
            "candidate_contributions_ha": record.candidate_contributions_ha.detach().tolist(),
            "reference_contributions_ha": record.reference_contributions_ha.detach().tolist(),
        } for record in objective.q_records],
    )
    return (objective.loss, minimum_capture, maximum_condition,
            {"periodic": objective.loss}, diagnostics)


def _adjoint(value):
    return value.transpose(-2, -1).conj()


def _minimum_occupied_capture(
    datasets,
    coefficients,
    *,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
):
    minimum_capture = math.inf
    with torch.no_grad():
        for dataset in datasets:
            candidate = build_primitive_to_candidate(
                dataset.primitive_blocks,
                dataset.primitive_count,
                coefficients,
            )
            transform = candidate.transform
            for record in dataset.kpoints:
                overlap = _adjoint(transform).matmul(record.overlap).matmul(
                    transform
                )
                overlap = 0.5 * (overlap + _adjoint(overlap))
                overlap_eigenvalue = torch.linalg.eigvalsh(overlap)
                maximum = torch.max(overlap_eigenvalue)
                if float(maximum) <= 0.0:
                    raise RuntimeError(
                        "fixed radial prefix overlap has no positive direction"
                    )
                minimum = torch.min(overlap_eigenvalue)
                if float(minimum) <= relative_rank_tolerance * float(maximum):
                    raise RuntimeError("fixed radial prefix overlap is rank deficient")
                if float(maximum / minimum) > condition_limit:
                    raise RuntimeError(
                        "fixed radial prefix overlap condition number exceeds limit"
                    )
                identity = torch.eye(overlap.shape[0], dtype=torch.complex128)
                cholesky = torch.linalg.cholesky(overlap)
                lowdin = torch.linalg.solve(_adjoint(cholesky), identity)
                occupied = record.occupied_projection.matmul(transform).matmul(
                    lowdin
                )
                if record.occupied_projection_normalization is not None:
                    occupied = record.occupied_projection_normalization.matmul(
                        occupied
                    )
                capture = occupied.matmul(_adjoint(occupied))
                capture = 0.5 * (capture + _adjoint(capture))
                capture_eigenvalue = torch.linalg.eigvalsh(capture)
                if (
                    capture_eigenvalue.numel() != record.occupation.numel()
                    or float(torch.max(capture_eigenvalue)) <= 0.0
                ):
                    raise RuntimeError(
                        "fixed radial prefix does not span the occupied manifold"
                    )
                minimum_capture = min(
                    minimum_capture,
                    float(torch.min(capture_eigenvalue)),
                )
    if not math.isfinite(minimum_capture):
        raise RuntimeError("fixed radial prefix occupied capture is non-finite")
    return minimum_capture


def _prepare_block_contraction_task(task):
    record, primitive_blocks, coefficients = task
    return prepare_periodic_block_contraction_record(
        record,
        primitive_blocks,
        coefficients,
    )


def _prepare_block_contraction_caches(datasets, coefficients, workers):
    tasks = tuple(
        (record, dataset.primitive_blocks, coefficients)
        for dataset in datasets
        for record in dataset.kpoints
    )
    if workers == 1:
        prepared = tuple(_prepare_block_contraction_task(task) for task in tasks)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared = tuple(executor.map(_prepare_block_contraction_task, tasks))

    offset = 0
    cached_datasets = []
    for dataset in datasets:
        count = len(dataset.kpoints)
        cached_datasets.append(
            replace(dataset, kpoints=prepared[offset:offset + count])
        )
        offset += count
    return tuple(cached_datasets)


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
    occupied_capture_reference="initial_candidate",
    block_cache_workers=1,
    dataset_families=None,
    additional_family_evaluators=None,
    progress_callback=None,
    best_callback=None,
    objective="pi",
    rpa_weights=None,
):
    """Fit nonfixed radial columns to Pi (default) or frozen-space body RPA.

    Partial-q RPA trials are allowed but retain their physical weights. They
    are workflow diagnostics, not complete-solid energy or basis acceptance.
    """
    if objective not in ("pi", "rpa"):
        raise ValueError("objective must be pi or rpa")
    weights = None
    if objective == "pi":
        if rpa_weights is not None:
            raise ValueError("rpa_weights require objective=rpa")
    else:
        if dataset_families is not None or additional_family_evaluators:
            raise ValueError("RPA fitting requires a single periodic response family")
        weights = {"pi_weight": 1.0, "trace_log_weight": 1.0, "energy_weight": 1.0}
        if rpa_weights is not None:
            if not isinstance(rpa_weights, dict) or set(rpa_weights) - set(weights):
                raise ValueError("unknown RPA objective weight")
            weights.update(rpa_weights)
        for key, value in weights.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0.0
                    or (key != "energy_weight" and value == 0.0)):
                raise ValueError("invalid RPA objective weight")
    learning_rate = _finite_positive("learning_rate", learning_rate)
    plateau_relative_improvement = _finite_positive(
        "plateau_relative_improvement", plateau_relative_improvement
    )
    if type(block_cache_workers) is not int or block_cache_workers <= 0:
        raise ValueError("block_cache_workers must be a positive integer")
    if occupied_capture_reference not in ("initial_candidate", "fixed_prefix"):
        raise ValueError(
            "occupied_capture_reference must be initial_candidate or fixed_prefix"
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
    if best_callback is not None and not callable(best_callback):
        raise ValueError("best_callback must be callable")
    _retract_variables(fixed, variable)
    initial_coefficients = _assemble(fixed, variable)
    reference_minimum_occupied_capture = None
    if occupied_capture_reference == "fixed_prefix":
        reference_minimum_occupied_capture = _minimum_occupied_capture(
            datasets,
            fixed,
        )
    datasets = _prepare_block_contraction_caches(
        datasets,
        initial_coefficients,
        block_cache_workers,
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    loss_options = {}
    if dataset_families is not None:
        loss_options["dataset_families"] = tuple(dataset_families)
    if additional_family_evaluators:
        loss_options["additional_family_evaluators"] = dict(
            additional_family_evaluators
        )

    def evaluate_loss(coefficients, occupied_capture_tolerance):
        if objective == "rpa":
            return _global_rpa_loss(
                datasets, coefficients,
                occupied_capture_tolerance=occupied_capture_tolerance,
                weights=weights,
            )
        return _global_pi_loss(
            datasets,
            coefficients,
            occupied_capture_tolerance=occupied_capture_tolerance,
            **loss_options,
        ) + (None,)

    history = []
    initial_loss = None
    initial_family_losses = None
    best_loss = math.inf
    best_family_losses = None
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
    previous_step_gradient_norm = None

    for step in range(max_steps + 1):
        coefficients = _assemble(fixed, variable)
        if pending_evaluation is None:
            loss, minimum_capture, maximum_condition, family_losses, diagnostics = evaluate_loss(
                coefficients,
                (
                    1.0 - 1.0e-12
                    if occupied_capture_tolerance is None
                    else occupied_capture_tolerance
                ),
            )
        else:
            loss, minimum_capture, maximum_condition, family_losses, diagnostics = pending_evaluation
            pending_evaluation = None
        family_loss_values = {
            name: float(value.detach()) for name, value in family_losses.items()
        }
        if initial_minimum_occupied_capture is None:
            initial_minimum_occupied_capture = minimum_capture
            if reference_minimum_occupied_capture is None:
                reference_minimum_occupied_capture = minimum_capture
            occupied_capture_floor = max(
                0.0,
                reference_minimum_occupied_capture
                - occupied_capture_degradation_tolerance,
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
            initial_family_losses = dict(family_loss_values)
            significant_loss = loss_value
        if loss_value < best_loss:
            best_loss = loss_value
            best_family_losses = dict(family_loss_values)
            best_step = step
            best_coefficients = _clone_coefficients(coefficients)
            if best_callback is not None:
                best_callback(
                    best_step,
                    best_loss,
                    _clone_coefficients(best_coefficients),
                )
        if loss_value < significant_loss * (1.0 - plateau_relative_improvement):
            significant_loss = loss_value
            last_significant_step = step
        history.append(
            {
                "step": step,
                "loss": loss_value,
                "relative_pi_error": math.sqrt(loss_value),
                "family_losses": dict(family_loss_values),
                "family_relative_pi_errors": {
                    name: math.sqrt(value)
                    for name, value in family_loss_values.items()
                },
                "minimum_occupied_capture": minimum_capture,
                "maximum_overlap_condition": maximum_condition,
                "occupied_capture_floor": occupied_capture_floor,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "backtracks_from_previous_step": backtracks_from_previous_step,
            }
        )
        if objective == "rpa":
            relative_pi_error = math.sqrt(diagnostics["pi_relative_squared_error"])
            history[-1].update(
                objective="rpa", rpa=diagnostics,
                relative_pi_error=relative_pi_error,
                family_relative_pi_errors={"periodic": relative_pi_error},
                previous_step_gradient_norm=previous_step_gradient_norm,
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
        if objective == "rpa":
            previous_step_gradient_norm = math.sqrt(sum(
                float(parameter.grad.detach().square().sum()) for parameter in parameters
            ))
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
                pending_evaluation = evaluate_loss(
                    trial_coefficients,
                    occupied_capture_tolerance,
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
        initial_family_losses=initial_family_losses,
        best_loss=best_loss,
        best_family_losses=best_family_losses,
        best_step=best_step,
        steps_completed=history[-1]["step"],
        stop_reason=stop_reason,
        total_backtracks=total_backtracks,
        final_learning_rate=float(optimizer.param_groups[0]["lr"]),
        occupied_capture_reference=occupied_capture_reference,
        reference_minimum_occupied_capture=reference_minimum_occupied_capture,
        initial_minimum_occupied_capture=initial_minimum_occupied_capture,
        occupied_capture_floor=occupied_capture_floor,
        objective=objective,
        objective_weights=weights,
    )
