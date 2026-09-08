"""Bounded consecutive descent with measured PBE acceptance, using only NumPy.

The ambient PBE vector is an approximate directional secant model, not an exact
derivative. Neither this model nor any stopping status certifies stationarity.
The adapter owns the manifold, actual SCF measurement, and objective derivative.
"""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real

import numpy as np


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _vector(value, name, shape=None):
    value = np.asarray(value)
    if value.dtype.kind not in "iuf" or value.ndim != 1 or not value.size:
        raise ValueError(f"{name} must be a nonempty flat real vector")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
    value = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _norm(vector):
    # Scaling avoids overflow/underflow in dot(v, v) for valid finite vectors.
    scale = float(np.max(np.abs(vector)))
    if scale == 0.:
        return 0.
    with np.errstate(over="ignore", invalid="ignore"):
        return _scalar(scale * np.linalg.norm(vector / scale), "vector norm")


def _dot(left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        return _scalar(np.dot(left, right), "model dot product")


def _record(value, pbe_limit, measured_pbe=None):
    if not isinstance(value, Mapping):
        raise ValueError("record must be a mapping")
    if value.get("gate", "pass") != "pass":
        raise ValueError("record gate must pass")
    result = deepcopy(dict(value))
    for key in ("objective", "loss"):
        result[key] = _scalar(result.get(key), key)
        if result[key] < 0.:
            raise ValueError(f"{key} must be nonnegative")
    if measured_pbe is not None:
        if "pbe" in result and _scalar(result["pbe"], "pbe") != measured_pbe:
            raise ValueError("evaluation pbe disagrees with actual measurement")
        result["pbe"] = measured_pbe
    result["pbe"] = _scalar(result.get("pbe"), "pbe")
    if abs(result["pbe"]) > pbe_limit:
        raise ValueError("actual pbe exceeds pbe_limit")
    return result


def run_consecutive(initial, initial_record, pbe_gradient, *, gradient, retract,
                    project, measure, evaluate, checkpoint, max_steps=20,
                    max_trials=80, initial_radius=.02, max_radius=.04,
                    min_radius=1e-5, pbe_limit=.010, pbe_target=.0085,
                    gradient_tolerance=1e-10, gain_tolerance=1e-8,
                    loss_gradient=None):
    """Optimize all coefficients with a fresh objective gradient at each center.

    ``gradient(x)`` returns the full objective derivative. ``project(x, v)`` is
    the orthogonal tangent projection; ``retract(x, step)`` returns the signed-QR
    (or other adapter-owned) manifold point. These callbacks must return finite
    flat vectors of the initial shape; invalid results raise ValueError.

    Optional ``loss_gradient(x)`` returns the full exact weighted-loss derivative.
    It is called just after gradient at each new center, so the adapter may read
    a shared cache. Neither derivative is recalled after rejection at that center;
    both are reprojected when the PBE secant model changes. With None, the original
    energy-only proposal and diagnostics are unchanged. Otherwise the direction
    uses the minimum-norm convex combination of normalized PBE-tangent gradients.
    A tiny combination returns status/stop_reason='unresolved_joint_direction',
    not convergence. A tiny loss tangent uses the energy direction instead.
    An unresolved joint stop also returns direction_diagnostic with its reason,
    angle, mixing weight, gradient/combination norms, and tolerance. If the
    combination is nonzero but fails the slope check, both directional slopes
    are included. This does not create a trial or an acceptance checkpoint.

    Joint history adds loss_gradient_norm, projected_loss_gradient_norm,
    feasible_loss_gradient_norm, joint_angle_degrees, joint_mixing_weight (the Ec
    coefficient), joint_combination_norm, predicted_step_loss_delta, and
    predicted_loss_delta (the retracted chord prediction, diagnostic only).
    Angle is None for a tiny loss tangent. Inward correction reserves at least
    half of each objective's tangential descent. Actual nonincreasing loss and
    measured PBE acceptance remain mandatory, irrespective of the model.

    ``measure(x, trial_id)`` returns a gate and, when actual SCF exists, signed
    ``pbe`` in eV/C. Only gate='pass' with abs(pbe)<=pbe_limit calls evaluate.
    ``evaluate`` returns nonnegative finite objective (absolute Ha/cell error)
    and loss; optional pbe must match the measurement. Malformed or nonfinite
    measurement/evaluation records are rejected, not accepted or extrapolated.
    Callback exceptions propagate. Trial IDs are zero-based, including trials
    rejected before measurement, so measured IDs need not be contiguous.

    Checkpoints after each acceptance and the return value contain independent
    copies of x, record, accepted_steps, trials, radius, history, counts, best,
    and pbe_gradient. ``best`` is always the latest accepted x/record, or the
    initial pair if none passed. Counts include gradient_calls and, when enabled,
    loss_gradient_calls. Checkpoints
    have status='running'; the return has status and stop_reason indicating a
    budget or an unresolved direction, never convergence. No final checkpoint
    is emitted for a rejected point. The first gradient is called once and may
    come from the adapter's calibration cache; later centers need fresh values.

    Radius halves on rejection and grows by 1.5 on acceptance up to max_radius.
    Positive gains <=gain_tolerance are recorded as small_gain, not convergence
    or grounds to stop refreshing gradients. All callback arguments are copies.
    """
    for name, value in (("max_steps", max_steps), ("max_trials", max_trials)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    radius = _scalar(initial_radius, "initial_radius")
    max_radius = _scalar(max_radius, "max_radius")
    min_radius = _scalar(min_radius, "min_radius")
    pbe_limit = _scalar(pbe_limit, "pbe_limit")
    pbe_target = _scalar(pbe_target, "pbe_target")
    gradient_tolerance = _scalar(gradient_tolerance, "gradient_tolerance")
    gain_tolerance = _scalar(gain_tolerance, "gain_tolerance")
    if not 0. < min_radius <= radius <= max_radius:
        raise ValueError("radii must satisfy 0 < min_radius <= initial_radius <= max_radius")
    if not 0. <= pbe_target <= pbe_limit or pbe_limit <= 0.:
        raise ValueError("PBE limits must satisfy 0 <= pbe_target <= pbe_limit, pbe_limit > 0")
    if gradient_tolerance < 0. or gain_tolerance < 0.:
        raise ValueError("tolerances must be nonnegative")
    for callback in (gradient, retract, project, measure, evaluate, checkpoint):
        if not callable(callback):
            raise ValueError("all callbacks must be callable")
    if loss_gradient is not None and not callable(loss_gradient):
        raise ValueError("loss_gradient must be callable or None")
    x = _vector(initial, "initial")
    b = _vector(pbe_gradient, "pbe_gradient", x.shape)
    record = _record(initial_record, pbe_limit)
    counts = dict(accepted_steps=0, trials=0, gradient_calls=0,
                  measurements=0, evaluations=0, secant_updates=0)
    if loss_gradient is not None:
        counts["loss_gradient_calls"] = 0
    history = []
    g = None
    loss_g = None

    def snapshot(status, stop_reason=None, direction_diagnostic=None):
        state = dict(x=x.copy(), record=deepcopy(record), radius=radius,
                     accepted_steps=counts["accepted_steps"], trials=counts["trials"],
                     counts=counts.copy(), history=deepcopy(history),
                     pbe_gradient=b.copy(), best=dict(x=x.copy(), record=deepcopy(record)),
                     status=status, stop_reason=stop_reason)
        if direction_diagnostic is not None:
            state["direction_diagnostic"] = deepcopy(direction_diagnostic)
        return state

    while True:
        if counts["accepted_steps"] >= max_steps:
            return snapshot("max_steps", "max_steps")
        if counts["trials"] >= max_trials:
            return snapshot("max_trials", "max_trials")
        if radius < min_radius:
            return snapshot("feasible_direction_unresolved", "min_radius")
        if g is None:
            counts["gradient_calls"] += 1
            g = _vector(gradient(x.copy()), "gradient", x.shape)
            if loss_gradient is not None:
                counts["loss_gradient_calls"] += 1
                loss_g = _vector(loss_gradient(x.copy()), "loss_gradient", x.shape)
        pg = _vector(project(x.copy(), g.copy()), "projected gradient", x.shape)
        pb = _vector(project(x.copy(), b.copy()), "projected pbe_gradient", x.shape)
        g_norm, pg_norm, b_norm = _norm(g), _norm(pg), _norm(pb)
        if pg_norm <= gradient_tolerance:
            return snapshot("feasible_direction_unresolved", "projected_gradient_tiny")
        normal_axis = pb / b_norm if b_norm > gradient_tolerance else np.zeros_like(x)
        feasible = pg - _dot(pg, normal_axis) * normal_axis
        feasible_norm = _norm(feasible)
        if feasible_norm <= gradient_tolerance:
            return snapshot("feasible_direction_unresolved", "pbe_tangent_gradient_tiny")
        direction = -feasible / feasible_norm
        joint = None
        if loss_g is not None:
            pl = _vector(project(x.copy(), loss_g.copy()), "projected loss_gradient", x.shape)
            feasible_loss = _vector(pl - _dot(pl, normal_axis) * normal_axis,
                                    "PBE-tangent loss_gradient", x.shape)
            loss_norm = _norm(feasible_loss)
            joint = dict(loss_gradient_norm=_norm(loss_g), projected_loss_gradient_norm=_norm(pl),
                         feasible_loss_gradient_norm=loss_norm, joint_angle_degrees=None,
                         joint_mixing_weight=1., joint_combination_norm=1.)
            if loss_norm > gradient_tolerance:
                u, v = feasible / feasible_norm, feasible_loss / loss_norm
                difference = u - v
                denominator = _dot(difference, difference)
                # Minimize ||v + alpha*(u-v)||^2 on [0, 1]. At a nonzero
                # minimum, its negative has negative slope for both gradients.
                alpha = float(np.clip(-_dot(v, difference) / denominator, 0., 1.)) if denominator > 0. else 1.
                combination = alpha * u + (1. - alpha) * v
                combination_norm = _norm(combination)
                joint.update(joint_angle_degrees=float(np.degrees(np.arccos(np.clip(_dot(u, v), -1., 1.)))),
                             joint_mixing_weight=alpha, joint_combination_norm=combination_norm)
                diagnostic = dict(joint, gradient_norm=g_norm, projected_gradient_norm=pg_norm,
                                  feasible_gradient_norm=feasible_norm, pbe_gradient_norm=b_norm,
                                  gradient_tolerance=gradient_tolerance)
                if combination_norm <= gradient_tolerance:
                    diagnostic["reason"] = "joint_combination_tiny"
                    return snapshot("unresolved_joint_direction", "unresolved_joint_direction", diagnostic)
                direction = -combination / combination_norm
                objective_slope, loss_slope = _dot(g, direction), _dot(loss_g, direction)
                if objective_slope >= 0. or loss_slope >= 0.:
                    diagnostic.update(reason="non_descending_common_direction",
                                      objective_direction_slope=objective_slope,
                                      loss_direction_slope=loss_slope)
                    return snapshot("unresolved_joint_direction", "unresolved_joint_direction", diagnostic)
        normal_length = 0.
        inward = -np.sign(record["pbe"]) * normal_axis
        if abs(record["pbe"]) > pbe_target and b_norm > gradient_tolerance:
            normal_length = min(.3 * radius, (abs(record["pbe"]) - pbe_target) / b_norm)
            for full_gradient in ((g,) if loss_g is None else (g, loss_g)):
                cost = _dot(full_gradient, inward)
                if cost > 0.:
                    # Reserve half of each tangential descent. A zero loss slope
                    # permits no harmful normal correction, even if Ec benefits.
                    tangent_gain = -radius * np.sqrt(1. - .3 ** 2) * _dot(full_gradient, direction)
                    normal_length = min(normal_length, max(0., .5 * tangent_gain / cost))
        tangent_length = radius * np.sqrt(max(0., 1. - (normal_length / radius) ** 2))
        step = _vector(tangent_length * direction + normal_length * inward, "step", x.shape)
        row = dict(trial_id=counts["trials"], radius=radius, accepted=False,
                   reason=None, gradient_norm=g_norm, projected_gradient_norm=pg_norm,
                   feasible_gradient_norm=feasible_norm, pbe_gradient_norm=b_norm,
                   normal_correction_norm=normal_length, step_norm=_norm(step),
                   predicted_step_objective_delta=_dot(g, step),
                   predicted_objective_delta=None, predicted_pbe=None,
                   pbe_prediction_error=None, actual_pbe=None, objective_gain=None,
                   small_gain=False, secant_updated=False)
        if joint is not None:
            row.update(joint, predicted_step_loss_delta=_dot(loss_g, step), predicted_loss_delta=None)
        counts["trials"] += 1
        candidate = None
        if row["predicted_step_objective_delta"] >= 0.:
            row["reason"] = "non_descent_proposal"
        elif loss_g is not None and row["predicted_step_loss_delta"] > 0.:
            row["reason"] = "non_descent_loss_proposal"
        else:
            candidate = _vector(retract(x.copy(), step.copy()), "retraction", x.shape)
            dx = _vector(candidate - x, "retracted displacement", x.shape)
            row["x"] = candidate.copy()
            row["predicted_objective_delta"] = _dot(g, dx)
            row["predicted_pbe"] = _scalar(record["pbe"] + _dot(b, dx), "predicted pbe")
            if loss_g is not None:
                row["predicted_loss_delta"] = _dot(loss_g, dx)
            if row["predicted_objective_delta"] >= 0.:
                row["reason"] = "non_descent_retraction"
        if row["reason"] is None:
            counts["measurements"] += 1
            measured = measure(candidate.copy(), row["trial_id"])
            row["measurement"] = deepcopy(measured)
            actual_pbe = None
            if isinstance(measured, Mapping):
                try:
                    actual_pbe = _scalar(measured.get("pbe"), "actual pbe")
                except ValueError:
                    pass
            if actual_pbe is not None:
                row["actual_pbe"] = actual_pbe
                error = _scalar(actual_pbe - row["predicted_pbe"], "PBE prediction error")
                row["pbe_prediction_error"] = error
                # Use the ambient retracted chord, never coefficients in old directions.
                # This is algebraically error*dx/(dx@dx), with scaled arithmetic.
                dx_norm = _norm(dx)
                if dx_norm > 0.:
                    with np.errstate(over="ignore", invalid="ignore"):
                        b = _vector(b + (error / dx_norm) * (dx / dx_norm),
                                    "updated pbe_gradient", x.shape)
                    counts["secant_updates"] += 1
                    row["secant_updated"] = True
            row["pbe_gradient_after"] = b.copy()
            if not isinstance(measured, Mapping) or measured.get("gate") != "pass":
                row["reason"] = ("cheap_reject" if isinstance(measured, Mapping)
                                 and measured.get("gate") == "cheap_reject" else "measurement_failed")
            elif actual_pbe is None:
                row["reason"] = "invalid_actual_pbe"
            elif abs(actual_pbe) > pbe_limit:
                row["reason"] = "pbe_limit_exceeded"
            else:
                counts["evaluations"] += 1
                evaluated = evaluate(candidate.copy(), row["trial_id"])
                row["evaluation"] = deepcopy(evaluated)
                try:
                    trial_record = _record(evaluated, pbe_limit, actual_pbe)
                except ValueError as error:
                    row["reason"] = "invalid_evaluation"
                    row["detail"] = str(error)
                else:
                    gain = record["objective"] - trial_record["objective"]
                    row["objective_gain"] = gain
                    if gain <= 0.:
                        row["reason"] = "objective_not_improved"
                    elif trial_record["loss"] > record["loss"]:
                        row["reason"] = "loss_increased"
                    else:
                        row.update(accepted=True, reason="accepted", small_gain=gain <= gain_tolerance)
                        x, record = candidate.copy(), trial_record
                        counts["accepted_steps"] += 1
                        g = None
                        loss_g = None
        radius = min(max_radius, radius * 1.5) if row["accepted"] else radius * .5
        row["radius_next"] = radius
        history.append(row)
        if row["accepted"]:
            checkpoint(snapshot("running"))
