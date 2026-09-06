"""Read-only radial response gradients, separate from actual PBE constraints."""

import math
import time

import torch

from periodic_galerkin_fit import _retract_variables, CandidateGuardError
from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
from periodic_galerkin_rpa import periodic_rpa_objective, prepare_periodic_rpa_reference
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


def _validate(coefficients, other=None):
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError("nonempty coefficient dictionary required")
    if other is not None and (not isinstance(other, dict) or set(other) != set(coefficients)):
        raise ValueError("gradient elements must match coefficients")
    for element, channels in coefficients.items():
        if not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError("nonempty channel sequence required")
        if other is not None and len(other[element]) != len(channels):
            raise ValueError("gradient channels must match coefficients")
        for l, c in enumerate(channels):
            if (not isinstance(c, torch.Tensor) or c.ndim != 2 or c.device.type != "cpu"
                    or c.dtype != torch.float64 or c.shape[0] < 1 or not bool(torch.isfinite(c).all())):
                raise ValueError("finite CPU float64 coefficient matrices required")
            if not torch.allclose(c.T @ c, torch.eye(c.shape[1], dtype=c.dtype), atol=1e-10, rtol=0):
                raise ValueError("coefficients must use the accepted orthonormal QR frame")
            if other is not None:
                g = other[element][l]
                if (not isinstance(g, torch.Tensor) or g.shape != c.shape or g.dtype != c.dtype
                        or g.device != c.device or not bool(torch.isfinite(g).all())):
                    raise ValueError("finite shape-matched coefficient gradients required")


def _finite_norm(value):
    norm = float(value.norm())
    if not math.isfinite(norm):
        raise ValueError("nonfinite derived gradient norm")
    return norm


def radial_gradient_report(coefficients, gradient):
    """Remove channel-internal changes, then report each saved-frame column."""
    _validate(coefficients, gradient)
    channels, raw_total, horizontal_total = [], 0., 0.
    with torch.no_grad():
        for element, values in coefficients.items():
            for l, c in enumerate(values):
                g = gradient[element][l].detach()
                h = g - c @ (c.T @ g)
                raw_norm, horizontal_norm = _finite_norm(g), _finite_norm(h)
                raw_total = math.hypot(raw_total, raw_norm)
                horizontal_total = math.hypot(horizontal_total, horizontal_norm)
                channels.append(dict(element=element, l=l, shape=list(c.shape),
                    raw_gradient=g.tolist(), horizontal_gradient=h.tolist(),
                    raw_norm=raw_norm, horizontal_norm=horizontal_norm,
                    span_component_norm=_finite_norm(g-h),
                    horizontal_residual_norm=_finite_norm(c.T @ h),
                    radials=[dict(zeta=z+1, raw_norm=_finite_norm(g[:, z]),
                                  horizontal_norm=_finite_norm(h[:, z]))
                             for z in range(c.shape[1])]))
    if not math.isfinite(raw_total) or not math.isfinite(horizontal_total):
        raise ValueError("nonfinite combined gradient norm")
    return dict(coordinate_metric="Euclidean_coefficient_signed_QR_frame",
        projection="G_horizontal=G-C(C^T G)",
        per_radial_scope="saved_frame_dependent_not_rotation_invariant",
        raw_gradient_norm=raw_total, horizontal_gradient_norm=horizontal_total,
        channels=channels)


def _validate_report(coefficients, report):
    _validate(coefficients)
    identities = [(e, l) for e, channels in coefficients.items() for l in range(len(channels))]
    if [(r.get("element"), r.get("l")) for r in report["channels"]] != identities:
        raise ValueError("gradient report must contain every channel exactly once in coefficient order")
    for row, (element, l) in zip(report["channels"], identities):
        c = coefficients[element][l]
        if row["shape"] != list(c.shape) or [r["zeta"] for r in row["radials"]] != list(range(1, c.shape[1]+1)):
            raise ValueError("gradient report must contain every saved radial exactly once")
        h = torch.tensor(row["horizontal_gradient"], dtype=torch.float64)
        if h.shape != c.shape or not bool(torch.isfinite(h).all()):
            raise ValueError("invalid horizontal gradient matrix")
        _finite_norm(h)
        for z, radial in enumerate(row["radials"]):
            if not math.isclose(_finite_norm(h[:, z]), radial["horizontal_norm"], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("radial gradient norm mismatch")


def radial_descent_direction(coefficients, report, element, l, column):
    _validate_report(coefficients, report)
    if element not in coefficients or not 0 <= l < len(coefficients[element]):
        raise ValueError("invalid radial channel identity")
    row = next(r for r in report["channels"] if r["element"] == element and r["l"] == l)
    h = torch.tensor(row["horizontal_gradient"], dtype=torch.float64)
    if h.shape != coefficients[element][l].shape or not 0 <= column < h.shape[1]:
        raise ValueError("invalid radial direction identity")
    norm = _finite_norm(h[:, column])
    if norm <= 1e-14:
        return None
    direction = {e: [torch.zeros_like(c) for c in channels] for e, channels in coefficients.items()}
    direction[element][l][:, column] = -h[:, column]/norm
    if not math.isclose(_finite_norm(direction[element][l]), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError("radial descent must have unit norm")
    return direction


def retract_displacement(coefficients, direction, radius):
    _validate(coefficients, direction)
    if isinstance(radius, bool) or not isinstance(radius, (float, int)) or not math.isfinite(radius):
        raise ValueError("finite signed displacement radius required")
    fixed, variable = {}, {}
    for element, channels in coefficients.items():
        fixed[element] = [c.new_empty((c.shape[0], 0)) for c in channels]
        variable[element] = [(c + radius*d).detach().clone()
                             for c, d in zip(channels, direction[element])]
    _retract_variables(fixed, variable)
    return variable


def radial_guard_sensitivity(coefficients, report, guard, *, radii=(1e-4, 5e-5)):
    _validate_report(coefficients, report)
    if (len(radii) != 2 or radii[0] == radii[1] or any(isinstance(r, bool)
            or not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0 for r in radii)):
        raise ValueError("two distinct finite positive stencil radii required")
    rows = []
    with torch.no_grad():
        for channel in report["channels"]:
            element, l = channel["element"], channel["l"]
            for z in range(len(channel["radials"])):
                direction = radial_descent_direction(coefficients, report, element, l, z)
                row = dict(element=element, l=l, zeta=z+1,
                           status="zero_horizontal_gradient" if direction is None else "success", stencils=[])
                if direction is not None:
                    for radius in radii:
                        points = []
                        for sign in (-1., 1.):
                            trial = retract_displacement(coefficients, direction, sign*radius)
                            try:
                                point = guard(trial)
                                if not point.get("gate", False):
                                    raise CandidateGuardError("guard rejected stencil")
                                for key in ("occupied_band_sum_change_ev_per_atom",
                                            "maximum_target_band_change_ev", "minimum_gap_ev"):
                                    if not math.isfinite(point[key]):
                                        raise ValueError("guard stencil contains nonfinite metric")
                            except CandidateGuardError as error:
                                point = dict(gate=False, reason=str(error))
                            points.append(point)
                        derivative = None
                        if all(p["gate"] for p in points):
                            derivative = (points[1]["occupied_band_sum_change_ev_per_atom"]
                                          - points[0]["occupied_band_sum_change_ev_per_atom"])/(2*radius)
                        else:
                            row["status"] = "guard_rejected"
                        row["stencils"].append(dict(radius=radius, minus=points[0], plus=points[1],
                                                   band_sum_derivative_ev_per_atom=derivative))
                    if row["status"] == "success":
                        row["two_size_derivative_difference"] = abs(
                            row["stencils"][0]["band_sum_derivative_ev_per_atom"]
                            - row["stencils"][1]["band_sum_derivative_ev_per_atom"])
                rows.append(row)
    return dict(scope="frozen_band_sensitivity_not_actual_PBE_derivative",
                direction="unit_negative_horizontal_gradient_per_saved_radial", radials=rows,
                actual_pbe_sensitivity="unmeasured", physical_release_gate="hold")


def evaluate_radial_gradients(datasets, coefficients, *, occupied_capture_tolerance,
                              frequency_batch_size, weights):
    """One full response graph, two backward passes, and no optimizer update."""
    _validate(coefficients)
    started = time.perf_counter()
    datasets = tuple(prepare_periodic_occupied_reference(d) for d in datasets)
    reference = prepare_periodic_rpa_reference(datasets)
    variable = {e: [c.detach().clone().requires_grad_(c.shape[1] > 0) for c in channels]
                for e, channels in coefficients.items()}
    parameters = [c for channels in variable.values() for c in channels if c.shape[1]]
    responses, capture, condition = [], math.inf, 0.
    for dataset in datasets:
        result = evaluate_periodic_galerkin_coefficient_response(dataset, variable,
            contraction_backend="block", occupied_capture_tolerance=occupied_capture_tolerance,
            frequency_batch_size=frequency_batch_size)
        responses.append(result.response)
        capture = min(capture, result.minimum_occupied_capture)
        condition = max(condition, result.maximum_overlap_condition)
    objective = periodic_rpa_objective(datasets, tuple(responses), reference_cache=reference, **weights)
    forward_seconds = time.perf_counter()-started
    loss_gradients = torch.autograd.grad(objective.loss, parameters, retain_graph=True)
    energy_gradients = torch.autograd.grad(objective.candidate_energy_ha, parameters)

    def assemble(gradients):
        iterator = iter(gradients)
        return {e: [next(iterator).detach() if c.shape[1] else torch.empty_like(c)
                    for c in channels] for e, channels in variable.items()}

    rpa = {key: float(getattr(objective, key).detach()) for key in (
        "candidate_energy_ha", "reference_energy_ha", "pi_relative_squared_error",
        "trace_log_relative_squared_error", "energy_relative_squared_error")}
    if not all(math.isfinite(v) for v in rpa.values()) or not math.isfinite(float(objective.loss)):
        raise ValueError("nonfinite RPA diagnostic")
    rpa.update(complete_q_weight=objective.complete_q_weight, q_weight_coverage=objective.q_weight_coverage,
        per_q=[dict(selected_iq=r.selected_iq, q_weight=r.q_weight,
                    frequency_ha=r.frequency_ha.tolist(),
                    candidate_contributions_ha=r.candidate_contributions_ha.detach().tolist(),
                    reference_contributions_ha=r.reference_contributions_ha.detach().tolist())
               for r in objective.q_records])
    return dict(scope="radial_gradients_at_fixed_candidate_no_optimization",
        loss=float(objective.loss.detach()), rpa=rpa,
        minimum_occupied_capture=capture, maximum_overlap_condition=condition,
        loss_gradient=radial_gradient_report(coefficients, assemble(loss_gradients)),
        energy_gradient=radial_gradient_report(coefficients, assemble(energy_gradients)),
        energy_gradient_units="Ha_per_cell_per_unit_coefficient", forward_seconds=forward_seconds,
        forward_and_backward_seconds=time.perf_counter()-started, physical_release_gate="hold")
