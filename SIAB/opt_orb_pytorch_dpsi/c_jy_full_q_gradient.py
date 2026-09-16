"""Direct fixed-rank jY Ec gradients and deterministic full-q reduction."""

from dataclasses import replace
import math

import torch

from c_jy_compressed_evaluation import validate_profile_specs
from periodic_galerkin_radial_diagnostics import (
    _validate_report,
    radial_gradient_report,
    retract_displacement,
)


PROFILE = (4, 4, 3, 2, 0)
# Flattened selected_iq values in the frozen 4x4x4 cache.  These are not the
# human-facing q-star labels (1, 2, 3, 6, 7, 8, 11, 28).
INDICES = (1, 22, 43, 6, 27, 23, 11, 55)
MULTIPLICITIES = (1, 8, 4, 6, 24, 12, 3, 6)
SCOPE = "compressed_shared_radial_q_energy_gradient"
FULL_Q_SCOPE = "compressed_shared_radial_full_q_energy_gradient"


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite " + name)
    return value


def _coefficient_sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("exact coefficient SHA256 required")
    return value


def _q_slot(selected_iq):
    try:
        return INDICES.index(selected_iq)
    except ValueError as error:
        raise ValueError("unexpected q representative") from error


def evaluate_q_energy_gradient(
        dataset, spec, *, read_coefficients, evaluate_response,
        evaluate_objective, radial_gradient_report=radial_gradient_report,
        prepare_block_cache=None, relative_rank_tolerance=1e-10,
        occupied_capture_floor=.99999, radial_rows=48):
    """Evaluate one star-weighted q contribution and its raw Ec gradient."""
    spec = validate_profile_specs([spec], profiles=(PROFILE,))[0]
    if (type(radial_rows) is not int or radial_rows <= 31
            or int(dataset.frequency_ha.numel()) != 12
            or not math.isfinite(occupied_capture_floor)
            or not 0 < occupied_capture_floor <= 1):
        raise ValueError("expanded mother and 12-frequency controls required")
    coefficients = read_coefficients(
        spec["coefficients_path"], element="C", radial_rows=radial_rows,
        expected_nu=PROFILE)
    if prepare_block_cache is not None:
        dataset = replace(dataset, kpoints=tuple(
            prepare_block_cache(record, dataset.primitive_blocks, coefficients)
            for record in dataset.kpoints))
    variable = {
        element: [value.detach().clone().requires_grad_(value.shape[1] > 0)
                  for value in channels]
        for element, channels in coefficients.items()
    }
    parameters = [value for channels in variable.values()
                  for value in channels if value.shape[1]]
    response = evaluate_response(
        dataset, variable, contraction_backend="block",
        relative_rank_tolerance=relative_rank_tolerance,
        condition_limit=1e12,
        occupied_capture_tolerance=max(1e-15, 1.-occupied_capture_floor),
        frequency_batch_size=12)
    objective = evaluate_objective(
        (dataset,), (response.response,), pi_weight=1.,
        trace_log_weight=1., energy_weight=0.)
    gradients = iter(torch.autograd.grad(
        objective.candidate_energy_ha, parameters))
    raw = {
        element: [next(gradients).detach() if value.shape[1]
                  else torch.empty_like(value)
                  for value in channels]
        for element, channels in variable.items()
    }
    try:
        next(gradients)
    except StopIteration:
        pass
    else:
        raise ValueError("unconsumed coefficient gradient")
    report = radial_gradient_report(coefficients, raw)
    record = objective.q_records[0]
    candidate = _finite(objective.candidate_energy_ha.detach(),
                        "candidate q energy")
    reference = _finite(objective.reference_energy_ha.detach(),
                        "reference q energy")
    candidate_contributions = [
        _finite(value, "candidate frequency contribution")
        for value in record.candidate_contributions_ha.detach().tolist()]
    reference_contributions = [
        _finite(value, "reference frequency contribution")
        for value in record.reference_contributions_ha.detach().tolist()]
    if (len(candidate_contributions) != 12
            or len(reference_contributions) != 12
            or not math.isclose(math.fsum(candidate_contributions), candidate,
                                rel_tol=0, abs_tol=1e-12)
            or not math.isclose(math.fsum(reference_contributions), reference,
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError("q energy does not equal its frequency contributions")
    capture = _finite(response.minimum_occupied_capture,
                      "occupied capture")
    if capture < occupied_capture_floor:
        raise ValueError("occupied capture below floor")
    selected_iq = int(dataset.selected_iq)
    slot = _q_slot(selected_iq)
    q_weight = _finite(dataset.q_weight, "q weight")
    if not math.isclose(q_weight, MULTIPLICITIES[slot]/64,
                        rel_tol=0, abs_tol=1e-15):
        raise ValueError("q-star weight mismatch")
    return dict(
        status="success", scope=SCOPE, q_slot=slot,
        selected_iq=selected_iq, q_weight=q_weight,
        frequency_count=12, profile=list(PROFILE), ao_per_C=45,
        radial_rows=radial_rows,
        coefficients_sha256=spec["coefficients_sha256"],
        candidate_energy_ha=candidate, reference_energy_ha=reference,
        candidate_contributions_ha=candidate_contributions,
        reference_contributions_ha=reference_contributions,
        minimum_occupied_capture=capture,
        maximum_overlap_condition=_finite(
            response.maximum_overlap_condition, "overlap condition"),
        minimum_candidate_rank=int(response.minimum_candidate_rank),
        expanded_mother_anchor_gate="pass",
        energy_gradient=report, complete_q_weight=False,
        physical_release_gate="hold")


def _artifact_raw_gradient(artifact, coefficients):
    report = artifact.get("energy_gradient")
    _validate_report(coefficients, report)
    raw = {element: [
        torch.tensor(row["raw_gradient"], dtype=torch.float64)
        for row in report["channels"] if row["element"] == element]
        for element in coefficients}
    if radial_gradient_report(coefficients, raw) != report:
        raise ValueError("q gradient report does not reproduce")
    return raw


def reduce_full_q_energy_gradients(
        artifacts, coefficients, *, coefficient_sha256,
        source_radial_rows=31, radial_rows=48):
    """Sum eight star-weighted Ec derivatives in one coefficient frame."""
    coefficient_sha256 = _coefficient_sha256(coefficient_sha256)
    if (not isinstance(artifacts, (list, tuple)) or len(artifacts) != 8
            or type(source_radial_rows) is not int
            or type(radial_rows) is not int
            or not 0 < source_radial_rows < radial_rows):
        raise ValueError("complete eight-q expanded gradient set required")
    by_slot = {}
    for artifact in artifacts:
        slot = artifact.get("q_slot")
        if type(slot) is not int or slot not in range(8) or slot in by_slot:
            raise ValueError("complete eight-q unique slots required")
        by_slot[slot] = artifact
    if set(by_slot) != set(range(8)):
        raise ValueError("complete eight-q unique slots required")
    total_raw = {element: [torch.zeros_like(value) for value in channels]
                 for element, channels in coefficients.items()}
    candidate, reference, per_q = [], [], []
    minimum_capture, maximum_condition, minimum_rank = math.inf, 0., None
    for slot in range(8):
        artifact = by_slot[slot]
        expected = dict(
            status="success", scope=FULL_Q_SCOPE, q_slot=slot,
            selected_iq=INDICES[slot], frequency_count=12,
            profile=list(PROFILE), ao_per_C=45, radial_rows=radial_rows,
            coefficients_sha256=coefficient_sha256,
            expanded_mother_anchor_gate="pass",
            physical_release_gate="hold")
        if any(artifact.get(key) != value for key, value in expected.items()):
            if artifact.get("coefficients_sha256") != coefficient_sha256:
                raise ValueError("mixed coefficient gradients")
            raise ValueError("invalid q gradient artifact")
        weight = _finite(artifact.get("q_weight"), "q weight")
        if not math.isclose(weight, MULTIPLICITIES[slot]/64,
                            rel_tol=0, abs_tol=1e-15):
            raise ValueError("q-star weight mismatch")
        raw = _artifact_raw_gradient(artifact, coefficients)
        for element, channels in raw.items():
            for target, value in zip(total_raw[element], channels):
                target.add_(value)
        candidate.append(_finite(artifact.get("candidate_energy_ha"),
                                 "candidate q energy"))
        reference.append(_finite(artifact.get("reference_energy_ha"),
                                 "reference q energy"))
        capture = _finite(artifact.get("minimum_occupied_capture"),
                          "occupied capture")
        condition = _finite(artifact.get("maximum_overlap_condition"),
                            "overlap condition")
        rank = artifact.get("minimum_candidate_rank")
        if type(rank) is not int or rank <= 0:
            raise ValueError("invalid candidate rank")
        minimum_capture = min(minimum_capture, capture)
        maximum_condition = max(maximum_condition, condition)
        minimum_rank = rank if minimum_rank is None else min(minimum_rank, rank)
        per_q.append(dict(q_slot=slot, selected_iq=INDICES[slot],
                          multiplicity=MULTIPLICITIES[slot], q_weight=weight,
                          candidate_energy_ha=candidate[-1],
                          reference_energy_ha=reference[-1]))
    report = radial_gradient_report(coefficients, total_raw)
    total_squared = report["horizontal_gradient_norm"]**2
    if not math.isfinite(total_squared) or total_squared <= 1e-28:
        raise ValueError("full-q horizontal Ec gradient is unresolved")
    new_squared = 0.
    per_channel = []
    for row in report["channels"]:
        value = torch.tensor(row["horizontal_gradient"], dtype=torch.float64)
        if value.shape[0] != radial_rows:
            raise ValueError("gradient radial row count mismatch")
        channel_squared = float((value*value).sum())
        channel_new_squared = float(
            (value[source_radial_rows:]*value[source_radial_rows:]).sum())
        new_squared += channel_new_squared
        per_channel.append(dict(
            element=row["element"], l=row["l"],
            horizontal_gradient_squared=channel_squared,
            new_row_horizontal_gradient_squared=channel_new_squared,
            new_row_squared_fraction=(channel_new_squared/channel_squared
                                      if channel_squared else 0.)))
    return dict(
        status="success", scope="compressed_shared_radial_full_q_energy_gradient",
        profile=list(PROFILE), ao_per_C=45, radial_rows=radial_rows,
        source_radial_rows=source_radial_rows,
        coefficients_sha256=coefficient_sha256,
        candidate_energy_ha=math.fsum(candidate),
        reference_energy_ha=math.fsum(reference),
        q_weight_coverage=math.fsum(row["q_weight"] for row in per_q),
        complete_q_weight=True, per_q=per_q,
        minimum_occupied_capture=minimum_capture,
        maximum_overlap_condition=maximum_condition,
        minimum_candidate_rank=minimum_rank,
        energy_gradient=report,
        new_row_horizontal_gradient_squared_fraction=new_squared/total_squared,
        per_channel_new_row_gradient=per_channel,
        physical_release_gate="hold")


def propose_full_q_ec_step(coefficients, gradient_report, *, radius):
    """Retract one bounded negative full-q Ec gradient step at fixed rank."""
    if (isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not math.isfinite(radius) or not 0 < radius <= .05):
        raise ValueError("Ec trust radius must be in (0, 0.05]")
    _validate_report(coefficients, gradient_report)
    norm = _finite(gradient_report.get("horizontal_gradient_norm"),
                   "horizontal Ec gradient norm")
    if norm <= 1e-14:
        raise ValueError("nonzero horizontal Ec gradient required")
    direction = {
        element: [-torch.tensor(row["horizontal_gradient"],
                                dtype=torch.float64)/norm
                  for row in gradient_report["channels"]
                  if row["element"] == element]
        for element in coefficients}
    unit = math.fsum(float((value*value).sum())
                     for channels in direction.values() for value in channels)
    if not math.isclose(unit, 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError("full-q Ec descent direction is not normalized")
    trial = retract_displacement(coefficients, direction, radius)
    displacement = math.sqrt(math.fsum(
        float(((new-old)*(new-old)).sum())
        for element in coefficients
        for old, new in zip(coefficients[element], trial[element])))
    return dict(
        status="prepared", scope="expanded_45ao_full_q_ec_step",
        direction_name="negative_full_q_horizontal_ec_gradient",
        radius=float(radius), coefficients=trial, direction=direction,
        coefficient_displacement_norm=displacement,
        predicted_ec_delta_ha_per_cell=-float(radius)*norm,
        actual_full_q_energy="pending", physical_release_gate="hold")
