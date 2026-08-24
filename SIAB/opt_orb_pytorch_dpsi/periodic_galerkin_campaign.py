"""Physical capacity gates for periodic Galerkin Sternheimer basis fitting."""

import math

from periodic_galerkin_basis import build_primitive_to_candidate
from periodic_galerkin_sternheimer import (
    evaluate_periodic_galerkin_mother_response,
    evaluate_periodic_galerkin_response,
)


def _result_metrics(result):
    return {
        "relative_pi_error": float(result.relative_response_error.detach()),
        "relative_dpsi_projection_error": float(
            result.relative_projection_error.detach()
        ),
        "minimum_occupied_capture": result.minimum_occupied_capture,
        "maximum_overlap_condition": result.maximum_overlap_condition,
        "minimum_effective_rank": result.minimum_candidate_rank,
    }


def _coefficient_counts(coefficients):
    return {
        element: [int(channel.shape[1]) for channel in channels]
        for element, channels in coefficients.items()
    }


def evaluate_periodic_basis_capacity(
    dataset,
    coefficients,
    *,
    mother_response_tolerance=1.0e-3,
    relative_rank_tolerance=1.0e-12,
    condition_limit=1.0e12,
    occupied_capture_tolerance=1.0e-6,
):
    """Compare one candidate basis and the full Bessel mother space to exact Pi.

    The mother-space gate is deliberately separate from final basis acceptance.
    It answers whether optimization in the exported primitive space is meaningful;
    final acceptance still requires independent SOS and Delta-ST energy checks.
    """
    if (
        not isinstance(mother_response_tolerance, (int, float))
        or isinstance(mother_response_tolerance, bool)
        or not math.isfinite(mother_response_tolerance)
        or mother_response_tolerance <= 0.0
    ):
        raise ValueError("mother_response_tolerance must be finite and positive")

    candidate_basis = build_primitive_to_candidate(
        dataset.primitive_blocks,
        dataset.primitive_count,
        coefficients,
    )
    mother_result = evaluate_periodic_galerkin_mother_response(
        dataset,
        relative_rank_tolerance=relative_rank_tolerance,
        condition_limit=condition_limit,
        occupied_capture_tolerance=occupied_capture_tolerance,
    )

    candidate = {
        "ao_count": int(candidate_basis.transform.shape[1]),
        "nu": _coefficient_counts(coefficients),
    }
    try:
        candidate_result = evaluate_periodic_galerkin_response(
            dataset,
            candidate_basis.transform,
            relative_rank_tolerance=relative_rank_tolerance,
            condition_limit=condition_limit,
            occupied_capture_tolerance=occupied_capture_tolerance,
        )
    except RuntimeError as error:
        candidate.update(
            {
                "evaluation_gate": "FAIL",
                "error": str(error),
            }
        )
    else:
        candidate.update(_result_metrics(candidate_result))
        candidate["evaluation_gate"] = "PASS"
    mother = _result_metrics(mother_result)
    mother["primitive_count"] = dataset.primitive_count
    mother["capacity_tolerance"] = float(mother_response_tolerance)
    mother["capacity_gate"] = (
        "PASS"
        if mother["relative_pi_error"] <= mother_response_tolerance
        else "FAIL"
    )
    return {
        "scope": (
            "periodic Galerkin capacity gate; not an independent SOS or "
            "Delta-ST energy validation"
        ),
        "physics_hash": dataset.physics_hash,
        "selected_iq": dataset.selected_iq,
        "qpoint": list(dataset.qpoint),
        "frequency_ha": [float(value) for value in dataset.frequency_ha],
        "candidate": candidate,
        "mother": mother,
        "optimization_allowed": mother["capacity_gate"] == "PASS",
    }
