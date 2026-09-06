"""Opt-in, detached residuals in each q's frozen Coulomb-whitened Pi space.

This module does not alter or hook into the objective, gradient or CLI. Its
records compare a candidate only to its supplied frozen reference, not to a
mother-space result or an array from another PCA space.
"""

import hashlib
import math

import numpy as np
import torch

from periodic_galerkin_rpa import periodic_rpa_objective


_SCOPE = "candidate_minus_frozen_reference_not_mother_or_cross_pca"
_METRIC_DEFINITION = (
    "Coulomb-whitened Pi Frobenius residual in each q's own frozen auxiliary "
    "space: residual_squared_norm[q,iw] = sum_ab abs(Pi[q,iw,a,b] - "
    "Pi_reference[q,iw,a,b])**2; reference_squared_norm is sum_ab "
    "abs(Pi_reference)**2. Pi already includes k weights and occupations. "
    "Apply q_weight and frequency_weights_ha exactly once, with no partial-q "
    "renormalization and no 1/(2*pi) energy prefactor. The global squared "
    "relative error is sum_q,iw(q_weight * frequency_weight * residual norm) "
    "/ sum_q,iw(q_weight * frequency_weight * reference norm). "
    "relative_pi_error is the unweighted per-frequency square root of the "
    "local ratio: null for zero reference with nonzero residual, zero when "
    "both vanish. No cross-q or cross-PCA array subtraction is performed."
)
_WHITENING_SHA256_DEFINITION = (
    "SHA-256 of dataset.coulomb_whitening detached to CPU, with conjugate "
    "and negative view bits resolved, converted to complex128 and encoded "
    "as contiguous C-order little-endian complex128 (<c16) bytes. "
    "No shape header or numerical rounding; signed zeros are preserved. "
    "This identifies the supplied frame, not an invariant PCA subspace."
)


def _finite_values(value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError("matrix diagnostic norm is not finite")
    return value.tolist()


def _whitening_sha256(whitening):
    array = (
        whitening.detach()
        .cpu()
        .to(torch.complex128)
        .resolve_conj()
        .resolve_neg()
        .contiguous()
        .numpy()
    )
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<c16"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@torch.no_grad()
def periodic_rpa_matrix_diagnostics(datasets, responses):
    """Return plain JSON-safe matrix residuals without retaining autograd data.

    Inputs follow periodic_rpa_objective: matching nonempty tuples, with
    each response in its dataset's declared whitened space. The existing
    objective invokes _validate_inputs and rpa_trace_log for shared-protocol,
    same-space, finite, Hermitian and causality checks, including its global
    nonzero-reference requirement. Different q points may have different ranks.

    per_q preserves input order. largest_weighted_residual_finite_q is the
    complete per-q record maximizing integrated pi_numerator among actual
    nonzero dataset.qpoint vectors, with lower selected_iq breaking exact ties;
    it is None when no such q is present. The supplied coordinate convention
    is used exactly, without an iq-based Gamma assumption or q extrapolation.
    """
    objective = periodic_rpa_objective(datasets, responses)
    records = []
    finite_q_records = []
    for dataset, response in zip(datasets, responses):
        reference = dataset.reference_response.detach()
        residual = (response.detach() - reference).abs().square().sum(dim=(-2, -1))
        reference_norm = reference.abs().square().sum(dim=(-2, -1))
        weights = dataset.q_weight * dataset.frequency_weights_ha.detach()
        weighted_residual = weights * residual
        weighted_reference = weights * reference_norm
        residual_values = _finite_values(residual)
        reference_values = _finite_values(reference_norm)
        relative = []
        for numerator, denominator in zip(residual_values, reference_values):
            if denominator == 0.0:
                relative.append(0.0 if numerator == 0.0 else None)
            else:
                # Taking roots first avoids overflow in an otherwise finite ratio.
                error = math.sqrt(numerator) / math.sqrt(denominator)
                if not math.isfinite(error):
                    raise ValueError("matrix diagnostic relative error is not finite")
                relative.append(error)
        record = {
            "selected_iq": int(dataset.selected_iq),
            "q_weight": float(dataset.q_weight),
            "physics_hash": str(dataset.physics_hash),
            "whitening_sha256": _whitening_sha256(dataset.coulomb_whitening),
            "frequency_ha": dataset.frequency_ha.detach().tolist(),
            "frequency_weights_ha": dataset.frequency_weights_ha.detach().tolist(),
            "residual_squared_norm": residual_values,
            "reference_squared_norm": reference_values,
            "weighted_residual_squared_norm": _finite_values(weighted_residual),
            "weighted_reference_squared_norm": _finite_values(weighted_reference),
            "relative_pi_error": relative,
            "pi_numerator": _finite_values(weighted_residual.sum()),
            "pi_denominator": _finite_values(weighted_reference.sum()),
        }
        records.append(record)
        if any(coordinate != 0.0 for coordinate in dataset.qpoint):
            finite_q_records.append(record)
    return {
        "scope": _SCOPE,
        "metric_definition": _METRIC_DEFINITION,
        "whitening_sha256_definition": _WHITENING_SHA256_DEFINITION,
        "pi_numerator": math.fsum(record["pi_numerator"] for record in records),
        "pi_denominator": math.fsum(record["pi_denominator"] for record in records),
        "pi_relative_squared_error": float(objective.pi_relative_squared_error),
        "per_q": records,
        "largest_weighted_residual_finite_q": max(
            finite_q_records,
            key=lambda record: (record["pi_numerator"], -record["selected_iq"]),
            default=None,
        ),
    }
