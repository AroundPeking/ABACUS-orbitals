"""One bounded full-horizontal-Ec proposal, with no inferred PBE safety."""

import math
import torch

from periodic_galerkin_radial_diagnostics import (
    _validate_report, radial_gradient_report, retract_displacement,
)

SCOPE = 'accepted_ec_gradient_step_candidate'
RADIUS = .02


def _same_report(actual, expected):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_same_report(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_report(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, float):
        return (type(actual) is float and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14))
    return type(actual) is type(expected) and actual == expected


def propose_ec_gradient_step(coefficients, gradient_report, radius=RADIUS):
    if (isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not math.isfinite(radius) or radius != RADIUS):
        raise ValueError('only the approved radius 0.02 is eligible')
    _validate_report(coefficients, gradient_report)
    raw = {e: [torch.tensor(row['raw_gradient'], dtype=torch.float64)
               for row in gradient_report['channels'] if row['element'] == e]
           for e in coefficients}
    # JSON restores row-major tensors; original autograd tensors may be strided.
    if not _same_report(radial_gradient_report(coefficients, raw), gradient_report):
        raise ValueError('gradient report does not reproduce at this coefficient frame')
    norm = gradient_report['horizontal_gradient_norm']
    if not math.isfinite(norm) or norm <= 1e-14:
        raise ValueError('nonzero finite horizontal Ec gradient required')
    direction = {e: [-torch.tensor(row['horizontal_gradient'], dtype=torch.float64)/norm
                     for row in gradient_report['channels'] if row['element'] == e]
                 for e in coefficients}
    unit = sum(float((d*d).sum()) for channels in direction.values() for d in channels)
    if not math.isclose(unit, 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError('full Ec descent must have unit norm')
    return dict(scope=SCOPE, direction_name='negative_horizontal_ec_gradient', radius=RADIUS,
        coefficients=retract_displacement(coefficients, direction, RADIUS), direction=direction,
        predicted_ec_delta_ha_per_cell=-RADIUS*norm,
        actual_pbe_direction_derivative='unmeasured', finite_step_safety='unmeasured',
        actual_pbe_gate='pending', galerkin_energy='unmeasured', physical_release_gate='hold')
