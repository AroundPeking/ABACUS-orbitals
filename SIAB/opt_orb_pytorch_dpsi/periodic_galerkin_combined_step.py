"""One measured first-order PBE-tangent proposal in the existing T/N plane."""

import math

from periodic_galerkin_direction_calibration import _finite, analyze_pbe_axes, read_direction
from periodic_galerkin_radial_diagnostics import retract_displacement

RADIUS = .02
SCOPE = 'actual_pbe_tangent_candidate'


def combine_pbe_tangent(coefficients, directions, calibration, radius=RADIUS):
    if _finite(radius) != RADIUS:
        raise ValueError('only the single approved radius 0.02 is eligible')
    expected = dict(status='success', scope='actual_pbe_direction_calibration_not_optimization',
                    actual_scf_count=8, consistency_gate='pass', physical_release_gate='hold')
    if any(calibration.get(k) != v for k, v in expected.items()):
        raise ValueError('accepted eight-sample PBE calibration required')
    verified = analyze_pbe_axes(calibration['center_energy_ev'], calibration['samples'])
    if verified['consistency_gate'] != 'pass' or any(calibration.get(k) != v for k, v in verified.items()):
        raise ValueError('stored PBE calibration disagrees with its actual samples')
    a_t, a_n = (verified['axes'][name][0]['derivative_ev_per_c'] for name in ('T', 'N'))
    if abs(a_n) <= 1e-10:
        raise ValueError('actual PBE normal is numerically unresolved')
    ratio = _finite(a_t/a_n)
    norm = math.hypot(1., ratio)
    t, n = (read_direction(coefficients, directions, name) for name in ('T', 'N'))
    d = {e: [(u-ratio*v)/norm for u, v in zip(t[e], n[e])] for e in coefficients}
    slope = (_finite(directions['directions']['T']['ec_slope_ha_per_cell'])
             - ratio*_finite(directions['directions']['N']['ec_slope_ha_per_cell']))/norm
    if not math.isfinite(slope) or slope >= 0:
        raise ValueError('PBE-corrected direction must predict Ec descent')
    return dict(scope=SCOPE, direction_name='actual_pbe_tangent', radius=RADIUS,
        mixing_ratio=ratio, direction=d,
        coefficients=retract_displacement(coefficients, d, RADIUS),
        predicted_ec_delta_ha_per_cell=RADIUS*slope,
        predicted_linear_pbe_delta_ev_per_c=RADIUS*(a_t-ratio*a_n)/norm,
        radius_to_outer_calibration_ratio=RADIUS/.002,
        mixed_curvature='unmeasured', finite_step_safety='unmeasured',
        actual_pbe_gate='pending', galerkin_energy='unmeasured', physical_release_gate='hold')
