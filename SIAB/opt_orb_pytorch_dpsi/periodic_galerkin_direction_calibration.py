"""Bounded true-PBE calibration in two frozen-response directions."""

import math

import torch

from periodic_galerkin_radial_diagnostics import (
    _validate_report, _validate, radial_descent_direction, retract_displacement,
)

SIGNED_RADII = (-.001, .001, -.002, .002)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("finite real scalar required")
    return float(value)


def _unit(v):
    norm = _finite(float(v.norm()))
    if norm <= 1e-14:
        raise ValueError("independent nonzero calibration directions required")
    return v/norm


def build_directions(coefficients, energy_report, sensitivity):
    """Use saved-frame horizontal Ec gradients, not orbital energy attribution."""
    _validate_report(coefficients, energy_report)
    if sensitivity.get("scope") != "frozen_band_sensitivity_not_actual_PBE_derivative":
        raise ValueError("frozen band derivative identity required")
    identities = [(c["element"], c["l"], r["zeta"])
                  for c in energy_report["channels"] for r in c["radials"]]
    rows = sensitivity["radials"]
    if [(r["element"], r["l"], r["zeta"]) for r in rows] != identities:
        raise ValueError("complete ordered radial sensitivities required")
    # Validate the persisted projection against both its frame and raw vector.
    for channel in energy_report["channels"]:
        c = coefficients[channel["element"]][channel["l"]]
        raw = torch.tensor(channel["raw_gradient"], dtype=torch.float64)
        h = torch.tensor(channel["horizontal_gradient"], dtype=torch.float64)
        if (raw.shape != c.shape or not bool(torch.isfinite(raw).all())
                or not torch.allclose(h, raw-c@(c.T@raw), atol=1e-12, rtol=1e-10)
                or not torch.allclose(c.T@h, torch.zeros((c.shape[1], c.shape[1]), dtype=c.dtype), atol=1e-10, rtol=0)):
            raise ValueError("saved gradient is not the horizontal projection")
    slopes, norms, basis = [], [], []
    for row, identity in zip(rows, identities):
        if row["status"] != "success" or len(row["stencils"]) != 2:
            raise ValueError("unresolved derivative cannot define a calibration direction")
        derivatives, radii = [], []
        for stencil in row["stencils"]:
            radius = _finite(stencil["radius"])
            if radius <= 0 or any(p.get("gate") is not True for p in (stencil["minus"], stencil["plus"])):
                raise ValueError("invalid frozen derivative stencil")
            slope = (_finite(stencil["plus"]["occupied_band_sum_change_ev_per_atom"])
                     - _finite(stencil["minus"]["occupied_band_sum_change_ev_per_atom"]))/ (2*radius)
            if not math.isclose(slope, _finite(stencil["band_sum_derivative_ev_per_atom"]), abs_tol=1e-12, rel_tol=1e-10):
                raise ValueError("stored frozen derivative mismatch")
            derivatives.append(slope)
            radii.append(radius)
        if radii[0] == radii[1] or abs(derivatives[0]-derivatives[1]) > 1e-6:
            raise ValueError("inconsistent two-radius frozen derivative")
        e, l, z = identity
        direction = radial_descent_direction(coefficients, energy_report, e, l, z-1)
        if direction is None:
            raise ValueError("zero radial gradient cannot define this full-radial calibration")
        basis.append(direction)
        channel = next(c for c in energy_report["channels"] if (c["element"], c["l"]) == (e, l))
        norms.append(channel["radials"][z-1]["horizontal_norm"])
        slopes.append(derivatives[radii.index(min(radii))])
    g, b = (torch.tensor(v, dtype=torch.float64) for v in (norms, slopes))
    n = _unit(b)
    t = _unit(g-n*torch.dot(n, g))
    directions = {}
    for name, weights in (("T", t), ("N", n)):
        d = {e: [sum((w*v[e][l] for w, v in zip(weights, basis)), torch.zeros_like(c))
                 for l, c in enumerate(channels)] for e, channels in coefficients.items()}
        directions[name] = dict(radial_weights=weights.tolist(),
            matrices={e: [c.tolist() for c in channels] for e, channels in d.items()},
            frozen_band_slope_ev_per_c=_finite(float(torch.dot(b, weights))),
            ec_slope_ha_per_cell=_finite(-float(torch.dot(g, weights))))
    if (abs(directions["T"]["frozen_band_slope_ev_per_c"]) > 1e-12
            or directions["T"]["ec_slope_ha_per_cell"] >= 0):
        raise ValueError("invalid frozen tangent descent")
    artifact = dict(status="success", scope="two_direction_actual_pbe_calibration",
        coordinate_metric="Euclidean_coefficient_signed_QR_frame",
        radials=[dict(element=e, l=l, zeta=z) for e, l, z in identities],
        radial_ec_descent_rates_ha_per_cell=norms, radial_frozen_band_slopes_ev_per_c=slopes,
        directions=directions, tangent_descent_fraction=-directions["T"]["ec_slope_ha_per_cell"]/float(g.norm()),
        signed_radii=list(SIGNED_RADII), actual_pbe_direction_derivatives="unmeasured",
        physical_release_gate="hold")
    for name in directions:
        read_direction(coefficients, artifact, name)
    return artifact


def read_direction(coefficients, artifact, name):
    if name not in ("T", "N") or set(artifact["directions"]) != {"T", "N"}:
        raise ValueError("exact T/N directions required")
    values = []
    for label in ("T", "N"):
        matrices = artifact["directions"][label]["matrices"]
        d = {e: [torch.tensor(c, dtype=torch.float64) for c in channels] for e, channels in matrices.items()}
        _validate(coefficients, d)
        for e, channels in coefficients.items():
            for c, v in zip(channels, d[e]):
                if not torch.allclose(c.T@v, torch.zeros((c.shape[1], c.shape[1]), dtype=c.dtype), atol=1e-10, rtol=0):
                    raise ValueError("direction is not horizontal in the saved frame")
        flat = torch.cat([v.flatten() for channels in d.values() for v in channels])
        if not math.isclose(_finite(float(flat.norm())), 1., abs_tol=1e-12, rel_tol=0):
            raise ValueError("direction is not unit length")
        values.append((d, flat))
    if abs(float(values[0][1]@values[1][1])) > 1e-12:
        raise ValueError("calibration directions must be orthogonal")
    return values[("T", "N").index(name)][0]


def signed_probes(coefficients, artifact):
    if artifact.get("signed_radii") != list(SIGNED_RADII):
        raise ValueError("only the eight approved probes may be exported")
    return [dict(direction_name=name, signed_radius=radius,
                 coefficients=retract_displacement(coefficients, read_direction(coefficients, artifact, name), radius))
            for name in ("T", "N") for radius in SIGNED_RADII]


def analyze_pbe_axes(center_energy_ev, samples):
    """Central SCF differences per C; axis data do not measure mixed curvature."""
    center = _finite(center_energy_ev)
    expected = [(name, r) for name in ("T", "N") for r in SIGNED_RADII]
    identities = [(s["direction_name"], s["signed_radius"]) for s in samples]
    if len(identities) != 8 or set(identities) != set(expected):
        raise ValueError("exact eight unique signed PBE probes required")
    energies = {}
    for sample, identity in zip(samples, identities):
        if sample.get("scf_log_gate") != "pass":
            raise ValueError("failed SCF is not a directional measurement")
        energies[identity] = _finite(sample["candidate_energy_ev"])
    axes, differences = {}, {}
    for name in ("T", "N"):
        axes[name] = []
        for r in (.001, .002):
            minus, plus = energies[(name, -r)], energies[(name, r)]
            axes[name].append(dict(radius=r,
                derivative_ev_per_c=(plus-minus)/(4*r),
                curvature_ev_per_c=((plus-center)+(minus-center))/(2*r*r)))
        a, b = axes[name]
        differences[name] = dict(derivative_abs_difference=abs(a["derivative_ev_per_c"]-b["derivative_ev_per_c"]),
            curvature_abs_difference=abs(a["curvature_ev_per_c"]-b["curvature_ev_per_c"]))
    # Explicit operational consistency, not a bound on finite combined steps.
    passed = all(differences[n]["derivative_abs_difference"] <= 1e-3+.05*max(abs(r["derivative_ev_per_c"]) for r in axes[n])
                 and differences[n]["curvature_abs_difference"] <= .05+.1*max(abs(r["curvature_ev_per_c"]) for r in axes[n])
                 for n in axes)
    return dict(status="success", scope="actual_pbe_direction_calibration_not_optimization",
        center_energy_ev=center, axes=axes, two_radius_differences=differences,
        consistency_gate="pass" if passed else "fail",
        consistency_definition="abs_d1<=0.001+5%max; abs_d2<=0.05+10%max; operational_only",
        mixed_curvature="unmeasured", combined_step_safety="unmeasured", physical_release_gate="hold")
