"""Two bounded axes inside a measured maximum-band linearized tangent."""

import math
import torch

from periodic_galerkin_radial_diagnostics import (
    _validate_report, radial_gradient_report, radial_descent_direction,
)
from periodic_galerkin_direction_calibration import read_direction, SIGNED_RADII, _unit, _finite
from periodic_galerkin_ec_gradient_step import _same_report


def build_band_tangent(coefficients, energy_report, rows):
    _validate_report(coefficients,energy_report)
    raw={e:[torch.tensor(c['raw_gradient'],dtype=torch.float64)
            for c in energy_report['channels'] if c['element']==e] for e in coefficients}
    if not _same_report(radial_gradient_report(coefficients,raw),energy_report):
        raise ValueError('raw gradient and saved frame do not reconstruct')
    identities=[(c['element'],c['l'],r['zeta']) for c in energy_report['channels'] for r in c['radials']]
    if [(r['element'],r['l'],r['zeta']) for r in rows]!=identities:
        raise ValueError('ordered complete radial sensitivities required')
    for r in rows:
        for key,tolerance in (('occupied_two_radius_difference',1e-6),('target_two_radius_difference',1e-5)):
            if not 0<=_finite(r[key])<=tolerance:
                raise ValueError('unresolved two-radius derivative')
    a=torch.tensor([_finite(r['occupied_slope_ev_per_c']) for r in rows],dtype=torch.float64)
    b=torch.tensor([_finite(r['target_slope_ev']) for r in rows],dtype=torch.float64)
    g=torch.tensor([r['horizontal_norm'] for c in energy_report['channels'] for r in c['radials']],dtype=torch.float64)
    bhat=_unit(b)
    n=_unit(a-bhat*torch.dot(bhat,a))
    t=_unit(g-bhat*torch.dot(bhat,g)-n*torch.dot(n,g))
    basis=[radial_descent_direction(coefficients,energy_report,e,l,z-1) for e,l,z in identities]
    if any(v is None for v in basis): raise ValueError('resolved eight-radial directions required')
    directions={}
    for name,weights in (('T',t),('N',n)):
        d={e:[sum((w*v[e][l] for w,v in zip(weights,basis)),torch.zeros_like(c))
              for l,c in enumerate(channels)] for e,channels in coefficients.items()}
        directions[name]=dict(radial_weights=weights.tolist(),matrices={e:[c.tolist() for c in cs] for e,cs in d.items()},
            frozen_band_slope_ev_per_c=_finite(float(a@weights)),
            maximum_band_slope_ev=_finite(float(b@weights)),ec_slope_ha_per_cell=_finite(-float(g@weights)))
        if abs(directions[name]['maximum_band_slope_ev'])>1e-12:
            raise ValueError('maximum-band tangent construction failed')
    if abs(directions['T']['frozen_band_slope_ev_per_c'])>1e-12 or directions['T']['ec_slope_ha_per_cell']>=0:
        raise ValueError('occupied-sum tangent must descend in Ec')
    artifact=dict(status='success',scope='two_direction_band_tangent_pbe_calibration',
        coordinate_metric='Euclidean_coefficient_signed_QR_frame',
        radials=[dict(element=e,l=l,zeta=z) for e,l,z in identities],
        radial_ec_descent_rates_ha_per_cell=g.tolist(),radial_frozen_band_slopes_ev_per_c=a.tolist(),
        radial_maximum_band_slopes_ev=b.tolist(),directions=directions,
        tangent_descent_fraction=-directions['T']['ec_slope_ha_per_cell']/float(g.norm()),
        signed_radii=list(SIGNED_RADII),actual_pbe_direction_derivatives='unmeasured',
        maximum_band_derivative_scope='measured_finite_difference_proposal_not_smoothness_proof',
        physical_release_gate='hold')
    for name in ('T','N'): read_direction(coefficients,artifact,name)
    return artifact
