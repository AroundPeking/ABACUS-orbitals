"""A measured PBE tangent is a proposal, not finite-step acceptance."""

import copy
import importlib
import importlib.util
import math
import unittest

import torch
import common  # noqa: F401
from test_periodic_galerkin_direction_calibration import DirectionCalibrationTest


class CombinedStepTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('periodic_galerkin_combined_step'),
                             'combined-step module missing')
        self.m = importlib.import_module('periodic_galerkin_combined_step')
        fixture = DirectionCalibrationTest()
        fixture.setUp()
        self.c, self.artifact = fixture.c, fixture.build()
        center = -309.854
        samples = [dict(direction_name=name, signed_radius=r, scf_log_gate='pass',
                        candidate_energy_ev=center+2*(slope*r+.5*curvature*r*r))
                   for name, slope, curvature in (('T', .075, .9), ('N', .727, 96.))
                   for r in (-.001, .001, -.002, .002)]
        self.calibration = fixture.m.analyze_pbe_axes(center, samples)
        self.calibration.update(samples=samples, actual_scf_count=8)

    def test_corrected_tangent_is_unit_horizontal_and_keeps_parent(self):
        before = copy.deepcopy(self.c)
        trial = self.m.combine_pbe_tangent(self.c, self.artifact, self.calibration)
        ratio = trial['mixing_ratio']
        self.assertAlmostEqual(ratio, .075/.727, places=7)
        self.assertEqual(trial['radius'], .02)
        self.assertEqual(trial['physical_release_gate'], 'hold')
        self.assertEqual(trial['actual_pbe_gate'], 'pending')
        self.assertEqual(trial['galerkin_energy'], 'unmeasured')
        self.assertEqual(trial['mixed_curvature'], 'unmeasured')
        self.assertEqual(trial['radius_to_outer_calibration_ratio'], 10.)
        self.assertAlmostEqual(trial['predicted_linear_pbe_delta_ev_per_c'], 0., places=12)
        direction = trial['direction']['C']
        self.assertAlmostEqual(sum(float((d*d).sum()) for d in direction), 1., places=12)
        for l, c in enumerate(self.c['C']):
            torch.testing.assert_close(c, before['C'][l], atol=0, rtol=0)
            torch.testing.assert_close(c.T@direction[l], torch.zeros((c.shape[1], c.shape[1]), dtype=c.dtype), atol=1e-12, rtol=0)
            new = trial['coefficients']['C'][l]
            torch.testing.assert_close(new.T@new, torch.eye(c.shape[1], dtype=c.dtype), atol=1e-12, rtol=0)
        slope = (self.artifact['directions']['T']['ec_slope_ha_per_cell']
                 - ratio*self.artifact['directions']['N']['ec_slope_ha_per_cell'])/math.sqrt(1+ratio*ratio)
        self.assertAlmostEqual(trial['predicted_ec_delta_ha_per_cell'], .02*slope, places=12)
        self.assertLess(trial['predicted_ec_delta_ha_per_cell'], 0.)

    def test_wrong_radius_inconsistent_or_false_calibration_rejected(self):
        for radius in (True, 0., -.02, .001, .021, math.nan):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.m.combine_pbe_tangent(self.c, self.artifact, self.calibration, radius=radius)
        for field, value in (('consistency_gate', 'fail'), ('actual_scf_count', 7),
                             ('physical_release_gate', 'pass')):
            changed = dict(self.calibration, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.m.combine_pbe_tangent(self.c, self.artifact, changed)
        for mutation in (lambda x: x['samples'][0].update(candidate_energy_ev=-309.8),
                         lambda x: x['axes']['T'][0].update(derivative_ev_per_c=0.),
                         lambda x: x['samples'].pop()):
            changed = copy.deepcopy(self.calibration)
            mutation(changed)
            with self.assertRaises(ValueError):
                self.m.combine_pbe_tangent(self.c, self.artifact, changed)

    def test_degenerate_normal_or_ascent_is_not_an_eligible_proposal(self):
        cal = copy.deepcopy(self.calibration)
        for s in cal['samples']:
            if s['direction_name'] == 'N':
                s['candidate_energy_ev'] = cal['center_energy_ev']
        from periodic_galerkin_direction_calibration import analyze_pbe_axes
        cal.update(analyze_pbe_axes(cal['center_energy_ev'], cal['samples']))
        with self.assertRaises(ValueError):
            self.m.combine_pbe_tangent(self.c, self.artifact, cal)
        art = copy.deepcopy(self.artifact)
        art['directions']['T']['ec_slope_ha_per_cell'] = 1.
        with self.assertRaises(ValueError):
            self.m.combine_pbe_tangent(self.c, art, self.calibration)


if __name__ == '__main__':
    unittest.main()
