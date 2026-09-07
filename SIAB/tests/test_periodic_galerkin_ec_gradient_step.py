"""A saved Ec gradient proposes one step; it never establishes PBE safety."""

import copy
import importlib
import math
import unittest

import torch
import common  # noqa: F401
from periodic_galerkin_radial_diagnostics import radial_gradient_report, retract_displacement


class EcGradientStepTest(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('periodic_galerkin_ec_gradient_step')
        torch.manual_seed(177)
        self.c = {'C': [torch.linalg.qr(torch.randn(7, n, dtype=torch.float64))[0]
                        for n in (3, 3, 2, 0, 0)]}
        self.raw = {'C': [torch.randn_like(c) for c in self.c['C']]}
        self.report = radial_gradient_report(self.c, self.raw)

    def test_full_unit_descent_exact_retraction_and_no_parent_mutation(self):
        before = copy.deepcopy(self.c)
        step = self.m.propose_ec_gradient_step(self.c, self.report)
        self.assertEqual(step['scope'], 'accepted_ec_gradient_step_candidate')
        self.assertEqual(step['radius'], .02)
        self.assertEqual(step['physical_release_gate'], 'hold')
        self.assertEqual(step['actual_pbe_direction_derivative'], 'unmeasured')
        self.assertEqual(step['finite_step_safety'], 'unmeasured')
        self.assertEqual(step['actual_pbe_gate'], 'pending')
        self.assertEqual(step['galerkin_energy'], 'unmeasured')
        norm = self.report['horizontal_gradient_norm']
        self.assertAlmostEqual(sum(float((d*d).sum()) for d in step['direction']['C']), 1., places=12)
        exact = retract_displacement(self.c, step['direction'], .02)
        for l, c in enumerate(self.c['C']):
            self.assertTrue(torch.equal(c, before['C'][l]))
            self.assertTrue(torch.equal(step['coefficients']['C'][l], exact['C'][l]))
            torch.testing.assert_close(c.T @ step['direction']['C'][l],
                torch.zeros((c.shape[1], c.shape[1]), dtype=c.dtype), atol=1e-12, rtol=0)
            if c.shape[1]:
                h = torch.tensor(self.report['channels'][l]['horizontal_gradient'], dtype=c.dtype)
                self.assertTrue(torch.equal(step['direction']['C'][l], -h/norm))
        self.assertEqual(step['predicted_ec_delta_ha_per_cell'], -.02*norm)

    def test_only_one_radius_is_eligible(self):
        for radius in (True, .01, -.02, 0, math.nan, math.inf):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.m.propose_ec_gradient_step(self.c, self.report, radius)

    def test_one_explicit_ten_percent_reduction_not_an_automatic_search(self):
        step = self.m.propose_ec_gradient_step(self.c, self.report, radius=.018)
        self.assertEqual(step['radius'], .018)
        self.assertEqual(step['predicted_ec_delta_ha_per_cell'], -.018*self.report['horizontal_gradient_norm'])
        self.assertEqual(step['actual_pbe_gate'], 'pending')
        for radius in (.019, .017, .021):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.m.propose_ec_gradient_step(self.c, self.report, radius=radius)

    def test_changed_gradient_or_summary_is_rejected(self):
        mutations = [lambda r: r.update(horizontal_gradient_norm=1.),
            lambda r: r['channels'][0]['raw_gradient'][0].__setitem__(0, 12.),
            lambda r: r['channels'][0]['horizontal_gradient'][0].__setitem__(0, 12.),
            lambda r: r['channels'].pop(),
            lambda r: r.update(projection='raw_gradient'),
            lambda r: r['channels'][0].update(horizontal_norm=math.nan)]
        for mutate in mutations:
            r = copy.deepcopy(self.report)
            mutate(r)
            with self.assertRaises(ValueError):
                self.m.propose_ec_gradient_step(self.c, r)

    def test_zero_horizontal_gradient_is_not_a_step(self):
        zero = {'C': [torch.zeros_like(c) for c in self.c['C']]}
        with self.assertRaises(ValueError):
            self.m.propose_ec_gradient_step(self.c, radial_gradient_report(self.c, zero))

    def test_changed_center_cannot_reuse_gradient(self):
        changed = copy.deepcopy(self.c)
        changed['C'][0] = torch.linalg.qr(torch.randn_like(changed['C'][0]))[0]
        with self.assertRaises(ValueError):
            self.m.propose_ec_gradient_step(changed, self.report)


if __name__ == '__main__':
    unittest.main()
