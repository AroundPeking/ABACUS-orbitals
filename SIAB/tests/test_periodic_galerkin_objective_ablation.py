"""Explicit diagnostic ablations must not silently weaken normal acceptance."""
import inspect
import unittest

import numpy as np

from test_periodic_galerkin_consecutive import EuclideanProblem
from periodic_galerkin_consecutive import run_consecutive


class ObjectiveAblationTest(unittest.TestCase):
    def api(self):
        self.assertIn('enforce_pbe', inspect.signature(run_consecutive).parameters)
        self.assertIn('require_nonincreasing_loss', inspect.signature(run_consecutive).parameters)
        return run_consecutive

    def test_energy_improvement_can_accept_loss_increase_only_when_explicit(self):
        run = self.api()
        p = EuclideanProblem()
        options = dict(max_steps=1, max_trials=1,
                       evaluate=lambda x,i: dict(objective=0., loss=999.))
        old = p.run(run, **options)
        new = p.run(run, require_nonincreasing_loss=False, **options)
        self.assertEqual(old['accepted_steps'], 0)
        self.assertEqual(new['accepted_steps'], 1)
        self.assertEqual(new['record']['loss'], 999.)

    def test_free_arm_has_no_pbe_model_or_fake_measurement(self):
        run = self.api()
        p = EuclideanProblem(2)
        p.target = np.array([1., 0.])
        p.record = lambda x: dict(objective=float((1-x[0])**2), loss=float(1+x[0]), pbe=None)
        p.measure = lambda x,i: dict(gate='pass', pbe=None, pbe_gate='not_measured')
        p.b = np.array([1e6, 0.])
        r = p.run(run, enforce_pbe=False, require_nonincreasing_loss=False,
                  pbe_proposal='inequality', max_steps=2, initial_radius=.02)
        self.assertEqual(r['accepted_steps'], 2)
        self.assertGreater(r['x'][0], .04)
        self.assertIsNone(r['record']['pbe'])
        self.assertEqual(r['counts']['secant_updates'], 0)
        self.assertTrue(all(h['predicted_pbe'] is None for h in r['history']))
        self.assertTrue(all(h['pbe_proposal_used']=='disabled_diagnostic' for h in r['history']))

    def test_free_arm_may_report_large_actual_pbe_without_rejecting(self):
        run = self.api()
        p = EuclideanProblem()
        p.pbe = lambda x: .2
        r = p.run(run, enforce_pbe=False, require_nonincreasing_loss=False, max_steps=1)
        self.assertEqual(r['record']['pbe'], .2)
        self.assertEqual(r['accepted_steps'], 1)
        self.assertEqual(r['counts']['secant_updates'], 0)

    def test_nonfinite_or_failed_records_never_pass_free_arm(self):
        run = self.api()
        for measurement in (dict(gate='fail'), dict(gate='pass', pbe=np.nan)):
            p = EuclideanProblem()
            r = p.run(run, enforce_pbe=False, require_nonincreasing_loss=False,
                      max_trials=1, measure=lambda x,i: measurement)
            self.assertEqual(r['accepted_steps'], 0)
        for loss in (np.nan, np.inf, -1.):
            p = EuclideanProblem()
            r = p.run(run, enforce_pbe=False, require_nonincreasing_loss=False,
                      max_trials=1, evaluate=lambda x,i: dict(objective=0., loss=loss))
            self.assertEqual(r['accepted_steps'], 0)

    def test_switches_are_booleans_and_defaults_are_exact(self):
        run = self.api()
        a = EuclideanProblem().run(run, max_steps=2)
        b = EuclideanProblem().run(run, max_steps=2, enforce_pbe=True,
                                   require_nonincreasing_loss=True)
        np.testing.assert_equal(a,b)
        for key in ('enforce_pbe','require_nonincreasing_loss'):
            for bad in (0, None, 'false'):
                with self.assertRaises(ValueError):
                    EuclideanProblem().run(run, **{key:bad})


if __name__=='__main__':
    unittest.main()
