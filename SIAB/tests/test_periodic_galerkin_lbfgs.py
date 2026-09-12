import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'opt_orb_pytorch_dpsi'))
from periodic_galerkin_lbfgs import GrassmannChart, EvaluationBudget, run_lbfgs

torch.set_num_threads(1)


class ChartTests(unittest.TestCase):
    def anchor(self):
        torch.manual_seed(5)
        return {'C': [torch.linalg.qr(torch.randn(7, k, dtype=torch.float64))[0]
                      for k in (3, 2, 0)]}

    def test_origin_subspace_and_orthonormality(self):
        anchor = self.anchor()
        chart = GrassmannChart(anchor)
        self.assertEqual(chart.dimension, 22)
        x = np.linspace(-.04, .04, chart.dimension)
        for a, zero, moved in zip(anchor['C'], chart.coefficients(np.zeros(22))['C'],
                                   chart.coefficients(x)['C']):
            torch.testing.assert_close(a, zero, atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(moved.T@moved, torch.eye(a.shape[1], dtype=a.dtype),
                                       atol=1e-14, rtol=1e-14)

    def test_exact_pullback_at_nonzero_chart_coordinate(self):
        chart = GrassmannChart(self.anchor())
        x = np.linspace(-.12, .10, chart.dimension)
        weights = [torch.linspace(.5, 3., b.numel(), dtype=b.dtype).reshape(b.shape)
                   for b in chart.coefficients(x)['C']]
        def value(v):
            return sum(float(((b*w)**2).sum()) for b,w in zip(chart.coefficients(v)['C'], weights))
        c = chart.coefficients(x)
        raw = {'C': [2*b*w*w for b,w in zip(c['C'], weights)]}
        g = chart.pullback(x, raw)
        fd = np.array([(value(x+np.eye(len(x))[i]*1e-6)-value(x-np.eye(len(x))[i]*1e-6))/2e-6
                       for i in range(len(x))])
        np.testing.assert_allclose(g, fd, rtol=2e-7, atol=2e-8)

    def test_no_radial_or_channel_is_frozen(self):
        torch.manual_seed(8)
        anchor = {'C': [torch.linalg.qr(torch.randn(31, k, dtype=torch.float64))[0]
                        for k in (3,3,2,0,0)]}
        chart = GrassmannChart(anchor)
        self.assertEqual(chart.dimension, 226)
        moved = chart.coefficients(np.full(226, .01))
        for a,b in zip(anchor['C'], moved['C']):
            if a.shape[1]:
                self.assertTrue(torch.all(torch.linalg.vector_norm(a-b, dim=0) > 1e-4))

    def test_invalid_anchor_coordinate_and_derivative_rejected(self):
        a = self.anchor()
        bad = {'C': [b*2 for b in a['C']]}
        with self.assertRaises(ValueError): GrassmannChart(bad)
        chart = GrassmannChart(a)
        for x in (np.zeros(21), np.full(22, np.nan)):
            with self.assertRaises(ValueError): chart.coefficients(x)
        with self.assertRaises(ValueError): chart.pullback(np.zeros(22), {'C': []})


class LBFGSTests(unittest.TestCase):
    def test_real_consecutive_core_calls_two_argument_adapter(self):
        wf=Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
        sys.path.insert(0,str(wf))
        from run_c_optimizer_comparison import consecutive_evaluator
        from periodic_galerkin_consecutive import run_consecutive
        record=lambda x:dict(objective=float(x@x),loss=float(x@x),pbe=None)
        f=consecutive_evaluator(lambda x:x,lambda x:x,record)
        result=run_consecutive(np.ones(2),record(np.ones(2)),np.zeros(2),
            gradient=lambda x:2*x,retract=lambda x,s:x+s,project=lambda x,g:g,
            measure=lambda x,i:dict(gate='pass',pbe=None),evaluate=f,
            checkpoint=lambda s:None,max_steps=1,enforce_pbe=False,
            require_nonincreasing_loss=False)
        self.assertEqual(result['accepted_steps'],1)

    def test_analytic_quadratic_and_accepted_checkpoints(self):
        checkpoints = []
        target = np.array([.02,-.025,.01])
        metric = np.array([1.,10.,100.])
        def fg(x):
            dx=x-target
            return float(np.dot(metric*dx,dx)/2), metric*dx
        result=run_lbfgs(np.zeros(3), fg, checkpoints.append, max_steps=40,
                         max_evaluations=80, coordinate_bound=.05)
        np.testing.assert_allclose(result['x'], target, atol=1e-6)
        self.assertLess(result['objective'], 1e-12)
        self.assertTrue(checkpoints)
        self.assertTrue(all(a['objective'] >= b['objective']
                            for a,b in zip(checkpoints,checkpoints[1:])))
        np.testing.assert_array_equal(checkpoints[-1]['x'], result['x'])
        self.assertEqual(result['physical_release_gate'], 'hold')

    def test_budget_never_returns_unaccepted_trial(self):
        calls=[]
        def fg(x):
            calls.append(x.copy())
            return float(np.sum((x-.02)**2)), 2*(x-.02)
        result=run_lbfgs(np.zeros(2), fg, lambda _: None, max_evaluations=1)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['stop_reason'],'max_evaluations')
        np.testing.assert_array_equal(result['x'],np.zeros(2))

    def test_external_forward_budget_retains_last_accepted(self):
        def fg(x):
            if np.any(x): raise EvaluationBudget('kernel budget')
            return 1., np.ones_like(x)
        result=run_lbfgs(np.zeros(2),fg,lambda _:None)
        self.assertEqual(result['stop_reason'],'max_evaluations')
        self.assertEqual(result['accepted_steps'],0)

    def test_bad_gradient_fails_closed(self):
        with self.assertRaises(ValueError):
            run_lbfgs(np.zeros(2), lambda x:(1.,np.full(2,np.nan)),lambda _:None)


if __name__ == '__main__': unittest.main()
