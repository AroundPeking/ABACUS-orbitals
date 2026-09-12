import copy
import importlib
import os
import sys
import unittest
from pathlib import Path

WF=Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0,str(WF))


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('c_optimizer_continuation')

    def test_terminal_state_exact_parent(self):
        self.m.terminal_state('3330271|COMPLETED|0:0\n','3330271')
        for s in ['3330271|RUNNING|0:0','3330271|TIMEOUT|0:0',
                  '3330271|COMPLETED|1:0','3330271.batch|COMPLETED|0:0','']:
            with self.assertRaises(ValueError): self.m.terminal_state(s,'3330271')

    def test_only_winning_method_can_continue(self):
        arms={k:dict(gain_mev_per_c=g,seconds=t,accepted_steps=2,
                      final_record={'objective':e}) for k,g,t,e in
              [('steepest',1.5,1100,.081),('lbfgs',15.,1500,.0799)]}
        self.assertGreater(self.m.winner(arms)['efficiency_ratio'],7.)
        for k,v in [('seconds',20000),('gain_mev_per_c',1.),('accepted_steps',0)]:
            bad=copy.deepcopy(arms); bad['lbfgs'][k]=v
            with self.assertRaises(ValueError): self.m.winner(bad)

    def test_q_sum_weights_and_finite_values(self):
        q=[dict(selected_iq=i,q_weight=w/64,frequency_ha=list(range(1,13)),
                candidate_contributions_ha=[-.001]*12,
                reference_contributions_ha=[self.m.REFERENCE/96]*12)
           for i,w in zip(self.m.INDICES,self.m.MULT)]
        r=dict(candidate_energy_ha=-.096,reference_energy_ha=self.m.REFERENCE,
               complete_q_weight=True,q_weight_coverage=1.,per_q=q)
        self.m.validate_rpa(r)
        for mutate in [lambda d:d['per_q'].pop(),
                       lambda d:d['per_q'][1].update(q_weight=1),
                       lambda d:d.update(candidate_energy_ha=-.097),
                       lambda d:d['per_q'][0]['candidate_contributions_ha'].__setitem__(0,float('nan'))]:
            bad=copy.deepcopy(r); mutate(bad)
            with self.assertRaises(ValueError): self.m.validate_rpa(bad)

    def test_frozen_kernel_warning_only(self):
        warning='radial_diagnostics.py:250: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first. (Triggered internally at xyz)\n  if not all(math.isfinite(v) for v in rpa.values()) or not math.isfinite(float(objective.loss)):\n'
        self.m.validate_stderr(warning)
        for bad in ['Traceback (most recent call last):','out of memory',warning+'new warning']:
            with self.assertRaises(ValueError): self.m.validate_stderr(bad)

    def test_continuation_budget_and_single_arm(self):
        from run_c_optimizer_comparison import execution_policy
        self.assertEqual(execution_policy(False,12,24),('steepest','lbfgs'))
        self.assertEqual(execution_policy(True,60,120),('lbfgs',))
        for args in [(False,60,120),(True,0,120),(True,61,120),(True,60,121)]:
            with self.assertRaises(ValueError): execution_policy(*args)

    def test_nonfinite_gradient_or_diagnostic_is_rejected(self):
        d=dict(loss=.01,minimum_occupied_capture=.99,maximum_overlap_condition=1e7,
               energy_gradient=dict(raw_gradient_norm=1.,horizontal_gradient_norm=.1,
                                    channels=[{'raw_gradient':[1.], 'horizontal_gradient':[.1]}]))
        self.m.finite_diagnostic(d)
        for mutate in [lambda x:x.update(loss=float('nan')),
                       lambda x:x['energy_gradient']['channels'][0].update(raw_gradient=[float('inf')]),
                       lambda x:x.update(maximum_overlap_condition=float('inf'))]:
            bad=copy.deepcopy(d); mutate(bad)
            with self.assertRaises(ValueError): self.m.finite_diagnostic(bad)

    @unittest.skipUnless(os.environ.get('C_OPTIMIZER_AUDIT_FIXTURE'),'requires completed read-only fixture')
    def test_completed_real_parent_and_corrupt_hash(self):
        p=Path(os.environ['C_OPTIMIZER_AUDIT_FIXTURE'])
        r=self.m.validate_parent(p,self.m.sha(p/'result/RESULT.json'),
                                 '3330271|COMPLETED|0:0',verify_source=False)
        self.assertAlmostEqual(r['summary']['lbfgs']['error_ev_per_c'],1.0870838020760727)
        self.assertGreater(r['selection']['efficiency_ratio'],7.)
        with self.assertRaises(ValueError):
            self.m.validate_parent(p,'0'*64,'3330271|COMPLETED|0:0',verify_source=False)


if __name__=='__main__': unittest.main()
