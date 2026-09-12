import importlib.util
from pathlib import Path
import sys
import unittest

WF = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(WF))


class ComparisonContractTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('comparison',WF/'run_c_optimizer_comparison.py')
        self.m=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_error_sign_both_sides_of_same_reference(self):
        ref=self.m.REFERENCE
        self.assertEqual(self.m.objective_and_sign(ref+.01,ref)[1],1.)
        self.assertEqual(self.m.objective_and_sign(ref-.01,ref)[1],-1.)
        self.assertEqual(self.m.objective_and_sign(ref,ref),(0.,0.))
        with self.assertRaises(ValueError): self.m.objective_and_sign(float('nan'),ref)
        with self.assertRaises(ValueError): self.m.objective_and_sign(-.43,-.5)

    def test_bounded_diagnostic_only(self):
        self.m.validate_budget(12,24)
        for steps,forwards in ((0,24),(12,0),(21,24),(12,41),(True,24)):
            with self.assertRaises(ValueError): self.m.validate_budget(steps,forwards)
        self.assertEqual(self.m.BUNDLE_SHA,'a0e285e12c05d1138f14f9e2e94b9a67641c25aacd6344eba70c9a0ed442a215')
        self.assertEqual(self.m.PHYSICAL_RELEASE,'hold')

    def test_consecutive_evaluator_accepts_trial_id(self):
        seen=[]
        def evaluate(c):
            seen.append(c)
            return {'energy':c}
        callback=self.m.consecutive_evaluator(lambda x:x*2,evaluate,lambda d:d['energy'])
        self.assertEqual(callback(3,17),6)
        self.assertEqual(seen,[6])

    def test_recovery_rejects_changed_parent_and_accepted_work(self):
        r=dict(status='failed_before_any_optimizer_evaluation',job_id='3330163',
               source_commit='5f8a50fd327319848bc708e2beae76c1de2d0c62',
               bundle_sha256=self.m.BUNDLE_SHA,shared_gradient_sha256='a'*64,
               scheduler_state='FAILED',exit_code='1:0',accepted_steps=0,evaluations=0)
        self.m.validate_recovery(r)
        for k,v in [('scheduler_state','RUNNING'),('accepted_steps',1),('evaluations',1),
                    ('source_commit','b'*40),('shared_gradient_sha256','')]:
            with self.assertRaises(ValueError): self.m.validate_recovery(dict(r,**{k:v}))

    def test_parent_scheduler_must_be_exact_failed_job(self):
        self.assertEqual(self.m.parent_scheduler_state('3330163|FAILED|1:0\n'),('FAILED','1:0'))
        for text in ('3330163|RUNNING|0:0\n','3330163|COMPLETED|0:0\n',
                     '3330163.batch|FAILED|1:0\n','3330163|FAILED|2:0\n', ''):
            with self.assertRaises(ValueError): self.m.parent_scheduler_state(text)


if __name__ == '__main__': unittest.main()
