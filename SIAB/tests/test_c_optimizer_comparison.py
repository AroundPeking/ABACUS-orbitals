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


if __name__ == '__main__': unittest.main()
