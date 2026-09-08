import copy
import importlib.util
from pathlib import Path
import sys
import unittest

WORKFLOW = Path(__file__).resolve().parents[1] / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
sys.path.insert(0, str(WORKFLOW))


class ConsecutiveAdmissionTest(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("run_c_consecutive_radials"))
        import run_c_consecutive_radials as api
        return api

    def fixture(self):
        return dict(radius=.04, gate="diagnostic_measured",
                    actual_pbe=dict(pbe_gate="pass", baseline_energy_ev=-309.8590820440637,
                                    energy_delta_ev_per_c=.009745847520548523),
                    proposal=dict(nu=[3,3,2,0,0], fixed_nu=[0]*5, ao_per_C=22),
                    rpa=dict(candidate=dict(loss=.07, rpa=dict(candidate_energy_ha=-.43,
                                                                  reference_energy_ha=-.514))))

    def test_only_verified_feasible_same_size_endpoint(self):
        self.api().validate_parent_trial(self.fixture())
        for path, value in ((("actual_pbe","pbe_gate"),"fail"),
                            (("actual_pbe","energy_delta_ev_per_c"),.01001),
                            (("actual_pbe","baseline_energy_ev"),-309.85),
                            (("proposal","fixed_nu"),[2,2,1,0,0]),
                            (("proposal","nu"),[3,3,3,0,0])):
            trial = copy.deepcopy(self.fixture())
            trial[path[0]][path[1]] = value
            with self.assertRaises(ValueError):
                self.api().validate_parent_trial(trial)

    def test_reference_objective_not_unbounded_minimum(self):
        api = self.api()
        r = dict(loss=.1, rpa=dict(candidate_energy_ha=-.5, reference_energy_ha=-.4))
        self.assertAlmostEqual(api.controller_record(r,.009)["objective"], .1)
        with self.assertRaises(ValueError):
            api.controller_record(r,float("nan"))

    def test_only_explicit_candidate_nonconvergence_is_recoverable(self):
        api=self.api()
        error=ValueError("SCF log is not unambiguously converged")
        self.assertTrue(api.candidate_nonconvergence(error,b"#SCF IS NOT CONVERGED#"))
        for text in (b"",b"#SCF IS CONVERGED#",b"#SCF IS CONVERGED# SCF NOT CONVERGED"):
            self.assertFalse(api.candidate_nonconvergence(error,text))
        self.assertFalse(api.candidate_nonconvergence(ValueError("input hash mismatch"),
                                                       b"SCF NOT CONVERGED"))


if __name__ == "__main__":
    unittest.main()
