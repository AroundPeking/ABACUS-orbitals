import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
import hashlib
import json
import tempfile

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

    def test_continuation_preserves_layout_pbe_origin_and_finite_ambient_model(self):
        api=self.api()
        result=dict(status="success",nu=[3,3,2,0,0],fixed_nu=[0]*5,ao_per_C=22,
                    full_constrained_stationarity="not_established",physical_release_gate="hold",
                    optimization=dict(pbe_gradient=[.1]*248,record=dict(pbe=.0087)))
        actual=dict(pbe_gate="pass",baseline_energy_ev=-309.8590820440637,
                    energy_delta_ev_per_c=.0087)
        api.validate_continuation_payload(result,actual)
        for key,value in (("pbe_gradient",[.1]*247),("pbe_gradient",[float("nan")]*248),
                          ("record",dict(pbe=.011))):
            invalid=copy.deepcopy(result);invalid["optimization"][key]=value
            with self.assertRaises(ValueError):api.validate_continuation_payload(invalid,actual)
        for key,value in (("baseline_energy_ev",-309.86),("pbe_gate","fail"),
                          ("energy_delta_ev_per_c",.007)):
            invalid=dict(actual);invalid[key]=value
            with self.assertRaises(ValueError):api.validate_continuation_payload(result,invalid)
        invalid=copy.deepcopy(result);invalid["fixed_nu"]=[2,2,1,0,0]
        with self.assertRaises(ValueError):api.validate_continuation_payload(invalid,actual)

    def test_parent_uses_isolated_original_declared_source_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp).resolve()
            name='SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/refresh_c_combined_step_gradient.py'
            validator=stage/'source'/name;validator.parent.mkdir(parents=True)
            validator.write_bytes(b'# frozen validator fixture\n')
            manifest=dict(source_commit='a'*40,source_directory=str(stage/'source'),
                          files={name:hashlib.sha256(validator.read_bytes()).hexdigest()})
            raw=json.dumps(manifest).encode();(stage/'DEPLOYMENT.json').write_bytes(raw)
            digest=hashlib.sha256(raw).hexdigest()
            with mock.patch('subprocess.check_output',return_value=raw) as child:
                self.assertEqual(self.api().verify_parent_source(stage,digest,'a'*40),manifest)
            command=child.call_args[0][0]
            self.assertEqual(command[1:3],['-I','-B'])
            self.assertEqual(command[-3:],[str(stage),digest,'a'*40])
            self.assertEqual(child.call_args[1]['env']['PYTHONDONTWRITEBYTECODE'],'1')
            validator.write_bytes(b'# changed\n')
            with mock.patch('subprocess.check_output') as child:
                with self.assertRaises(ValueError):self.api().verify_parent_source(stage,digest,'a'*40)
                child.assert_not_called()

    def test_continuation_log_and_inputs_are_one_scf_record(self):
        prior=Path('/prior')
        actual=dict(candidate_log_path='/prior/result/trial_021/pbe/OUT.C/running_scf.log')
        self.assertEqual(self.api().continuation_scf_root(prior,actual,dict(actual)),
                         Path('/prior/result/trial_021/pbe'))
        mismatch=dict(candidate_log_path='/prior/result/trial_022/pbe/OUT.C/running_scf.log')
        with self.assertRaises(ValueError):self.api().continuation_scf_root(prior,actual,mismatch)
        outside=dict(candidate_log_path='/other/result/trial_021/pbe/OUT.C/running_scf.log')
        with self.assertRaises(ValueError):self.api().continuation_scf_root(prior,outside,outside)


if __name__ == "__main__":
    unittest.main()
