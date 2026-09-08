import importlib.util
import math
import unittest

import common  # noqa: F401


class RadialStepDiagnosticTest(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("diagnose_c_radial_step_cap"),
                             "bounded step-cap diagnostic is not implemented")
        import diagnose_c_radial_step_cap as api
        return api

    def test_does_not_stop_at_first_improvement_or_replay_known_step(self):
        api = self.api()
        calls, forwards = [], []
        def measure(radius):
            calls.append(radius)
            return dict(gate="measured", actual_pbe=dict(pbe_gate="pass"))
        def evaluate(radius):
            forwards.append(radius)
            return dict(gain_mev=radius*1000)
        result = api.probe_radius_boundary(measure, evaluate)
        self.assertEqual(calls, [.04,.08,.12])
        self.assertEqual(forwards, calls)
        self.assertEqual(len(result), 3)
        self.assertNotIn(.02, calls)

    def test_pbe_and_cheap_failures_do_not_run_response(self):
        api = self.api()
        forwards = []
        def measure(radius):
            if radius == .04:
                return dict(gate="rejected_cheap_guard", reason="protected band")
            return dict(gate="measured", actual_pbe=dict(pbe_gate="fail" if radius==.08 else "pass"))
        def evaluate(radius):
            forwards.append(radius)
            return dict(gain_mev=1.)
        rows = api.probe_radius_boundary(measure, evaluate)
        self.assertEqual(forwards, [.12])
        self.assertEqual(rows[0]["rpa"], "not_run")
        self.assertEqual(rows[1]["gate"], "rejected_actual_PBE")

    def test_invalid_callback_state_fails_closed(self):
        api = self.api()
        with self.assertRaises(ValueError):
            api.probe_radius_boundary(lambda r:dict(gate="measured",actual_pbe=dict(pbe_gate="unknown")),
                                      lambda r:dict(gain_mev=0.))
        with self.assertRaises(ValueError):
            api.probe_radius_boundary(lambda r:dict(gate="measured",actual_pbe=dict(pbe_gate="pass")),
                                      lambda r:dict(gain_mev=math.nan))

    def test_old_driver_has_first_success_stop_even_when_further_gain_exists(self):
        import run_c_response_short_cycle as old
        calls, forwards = [], []
        def measure(name, weights):
            calls.append(name)
            return dict(candidate_energy_ev=-300., pbe_gate="pass")
        def evaluate(name):
            forwards.append(name)
            return dict(loss=.9,rpa=dict(candidate_energy_ha=-.41,reference_energy_ha=-.5))
        initial=dict(loss=1.,rpa=dict(candidate_energy_ha=-.4,reference_energy_ha=-.5))
        result=old.calibrate_search([1.]*8,-300.,initial,measure,evaluate)
        self.assertEqual(len(calls),17)
        self.assertEqual(forwards,["trial_0"])
        self.assertEqual(result["selected_trial"],"trial_0")


if __name__ == "__main__":
    unittest.main()
