"""Diagnostic radial gradients must not change optimization or physics."""

import importlib
import importlib.util
import copy
import math
import hashlib
import json
import sys
import unittest
from unittest import mock

import torch

import common  # noqa: F401
import periodic_galerkin_fit as fit
from periodic_galerkin_fit import CandidateGuardError
import test_periodic_galerkin_rpa_fit as fixtures


class RadialDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("periodic_galerkin_radial_diagnostics"),
                             "radial diagnostic module missing")
        self.module = importlib.import_module("periodic_galerkin_radial_diagnostics")
        self.c = {"C": [torch.eye(4, dtype=torch.float64)[:, :2].clone(),
                         torch.empty((4, 0), dtype=torch.float64)]}
        self.g = {"C": [torch.tensor([[2., 3.], [4., 5.], [6., 0.], [0., 8.]],
                                     dtype=torch.float64), torch.empty((4, 0), dtype=torch.float64)]}

    def test_horizontal_projection_and_independent_radial_directions(self):
        result = self.module.radial_gradient_report(self.c, self.g)
        self.assertEqual(result["coordinate_metric"], "Euclidean_coefficient_signed_QR_frame")
        self.assertEqual(result["channels"][1]["radials"], [])
        horizontal = torch.tensor(result["channels"][0]["horizontal_gradient"], dtype=torch.float64)
        torch.testing.assert_close(self.c["C"][0].T @ horizontal, torch.zeros((2, 2), dtype=torch.float64))
        self.assertEqual([r["horizontal_norm"] for r in result["channels"][0]["radials"]], [6., 8.])
        for z, expected in enumerate((-6., -8.)):
            direction = self.module.radial_descent_direction(self.c, result, "C", 0, z)
            self.assertEqual(float(direction["C"][0][:, 1-z].norm()), 0.)
            self.assertAlmostEqual(float(direction["C"][0].norm()), 1.)
            self.assertAlmostEqual(float((self.g["C"][0] * direction["C"][0]).sum()), expected)

    def test_projection_removes_pure_frame_changes(self):
        vertical = {"C": [self.c["C"][0] @ torch.tensor([[0., 2.], [-2., 0.]], dtype=torch.float64),
                           self.g["C"][1]]}
        result = self.module.radial_gradient_report(self.c, vertical)
        self.assertEqual(result["horizontal_gradient_norm"], 0.)
        self.assertIsNone(self.module.radial_descent_direction(self.c, result, "C", 0, 0))

    def test_rejects_invalid_shapes_nonfinite_or_nonorthonormal_coordinates(self):
        for coefficients, gradients in (({"C": [self.c["C"][0]*2]}, {"C": [self.g["C"][0]]}),
                                       (self.c, {"C": [self.g["C"][0]]}),
                                       (self.c, {"C": [torch.full_like(self.g["C"][0], float("nan")), self.g["C"][1]]})):
            with self.assertRaises(ValueError):
                self.module.radial_gradient_report(coefficients, gradients)

    def test_two_size_guard_derivatives_preserve_inputs_and_label_not_pbe(self):
        def guard(coefficients):
            c = coefficients["C"][0]
            return dict(gate=True, occupied_band_sum_change_ev_per_atom=float(c[2, 0]),
                        maximum_target_band_change_ev=abs(float(c[2, 0])), minimum_gap_ev=2.)
        before = self.c["C"][0].clone()
        gradient = self.module.radial_gradient_report(self.c, self.g)
        report = self.module.radial_guard_sensitivity(self.c, gradient, guard, radii=(1e-4, 5e-5))
        self.assertEqual(report["scope"], "frozen_band_sensitivity_not_actual_PBE_derivative")
        self.assertEqual(len(report["radials"]), 2)
        for row in report["radials"]:
            self.assertEqual(row["status"], "success")
            self.assertEqual(len(row["stencils"]), 2)
        for stencil in report["radials"][0]["stencils"]:
            self.assertAlmostEqual(stencil["band_sum_derivative_ev_per_atom"], -1., places=7)
        torch.testing.assert_close(self.c["C"][0], before, rtol=0, atol=0)

    def test_guard_failure_remains_unresolved_not_zero_sensitivity(self):
        report = self.module.radial_gradient_report(self.c, self.g)
        def guard(coefficients):
            raise CandidateGuardError("blocked")
        result = self.module.radial_guard_sensitivity(self.c, report, guard, radii=(1e-4, 5e-5))
        self.assertTrue(all(r["status"] == "guard_rejected" for r in result["radials"]))
        self.assertTrue(all(s["band_sum_derivative_ev_per_atom"] is None
                            for r in result["radials"] for s in r["stencils"]))

    def test_incomplete_duplicate_or_misidentified_radial_report_is_rejected(self):
        report = self.module.radial_gradient_report(self.c, self.g)
        mutations = [lambda r: r.update(channels=[]),
                     lambda r: r["channels"].pop(),
                     lambda r: r["channels"].append(copy.deepcopy(r["channels"][0])),
                     lambda r: r["channels"][0]["radials"].pop(),
                     lambda r: r["channels"][0]["radials"][0].update(zeta=2),
                     lambda r: r["channels"][0].update(l=3)]
        for mutate in mutations:
            damaged = copy.deepcopy(report)
            mutate(damaged)
            with self.subTest(report=damaged), self.assertRaises(ValueError):
                self.module.radial_guard_sensitivity(self.c, damaged, mock.Mock())

    def test_norm_overflow_and_nonunit_report_cannot_give_false_zero_derivative(self):
        huge = {"C": [self.g["C"][0]*1e200, self.g["C"][1]]}
        with self.assertRaises(ValueError):
            self.module.radial_gradient_report(self.c, huge)
        report = self.module.radial_gradient_report(self.c, self.g)
        report["channels"][0]["horizontal_gradient"][2][0] = 1e200
        with self.assertRaises(ValueError):
            self.module.radial_guard_sensitivity(self.c, report, mock.Mock())

    def test_full_complex_energy_and_loss_gradients_match_existing_objective(self):
        datasets, initial = fixtures.PeriodicGalerkinRpaFitTest().full_radial_fixture()
        fixed, variable, _ = fit._validate_inputs(datasets, initial, {"C": (0, 0, 0)})
        fit._retract_variables(fixed, variable)
        c = {"C": [p.detach().clone() for p in variable["C"]]}
        weights = dict(pi_weight=1., trace_log_weight=1., energy_weight=1.)
        with mock.patch.object(fit, "optimize_periodic_galerkin_basis", side_effect=AssertionError("optimizer called")):
            result = self.module.evaluate_radial_gradients(datasets, c,
                occupied_capture_tolerance=.3, frequency_batch_size=3, weights=weights)
        old_c = {"C": [p.clone().requires_grad_() for p in c["C"]]}
        expected, _, _, _, diagnostic = fit._global_rpa_loss(datasets, old_c,
            occupied_capture_tolerance=.3, weights=weights, frequency_batch_size=3)
        expected_gradient = torch.autograd.grad(expected, old_c["C"])
        self.assertAlmostEqual(result["loss"], float(expected), places=12)
        self.assertAlmostEqual(result["rpa"]["candidate_energy_ha"], diagnostic["candidate_energy_ha"], places=12)
        for row, value in zip(result["loss_gradient"]["channels"], expected_gradient):
            torch.testing.assert_close(torch.tensor(row["raw_gradient"], dtype=torch.float64), value)
        # Both observables and every nonempty channel must survive independent stencils.
        for observable in ("loss_gradient", "energy_gradient"):
            for l, channel in enumerate(c["C"]):
                for z in range(channel.shape[1]):
                    direction = self.module.radial_descent_direction(c, result[observable], "C", l, z)
                    values = []
                    for sign in (-1., 1.):
                        trial = self.module.retract_displacement(c, direction, sign*1e-5)
                        loss, _, _, _, d = fit._global_rpa_loss(datasets, trial,
                            occupied_capture_tolerance=.3, weights=weights, frequency_batch_size=3)
                        values.append(float(loss) if observable == "loss_gradient" else d["candidate_energy_ha"])
                    derivative = (values[1]-values[0])/2e-5
                    norm = result[observable]["channels"][l]["radials"][z]["horizontal_norm"]
                    self.assertAlmostEqual(derivative, -norm, places=7)
        # Raw Ec derivatives are checked outside retraction, independently of loss weights.
        for l, channel in enumerate(c["C"]):
            numerical = torch.empty_like(channel)
            for i in range(channel.shape[0]):
                for z in range(channel.shape[1]):
                    values = []
                    for sign in (-1., 1.):
                        trial = {"C": [p.clone() for p in c["C"]]}
                        trial["C"][l][i, z] += sign*1e-5
                        _, _, _, _, d = fit._global_rpa_loss(datasets, trial,
                            occupied_capture_tolerance=.3, weights=weights, frequency_batch_size=3)
                        values.append(d["candidate_energy_ha"])
                    numerical[i, z] = (values[1]-values[0])/2e-5
            torch.testing.assert_close(torch.tensor(result["energy_gradient"]["channels"][l]["raw_gradient"],
                dtype=torch.float64), numerical, atol=1e-8, rtol=1e-6)
        independent = self.module.evaluate_radial_gradients(datasets, c,
            occupied_capture_tolerance=.3, frequency_batch_size=3, weights=dict(weights, energy_weight=0.))
        self.assertEqual(independent["energy_gradient"], result["energy_gradient"])
        self.assertEqual(result["physical_release_gate"], "hold")
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in c["C"]))

    def test_c_runner_requires_pbe_matching_candidate_and_reproduces_existing_endpoint(self):
        from pathlib import Path
        workflow = Path(__file__).resolve().parents[1] / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
        sys_path = __import__("sys").path
        sys_path.insert(0, str(workflow))
        self.assertIsNotNone(importlib.util.find_spec("diagnose_c_radial_gradients"), "C diagnostic runner missing")
        runner = importlib.import_module("diagnose_c_radial_gradients")
        candidate = dict(coefficient_sha256="a"*64, orbital_sha256="b"*64, alpha=.25,
            candidate=dict(loss=.08, rpa=dict(candidate_energy_ha=-.42, reference_energy_ha=-.51)))
        pbe = dict(pbe_total_energy_gate="pass", scf_log_gate="pass", band_count_check="pass",
            band_counts=dict(occupied=4, total=44), candidate_result_sha256="c"*64,
            candidate_orbital_sha256="b"*64, candidate_energy_ev=-309.85, baseline_energy_ev=-309.86,
            energy_delta_ev_per_c=.005, tolerance_ev_per_c=.01)
        runner.validate_pbe_binding(candidate, "c"*64, pbe)
        for changed in (dict(candidate_result_sha256="d"*64), dict(pbe_total_energy_gate="fail"),
                        dict(candidate_energy_ev=-309.), dict(energy_delta_ev_per_c=0.),
                        dict(tolerance_ev_per_c=.1), dict(band_counts=dict(occupied=4, total=45))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.validate_pbe_binding(candidate, "c"*64, dict(pbe, **changed))
        reproduced = dict(loss=.08, rpa=dict(candidate_energy_ha=-.42, reference_energy_ha=-.51,
            complete_q_weight=True, q_weight_coverage=1., per_q=[dict(selected_iq=iq, frequency_ha=list(range(12)))
                for iq in (1, 22, 43, 6, 27, 23, 11, 55)]))
        runner.validate_reproduction(candidate, reproduced)
        for changed in (dict(loss=.081), dict(rpa=dict(reproduced["rpa"], candidate_energy_ha=-.41)),
                        dict(rpa=dict(reproduced["rpa"], complete_q_weight=False))):
            with self.assertRaises(ValueError):
                runner.validate_reproduction(candidate, dict(reproduced, **changed))

    def test_runner_writes_hashed_diagnostic_and_never_overwrites_output(self):
        import test_periodic_galerkin_c_backoff as backoff_fixtures
        fixture = backoff_fixtures.CBackoffTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        pair = fixture.pair()
        for record in pair:
            for q, multiplicity in zip(record["rpa"]["per_q"], (1, 8, 4, 6, 24, 12, 3, 6)):
                q.update(q_weight=multiplicity/64.,
                    candidate_contributions_ha=[record["rpa"]["candidate_energy_ha"]/96]*12,
                    reference_contributions_ha=[record["rpa"]["reference_energy_ha"]/96]*12)
        fixture.pair = lambda: pair
        candidate = fixture.produce()
        runner = importlib.import_module("diagnose_c_radial_gradients")
        source = fixture.output / "RESULT.json"
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        log = fixture.root / "pbe.log"
        log.write_text("archived SCF log")
        pbe = fixture.root / "pbe.json"
        pbe.write_text(json.dumps(dict(pbe_total_energy_gate="pass", scf_log_gate="pass",
            band_count_check="pass", band_counts=dict(occupied=4, total=44),
            candidate_result_sha256=digest(source), candidate_orbital_sha256=candidate["orbital_sha256"],
            candidate_log_path=str(log), candidate_log_sha256=digest(log),
            candidate_energy_ev=-309.85, baseline_energy_ev=-309.86,
            energy_delta_ev_per_c=.005, tolerance_ev_per_c=.01)))
        output = fixture.root / "diagnostic"
        args = ["diagnose", "--candidate-result", str(source), "--candidate-result-sha256", digest(source),
                "--pbe-collection", str(pbe), "--pbe-collection-sha256", digest(pbe),
                "--source-commit", "a"*40, "--output", str(output)]
        def evaluate(datasets, coefficients, **kwargs):
            report = self.module.radial_gradient_report(coefficients,
                {e: [torch.zeros_like(c) for c in cs] for e, cs in coefficients.items()})
            return dict(candidate["candidate"], loss_gradient=report, energy_gradient=report,
                        scope="radial_gradients_at_fixed_candidate_no_optimization")
        with mock.patch.object(sys, "argv", args), \
                mock.patch.object(runner, "load_frozen_c", return_value=((), [], lambda c: dict(gate=True))), \
                mock.patch.object(runner, "evaluate_radial_gradients", side_effect=evaluate):
            runner.main()
            provenance = json.loads((output/"PROVENANCE.json").read_text())
            self.assertEqual(provenance["diagnostic_sha256"], digest(output/"RADIAL_GRADIENT_DIAGNOSTIC.json"))
            with self.assertRaises(FileExistsError):
                runner.main()


if __name__ == "__main__":
    unittest.main()
