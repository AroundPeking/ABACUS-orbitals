"""An Ec refresh needs one backward pass, not a new fit or a new space."""

import copy
import inspect
import math
import unittest
from unittest import mock

import torch

import common  # noqa: F401
import periodic_galerkin_fit as fit
import periodic_galerkin_radial_diagnostics as diagnostics
import test_periodic_galerkin_rpa_fit as fixtures


class EcRefreshTest(unittest.TestCase):
    def coefficients(self):
        datasets, initial = fixtures.PeriodicGalerkinRpaFitTest().full_radial_fixture()
        fixed, variable, _ = fit._validate_inputs(datasets, initial, {"C": (0, 0, 0)})
        fit._retract_variables(fixed, variable)
        return datasets, {"C": [p.detach().clone() for p in variable["C"]]}

    def test_energy_only_reproduces_both_pass_mode_with_exactly_one_backward(self):
        self.assertIn("energy_only", inspect.signature(diagnostics.evaluate_radial_gradients).parameters)
        datasets, coefficients = self.coefficients()
        before = copy.deepcopy(coefficients)
        kwargs = dict(occupied_capture_tolerance=.3, frequency_batch_size=3,
                      weights=dict(pi_weight=1., trace_log_weight=1., energy_weight=1.))
        with mock.patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as calls:
            baseline = diagnostics.evaluate_radial_gradients(datasets, coefficients, **kwargs)
            self.assertEqual(calls.call_count, 2)
        with mock.patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as calls:
            actual = diagnostics.evaluate_radial_gradients(datasets, coefficients, energy_only=True, **kwargs)
            self.assertEqual(calls.call_count, 1)
        self.assertEqual(actual["gradient_mode"], "energy_only")
        self.assertEqual(actual["backward_passes"], 1)
        self.assertNotIn("loss_gradient", actual)
        for key in ("loss", "rpa", "energy_gradient", "minimum_occupied_capture", "maximum_overlap_condition"):
            self.assertEqual(actual[key], baseline[key])
        self.assertEqual(actual["physical_release_gate"], "hold")
        for actual_c, original in zip(coefficients["C"], before["C"]):
            torch.testing.assert_close(actual_c, original, rtol=0, atol=0)
            self.assertFalse(actual_c.requires_grad)
            self.assertIsNone(actual_c.grad)

    def test_energy_only_is_an_explicit_boolean_and_rejects_before_evaluation(self):
        self.assertIn("energy_only", inspect.signature(diagnostics.evaluate_radial_gradients).parameters)
        with mock.patch.object(diagnostics, "prepare_periodic_occupied_reference") as load:
            for value in (1, "true", None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    diagnostics.evaluate_radial_gradients([], {}, energy_only=value,
                        occupied_capture_tolerance=.1, frequency_batch_size=1, weights={})
            load.assert_not_called()

    def test_transport_projection_and_angles_match_direct_derivative(self):
        self.assertTrue(hasattr(diagnostics, "transported_descent_report"))
        c = {"C": [torch.eye(4, dtype=torch.float64)[:, :2], torch.empty((4, 0), dtype=torch.float64)]}
        g = {"C": [torch.tensor([[2., 0.], [0., 3.], [6., 0.], [0., 8.]], dtype=torch.float64),
                   torch.empty((4, 0), dtype=torch.float64)]}
        d = {"C": [torch.tensor([[7., 0.], [0., 9.], [-3., 0.], [0., -4.]], dtype=torch.float64),
                   torch.empty((4, 0), dtype=torch.float64)]}
        before = copy.deepcopy(d)
        report = diagnostics.radial_gradient_report(c, g)
        result = diagnostics.transported_descent_report(c, report, d)
        self.assertEqual(result["transport"], "horizontal_projection_at_current_signed_QR_frame")
        self.assertAlmostEqual(result["projected_direction_norm"], 5.)
        self.assertAlmostEqual(result["descent_cosine"], 1.)
        self.assertAlmostEqual(result["angle_degrees"], 0.)
        self.assertAlmostEqual(result["energy_derivative_ha_per_cell"], -10.)
        self.assertEqual(len(result["radials"]), 2)
        self.assertAlmostEqual(sum(r["energy_derivative_ha_per_cell"] for r in result["radials"]), -10.)
        for row in result["radials"]:
            self.assertAlmostEqual(row["descent_cosine"], 1.)
        h = d["C"][0] - c["C"][0] @ (c["C"][0].T @ d["C"][0])
        direction = {"C": [h / h.norm(), d["C"][1]]}
        values = []
        for radius in (-1e-5, 1e-5):
            displaced = diagnostics.retract_displacement(c, direction, radius)
            values.append(float((g["C"][0] * displaced["C"][0]).sum()))
        self.assertAlmostEqual((values[1]-values[0])/2e-5, -10., places=7)
        for actual, original in zip(d["C"], before["C"]):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_transport_zero_direction_or_gradient_is_not_false_alignment(self):
        self.assertTrue(hasattr(diagnostics, "transported_descent_report"))
        c = {"C": [torch.eye(3, dtype=torch.float64)[:, :1]]}
        g = {"C": [torch.tensor([[0.], [1.], [0.]], dtype=torch.float64)]}
        zero = {"C": [torch.zeros_like(c["C"][0])]}
        for gradient, direction in ((g, c), (zero, g)):
            result = diagnostics.transported_descent_report(c,
                diagnostics.radial_gradient_report(c, gradient), direction)
            self.assertIsNone(result["descent_cosine"])
            self.assertIsNone(result["angle_degrees"])
            self.assertEqual(result["energy_derivative_ha_per_cell"], 0.)

    def test_transport_rejects_bad_direction_and_inconsistent_report(self):
        self.assertTrue(hasattr(diagnostics, "transported_descent_report"))
        c = {"C": [torch.eye(3, dtype=torch.float64)[:, :1]]}
        g = {"C": [torch.tensor([[0.], [1.], [0.]], dtype=torch.float64)]}
        report = diagnostics.radial_gradient_report(c, g)
        for direction in ({"C": []}, {"C": [g["C"][0]*float("nan")]}, {"C": [g["C"][0]*1e200]}):
            with self.assertRaises(ValueError):
                diagnostics.transported_descent_report(c, report, direction)
        damaged = copy.deepcopy(report)
        damaged["channels"][0]["raw_gradient"][1][0] = 2.
        with self.assertRaises(ValueError):
            diagnostics.transported_descent_report(c, damaged, g)


if __name__ == "__main__":
    unittest.main()
