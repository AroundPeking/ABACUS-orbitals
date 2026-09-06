"""Fixed capture floor and center validation before candidate forwards; no production runs."""

import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

WORKFLOW = (Path(__file__).resolve().parents[1] / "example_C_sternheimer" /
            "periodic_basis_optimization" / "galerkin_binding_workflow")
sys.path.insert(0, str(WORKFLOW))

import backoff_c_optimized_direction as backoff


class CBackoffCaptureFloorTest(unittest.TestCase):
    def setUp(self):
        self.assertIn("occupied_capture_floor", inspect.signature(backoff.evaluate_pair).parameters,
                      "evaluate_pair must accept a fixed original capture floor")
        self.center, self.candidate = object(), object()
        self.dataset, self.reference = object(), object()
        self.captures = {self.center: .90, self.candidate: .87}
        self.calls = []
        self.options = dict(guard=lambda _: dict(gate=True), frequency_batch_size=3,
                            weights=dict(pi_weight=1., trace_log_weight=1., energy_weight=1.))
        for name, options in (
            ("prepare_periodic_occupied_reference", dict(side_effect=lambda dataset: dataset)),
            ("_prepare_block_contraction_caches", dict(side_effect=lambda datasets, *_: datasets)),
            ("prepare_periodic_rpa_reference", dict(return_value=self.reference)),
            ("_global_rpa_loss", dict(side_effect=self.forward)),
        ):
            patch = mock.patch.object(backoff, name, **options)
            setattr(self, name, patch.start())
            self.addCleanup(patch.stop)

    def forward(self, datasets, coefficients, *, occupied_capture_tolerance, weights,
                frequency_batch_size, reference_cache):
        self.assertFalse(backoff.torch.is_grad_enabled())
        self.assertEqual(datasets, (self.dataset,))
        self.assertIs(reference_cache, self.reference)
        self.assertEqual(weights, self.options["weights"])
        self.assertEqual(frequency_batch_size, 3)
        self.calls.append((coefficients, occupied_capture_tolerance))
        capture = self.captures[coefficients]
        if capture < 1.-occupied_capture_tolerance:
            raise RuntimeError("candidate basis does not capture the fixed occupied manifold")
        return (0.1, capture, 1., None,
                dict(candidate_energy_ha=-.43, reference_energy_ha=-.51))

    def evaluate(self, **options):
        return backoff.evaluate_pair((self.dataset,), self.center, self.candidate,
                                     **dict(self.options, **options))

    def test_fixed_original_floor_is_identical_for_both_forwards(self):
        first, second = self.evaluate(occupied_capture_floor=.85)
        self.assertEqual(self.calls, [(self.center, 1.-.85), (self.candidate, 1.-.85)])
        self.assertEqual(first["minimum_occupied_capture"], .90)
        self.assertEqual(second["minimum_occupied_capture"], .87)
        self.prepare_periodic_rpa_reference.assert_called_once()

    def test_fixed_original_floor_rejects_center_before_candidate_forward(self):
        self.captures[self.center] = .84
        with self.assertRaisesRegex(RuntimeError, "fixed occupied manifold"):
            self.evaluate(occupied_capture_floor=.85)
        self.assertEqual(self.calls, [(self.center, 1.-.85)])

    def test_fixed_original_floor_rejects_candidate_without_relaxation(self):
        self.captures.update({self.center: .85005, self.candidate: .84998})
        with self.assertRaisesRegex(RuntimeError, "fixed occupied manifold"):
            self.evaluate(occupied_capture_floor=.85)
        self.assertEqual(self.calls, [(self.center, 1.-.85), (self.candidate, 1.-.85)])

    def test_omitted_floor_preserves_existing_asymmetric_tolerances(self):
        self.captures[self.candidate] = .91
        first, second = self.evaluate()
        expected = min(1.-1e-15, max(1e-15, 1.-max(0., .90-1e-4)))
        self.assertEqual(self.calls, [(self.center, 1.-1e-12), (self.candidate, expected)])
        self.assertEqual((first["minimum_occupied_capture"], second["minimum_occupied_capture"]), (.90, .91))

    def test_none_preserves_default_behavior(self):
        self.captures[self.candidate] = .91
        self.evaluate()
        expected = list(self.calls)
        self.calls.clear()
        self.evaluate(occupied_capture_floor=None)
        self.assertEqual(self.calls, expected)

    def test_invalid_floor_rejects_before_any_preparation_or_forward(self):
        for floor in (-.01, 1., 1.01, True, False, "0.85", float("nan"),
                      float("inf"), -float("inf"), complex(.85, 0), 10**1000, -10**1000):
            with self.subTest(floor=floor), self.assertRaisesRegex(ValueError, "occupied_capture_floor"):
                self.evaluate(occupied_capture_floor=floor)
        self.prepare_periodic_occupied_reference.assert_not_called()
        self._prepare_block_contraction_caches.assert_not_called()
        self.prepare_periodic_rpa_reference.assert_not_called()
        self._global_rpa_loss.assert_not_called()

    def test_zero_floor_keeps_tolerance_inside_response_api_domain(self):
        self.evaluate(occupied_capture_floor=0)
        self.assertEqual(self.calls, [(self.center, 1.-1e-15), (self.candidate, 1.-1e-15)])

    def test_near_unit_original_floor_is_not_relaxed_by_tolerance_clamp(self):
        floor = 1.-2.**-53
        self.captures.update({self.center: 1., self.candidate: 1.-2.**-51})
        with self.assertRaisesRegex(RuntimeError, "fixed occupied manifold"):
            self.evaluate(occupied_capture_floor=floor)
        self.assertEqual(self.calls, [(self.center, 1.-floor), (self.candidate, 1.-floor)])

    def test_initial_validator_is_an_optional_keyword_parameter(self):
        parameter = inspect.signature(backoff.evaluate_pair).parameters.get("initial_validator")
        self.assertIsNotNone(parameter, "evaluate_pair must accept an initial_validator callback")
        self.assertIs(parameter.default, None)
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_initial_validator_runs_once_after_center_before_candidate_or_its_guard(self):
        self.options["guard"] = mock.Mock(return_value=dict(gate=True))
        def validate(first):
            self.assertEqual(self.calls, [(self.center, 1.-.85)])
            self.options["guard"].assert_called_once_with(self.center)
            self.assertFalse(backoff.torch.is_grad_enabled())
            self.assertEqual(first["minimum_occupied_capture"], .90)
            self.assertEqual(first["rpa"]["candidate_energy_ha"], -.43)
        validator = mock.Mock(side_effect=validate)
        first, second = self.evaluate(occupied_capture_floor=.85, initial_validator=validator)
        validator.assert_called_once_with(first)
        self.assertIs(validator.call_args[0][0], first)
        self.assertEqual(self.calls, [(self.center, 1.-.85), (self.candidate, 1.-.85)])
        self.assertEqual(second["minimum_occupied_capture"], .87)

    def test_initial_validator_failure_never_consumes_candidate_forward(self):
        self.options["guard"] = mock.Mock(return_value=dict(gate=True))
        failure = ValueError("center reproduction mismatch")
        validator = mock.Mock(side_effect=failure)
        with self.assertRaises(ValueError) as caught:
            self.evaluate(initial_validator=validator)
        self.assertIs(caught.exception, failure)
        validator.assert_called_once()
        self.assertEqual(self.calls, [(self.center, 1.-1e-12)])
        self.options["guard"].assert_called_once_with(self.center)

    def test_noncallable_initial_validator_rejects_before_any_work(self):
        self.options["guard"] = mock.Mock(return_value=dict(gate=True))
        for validator in (True, False, 1, 0, "validator", {}, []):
            with self.subTest(validator=validator), self.assertRaisesRegex(ValueError, "initial_validator"):
                self.evaluate(initial_validator=validator)
        self.prepare_periodic_occupied_reference.assert_not_called()
        self._prepare_block_contraction_caches.assert_not_called()
        self.prepare_periodic_rpa_reference.assert_not_called()
        self._global_rpa_loss.assert_not_called()
        self.options["guard"].assert_not_called()

    def test_initial_validator_none_preserves_default_and_fixed_floor_behavior(self):
        self.captures[self.candidate] = .91
        for options in ({}, dict(occupied_capture_floor=.85)):
            with self.subTest(options=options):
                self.calls.clear()
                self.evaluate(**options)
                expected = list(self.calls)
                self.calls.clear()
                self.evaluate(initial_validator=None, **options)
                self.assertEqual(self.calls, expected)

    def test_failed_center_forward_does_not_call_initial_validator(self):
        self.captures[self.center] = .84
        validator = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "fixed occupied manifold"):
            self.evaluate(occupied_capture_floor=.85, initial_validator=validator)
        validator.assert_not_called()
        self.assertEqual(self.calls, [(self.center, 1.-.85)])

    def test_falsey_callable_initial_validator_is_not_skipped(self):
        validator = mock.MagicMock()
        validator.__bool__.return_value = False
        first, _ = self.evaluate(occupied_capture_floor=.85, initial_validator=validator)
        validator.assert_called_once_with(first)


if __name__ == "__main__":
    unittest.main()
