"""Equivalent batched solves, including gradients of every radial column."""

from dataclasses import replace
import unittest

import torch

import common  # noqa: F401
from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
from periodic_galerkin_rpa import periodic_rpa_objective
import test_periodic_galerkin_sos_equivalence as fixtures


class PeriodicGalerkinBatchedTest(unittest.TestCase):
    def fixture(self, repeated=False):
        dataset, _ = fixtures.PeriodicGalerkinSosEquivalenceTest().fixture()
        if repeated:
            dataset = replace(dataset, kpoints=tuple(replace(
                r, source_eigenvalue_ha=torch.tensor([-0.8, -0.8], dtype=torch.float64)
            ) for r in dataset.kpoints))
        coefficients = torch.eye(5, dtype=torch.float64)[:, :4].clone()
        coefficients[4] = torch.tensor([0.01, -0.02, 0.2, 0.1], dtype=torch.float64)
        return dataset, coefficients

    def evaluate(self, dataset, values, **options):
        coefficients = values.detach().clone().requires_grad_(True)
        result = evaluate_periodic_galerkin_coefficient_response(
            dataset, {"C": [coefficients]}, contraction_backend="block",
            occupied_capture_tolerance=0.5, **options)
        objective = periodic_rpa_objective((dataset,), (result.response,))
        objective.loss.backward()
        return result, objective, coefficients.grad

    def test_response_loss_and_all_column_gradients_match_scalar(self):
        for repeated in (False, True):
            dataset, coefficients = self.fixture(repeated)
            expected, loss, gradient = self.evaluate(dataset, coefficients)
            self.assertTrue(bool((gradient.abs().sum(dim=0) > 1e-9).all()))
            for batch in (1, 3, 5, 12, 20):
                with self.subTest(repeated=repeated, batch=batch):
                    actual, objective, grad = self.evaluate(
                        dataset, coefficients, frequency_batch_size=batch)
                    torch.testing.assert_close(actual.response, expected.response,
                                               rtol=1e-11, atol=1e-12)
                    torch.testing.assert_close(objective.loss, loss.loss,
                                               rtol=1e-11, atol=1e-12)
                    torch.testing.assert_close(grad, gradient, rtol=1e-10, atol=1e-11)
                    self.assertEqual(actual.minimum_occupied_capture,
                                     expected.minimum_occupied_capture)
                    self.assertEqual(actual.maximum_overlap_condition,
                                     expected.maximum_overlap_condition)

    def test_default_and_explicit_scalar_path_are_identical(self):
        dataset, coefficients = self.fixture()
        a, loss_a, grad_a = self.evaluate(dataset, coefficients)
        b, loss_b, grad_b = self.evaluate(dataset, coefficients, frequency_batch_size=None)
        torch.testing.assert_close(a.response, b.response, rtol=0, atol=0)
        torch.testing.assert_close(loss_a.loss, loss_b.loss, rtol=0, atol=0)
        torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)

    def test_invalid_batch_sizes_fail_before_computation(self):
        dataset, coefficients = self.fixture()
        for batch in (0, -1, True, 1.5, "2"):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "batch"):
                self.evaluate(dataset, coefficients, frequency_batch_size=batch)

    def test_dense_and_block_batched_contractions_agree(self):
        dataset, coefficients = self.fixture()
        expected = evaluate_periodic_galerkin_coefficient_response(
            dataset, {"C": [coefficients]}, occupied_capture_tolerance=0.5)
        actual = evaluate_periodic_galerkin_coefficient_response(
            dataset, {"C": [coefficients]}, occupied_capture_tolerance=0.5,
            frequency_batch_size=5)
        torch.testing.assert_close(actual.response, expected.response, rtol=1e-11, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
