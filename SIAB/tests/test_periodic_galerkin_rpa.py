"""RPA objective tests use frozen finite-space Pi, not a new SCF/LRI path."""

from dataclasses import replace
import unittest

import torch

import common  # noqa: F401
import periodic_galerkin_rpa as rpa
import test_periodic_galerkin_sos_equivalence as oracle


class PeriodicGalerkinRpaTest(unittest.TestCase):
    def fixture(self):
        dataset, coordinates = oracle.PeriodicGalerkinSosEquivalenceTest().fixture()
        reference = oracle.spectral_sum(dataset, torch.eye(5, dtype=torch.complex128))
        return replace(dataset, reference_response=reference), coordinates

    def test_integral_matches_independent_all_band_oracle(self):
        dataset, _ = self.fixture()
        result = rpa.periodic_rpa_objective((dataset,), (dataset.reference_response,))
        expected = oracle.correlation_integral(dataset, dataset.reference_response)
        torch.testing.assert_close(result.candidate_energy_ha, expected)
        torch.testing.assert_close(result.reference_energy_ha, expected)
        self.assertEqual(float(result.loss), 0.0)
        self.assertEqual(result.q_weight_coverage, 0.125)
        self.assertFalse(result.complete_q_weight)
        self.assertEqual(result.q_records[0].selected_iq, dataset.selected_iq)
        self.assertEqual(result.q_records[0].candidate_raw.shape, (12,))

    def test_q_weight_applied_once_without_partial_grid_extrapolation(self):
        dataset, _ = self.fixture()
        first = replace(dataset, q_weight=0.25)
        second = replace(dataset, selected_iq=dataset.selected_iq + 1, q_weight=0.75)
        result = rpa.periodic_rpa_objective(
            (first, second), (first.reference_response, second.reference_response)
        )
        expected = oracle.correlation_integral(
            replace(dataset, q_weight=1.0), dataset.reference_response
        )
        torch.testing.assert_close(result.candidate_energy_ha, expected)
        self.assertTrue(result.complete_q_weight)
        pieces = torch.stack(
            tuple(x.candidate_contributions_ha.sum() for x in result.q_records)
        )
        torch.testing.assert_close(pieces.sum(), result.candidate_energy_ha)

    def test_zero_total_energy_error_cannot_hide_opposite_q_errors(self):
        dataset, _ = self.fixture()
        a = -0.1 * torch.eye(3, dtype=torch.complex128).repeat(12, 1, 1)
        b = -0.3 * torch.eye(3, dtype=torch.complex128).repeat(12, 1, 1)
        first = replace(dataset, q_weight=0.5, reference_response=a)
        second = replace(
            dataset,
            selected_iq=dataset.selected_iq + 1,
            q_weight=0.5,
            reference_response=b,
        )
        result = rpa.periodic_rpa_objective((first, second), (b, a))
        self.assertAlmostEqual(
            float(result.energy_relative_squared_error), 0.0, places=25
        )
        self.assertGreater(float(result.trace_log_relative_squared_error), 0.0)
        self.assertGreater(float(result.pi_relative_squared_error), 0.0)
        self.assertGreater(float(result.loss), 0.0)

    def test_zero_total_energy_error_cannot_hide_frequency_errors(self):
        dataset, _ = self.fixture()
        values = torch.linspace(0.1, 0.4, 12, dtype=torch.float64)
        ref = -values[:, None, None] * torch.eye(3, dtype=torch.complex128)
        dataset = replace(
            dataset,
            reference_response=ref,
            frequency_weights_ha=torch.ones(12, dtype=torch.float64),
        )
        result = rpa.periodic_rpa_objective((dataset,), (ref.flip(0),))
        self.assertLess(float(result.energy_relative_squared_error), 1e-25)
        self.assertGreater(float(result.trace_log_relative_squared_error), 0.0)

    def test_gradient_matches_independent_spectral_finite_difference(self):
        dataset, coordinates = self.fixture()
        angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        fixture = oracle.PeriodicGalerkinSosEquivalenceTest()

        def loss_at(value, spectral=False):
            transform = fixture.reduced_transform(coordinates, value)
            if spectral:
                response = oracle.spectral_sum(dataset, transform)
            else:
                response = oracle.evaluate_periodic_galerkin_response(
                    dataset, transform
                ).response
            return rpa.periodic_rpa_objective((dataset,), (response,)).loss

        loss_at(angle).backward()
        eps = 1e-5
        finite_difference = (
            loss_at(angle.detach() + eps, True) - loss_at(angle.detach() - eps, True)
        ) / (2 * eps)
        torch.testing.assert_close(angle.grad, finite_difference, rtol=2e-6, atol=1e-9)
        self.assertGreater(abs(float(angle.grad)), 1e-7)

    def test_tiny_high_frequency_tail_retains_second_order_term(self):
        pi = torch.tensor([[[-1e-12 + 0j]]], dtype=torch.complex128, requires_grad=True)
        raw, trace, logdet = rpa.rpa_trace_log(pi)
        expected = -0.5e-24 + (1e-36 / 3)
        self.assertAlmostEqual(float(raw) / expected, 1.0, places=12)
        raw.sum().backward()
        self.assertAlmostEqual(float(pi.grad.real) / 1e-12, 1.0, places=10)
        self.assertTrue(
            bool(torch.isfinite(trace).all() & torch.isfinite(logdet).all())
        )

    def test_repeated_pi_eigenvalues_have_finite_gradient(self):
        pi = (
            -0.2 * torch.eye(3, dtype=torch.complex128).repeat(12, 1, 1)
        ).requires_grad_()
        raw, _, _ = rpa.rpa_trace_log(pi)
        raw.sum().backward()
        expected = (1.0 / 6) * torch.eye(3, dtype=torch.complex128).repeat(12, 1, 1)
        torch.testing.assert_close(pi.grad, expected)

    def test_rejects_invalid_pi_instead_of_flipping_sign_or_clipping(self):
        invalid = (
            torch.tensor([[[1.1 + 0j]]], dtype=torch.complex128),
            torch.tensor([[[0.2 + 0j]]], dtype=torch.complex128),
            torch.tensor([[[complex(float("nan"), 0)]]], dtype=torch.complex128),
            torch.tensor([[[0.0, 1.0], [0.0, -1.0]]], dtype=torch.complex128),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                rpa.rpa_trace_log(value)

    def test_rejects_duplicate_q_mismatched_protocol_and_overweight(self):
        dataset, _ = self.fixture()
        changes = (
            {},
            {"selected_iq": dataset.selected_iq + 1, "orbital_sha256": "different"},
            {
                "selected_iq": dataset.selected_iq + 1,
                "frequency_weights_ha": dataset.frequency_weights_ha * 2,
            },
            {"selected_iq": dataset.selected_iq + 1, "q_weight": 1.0},
        )
        for change in changes:
            second = replace(dataset, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                rpa.periodic_rpa_objective(
                    (dataset, second),
                    (dataset.reference_response, second.reference_response),
                )

    def test_zero_reference_and_loss_weights_fail_explicitly(self):
        dataset, _ = self.fixture()
        with self.assertRaises(ValueError):
            rpa.periodic_rpa_objective(
                (
                    replace(
                        dataset,
                        reference_response=torch.zeros_like(dataset.reference_response),
                    ),
                ),
                (dataset.reference_response,),
            )
        for weight in (-1.0, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                rpa.periodic_rpa_objective(
                    (dataset,), (dataset.reference_response,), energy_weight=weight
                )
        with self.assertRaises(ValueError):
            rpa.periodic_rpa_objective(
                (dataset,),
                (dataset.reference_response,),
                pi_weight=0,
                trace_log_weight=0,
            )

    def test_q_dependent_whitened_rank_is_supported(self):
        dataset, _ = self.fixture()
        smaller = replace(
            dataset,
            selected_iq=dataset.selected_iq + 1,
            whitened_auxiliary_rank=2,
            reference_response=dataset.reference_response[:, :2, :2],
        )
        result = rpa.periodic_rpa_objective(
            (dataset, smaller), (dataset.reference_response, smaller.reference_response)
        )
        expected = oracle.correlation_integral(dataset, dataset.reference_response)
        expected += oracle.correlation_integral(smaller, smaller.reference_response)
        torch.testing.assert_close(result.candidate_energy_ha, expected)
        self.assertEqual(float(result.loss), 0.0)

    def test_invalid_frequency_dimensions_and_weights_fail(self):
        dataset, _ = self.fixture()
        cases = (
            replace(dataset, frequency_weights_ha=dataset.frequency_weights_ha[:3]),
            replace(dataset, frequency_weights_ha=-dataset.frequency_weights_ha),
            replace(dataset, frequency_ha=dataset.frequency_ha.flip(0)),
            replace(dataset, q_weight=float("nan")),
        )
        for case in cases:
            with self.subTest(case=case.q_weight), self.assertRaises(ValueError):
                rpa.periodic_rpa_objective((case,), (case.reference_response,))


if __name__ == "__main__":
    unittest.main()
