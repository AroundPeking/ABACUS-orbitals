from dataclasses import replace
import unittest
from unittest import mock

import torch

import common  # noqa: F401 - configures the optimizer import path
from projected_pi import (
    NormalizedPhysicalFamilyProjectedPi,
    ProjectedPiEvaluator,
    ProjectedPiFamilyResult,
)
from projected_pi_optimization import (
    NormalizedPhysicalFamilyProjectedPiOptimization,
)
from sternheimer_source_pair import pair_response_and_source
from test_projected_pi import coefficients, make_pair, scaled_pair


def modified_pair(pair, q_scale=1.0, frequencies=None, weights=None):
    response = pair.response
    replacements = {"q": response.q * q_scale}
    if frequencies is not None:
        replacements["frequency_ha"] = torch.tensor(
            frequencies, dtype=torch.float64
        )
    if weights is not None:
        replacements["frequency_weight"] = torch.tensor(
            weights, dtype=torch.float64
        )
    return pair_response_and_source(
        replace(response, **replacements), pair.source
    )


class ProjectedPiOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.h, _, _ = make_pair()
        self.h2 = modified_pair(self.h, q_scale=1.17)
        self.coefficient = coefficients()["H"][0]

    def adapter(self, *pairs, **kwargs):
        if not pairs:
            pairs = (("H", self.h), ("H2", self.h2))
        return NormalizedPhysicalFamilyProjectedPiOptimization(
            *pairs, **kwargs
        )

    def test_returns_equal_family_loss_and_frequency_diagnostics(self):
        result = self.adapter().evaluate(coefficients(self.coefficient))
        expected = NormalizedPhysicalFamilyProjectedPi(
            (("H", self.h), ("H2", self.h2))
        ).evaluate(coefficients(self.coefficient))

        torch.testing.assert_close(result.loss, expected.loss)
        torch.testing.assert_close(
            result.loss,
            expected.results["H"].loss + expected.results["H2"].loss,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(tuple(result.family_results), ("H", "H2"))
        torch.testing.assert_close(
            result.frequency_ha,
            expected.results["H"].frequency_ha,
        )
        torch.testing.assert_close(
            result.frequency_loss,
            (
                expected.results["H"].frequency_loss
                + expected.results["H2"].frequency_loss
            )
            / 2.0,
        )
        self.assertEqual(
            result.max_condition, expected.max_candidate_condition
        )
        self.assertEqual(result.lowest_frequency_ha, result.frequency_ha[0])
        self.assertEqual(
            result.lowest_frequency_loss, result.frequency_loss[0]
        )

    def test_accepts_any_two_unique_physical_family_names(self):
        result = self.adapter(
            ("C_atom", self.h), ("C2", self.h2)
        ).evaluate(coefficients(self.coefficient))

        self.assertEqual(tuple(result.family_results), ("C_atom", "C2"))
        reversed_result = self.adapter(
            ("C2", self.h2), ("C_atom", self.h)
        ).evaluate(coefficients(self.coefficient))
        self.assertEqual(
            tuple(reversed_result.family_results), ("C2", "C_atom")
        )
        torch.testing.assert_close(result.loss, reversed_result.loss)
        torch.testing.assert_close(
            result.frequency_loss, reversed_result.frequency_loss
        )

    def rpa_sensitive_pairs(self):
        h = scaled_pair(self.h)
        h2_q = h.response.q.clone()
        h2_q[0, 0] *= 1.31
        h2_q[-1, -1] *= 0.79
        h2 = pair_response_and_source(
            replace(h.response, q=h2_q), h.source
        )
        return h, h2

    def test_rpa_sensitive_family_loss_uses_fourth_order_norm(self):
        h, h2 = self.rpa_sensitive_pairs()
        candidate = coefficients(self.coefficient, requires_grad=True)
        direct_h = ProjectedPiEvaluator(
            h, sensitivity_alpha=0.25
        ).evaluate(candidate)
        direct_h2 = ProjectedPiEvaluator(
            h2, sensitivity_alpha=0.25
        ).evaluate(candidate)
        self.assertGreater(float(direct_h.loss.detach()), 0.0)
        self.assertGreater(float(direct_h2.loss.detach()), 0.0)
        self.assertGreater(
            abs(float((direct_h.loss - direct_h2.loss).detach())),
            1.0e-10,
        )

        result = self.adapter(
            ("H", h),
            ("H2", h2),
            sensitivity_alpha=0.25,
            family_power=4,
        ).evaluate(candidate)
        expected = (
            result.family_results["H"].loss.pow(4)
            + result.family_results["H2"].loss.pow(4)
        ).pow(0.25)

        torch.testing.assert_close(result.loss, expected)
        self.assertEqual(result.sensitivity_alpha, 0.25)
        self.assertEqual(result.family_power, 4)
        self.assertEqual(
            result.family_results["H"].sensitivity_alpha, 0.25
        )
        self.assertEqual(
            result.family_results["H2"].sensitivity_alpha, 0.25
        )
        torch.testing.assert_close(
            result.family_results["H"].loss, direct_h.loss
        )
        torch.testing.assert_close(
            result.family_results["H2"].loss, direct_h2.loss
        )

    def test_rpa_sensitive_zero_family_losses_have_zero_finite_gradient(self):
        overlap = torch.eye(3, dtype=torch.complex128)
        h = scaled_pair(self.h)
        h = pair_response_and_source(
            replace(h.response, overlap=overlap),
            replace(h.source, overlap=overlap),
        )
        h2 = pair_response_and_source(
            replace(h.response, q=h.response.q * 1.17),
            h.source,
        )
        candidate = coefficients(
            torch.eye(3, dtype=torch.float64),
            requires_grad=True,
        )

        result = self.adapter(
            ("H", h),
            ("H2", h2),
            sensitivity_alpha=0.25,
            family_power=4,
        ).evaluate(candidate)

        self.assertEqual(float(result.loss.detach()), 0.0)
        for family in ("H", "H2"):
            self.assertEqual(
                float(result.family_results[family].loss.detach()),
                0.0,
            )
        result.loss.backward()
        for element_coefficients in candidate.values():
            for coefficient in element_coefficients:
                self.assertTrue(
                    bool(torch.all(torch.isfinite(coefficient.grad)))
                )
                self.assertEqual(int(torch.count_nonzero(coefficient.grad)), 0)

    def test_rpa_sensitive_fourth_order_gradient_matches_centered_difference(self):
        h, h2 = self.rpa_sensitive_pairs()
        candidate = coefficients(self.coefficient, requires_grad=True)
        result = self.adapter(
            ("H", h),
            ("H2", h2),
            sensitivity_alpha=0.25,
            family_power=4,
        ).evaluate(candidate)
        result.loss.backward()
        analytic = float(candidate["H"][0].grad[1, 0])
        self.assertGreater(abs(analytic), 1.0e-8)

        step = 1.0e-6
        plus = self.coefficient.clone()
        minus = self.coefficient.clone()
        plus[1, 0] += step
        minus[1, 0] -= step
        finite_difference = float(
            (
                self.adapter(
                    ("H", h),
                    ("H2", h2),
                    sensitivity_alpha=0.25,
                    family_power=4,
                ).evaluate(coefficients(plus)).loss
                - self.adapter(
                    ("H", h),
                    ("H2", h2),
                    sensitivity_alpha=0.25,
                    family_power=4,
                ).evaluate(coefficients(minus)).loss
            )
            / (2.0 * step)
        )
        self.assertAlmostEqual(analytic, finite_difference, delta=3.0e-7)

    def test_rpa_sensitive_rejects_family_power_other_than_exactly_four(self):
        h, h2 = self.rpa_sensitive_pairs()
        for family_power in (
            1,
            2,
            3,
            5,
            4.5,
            float("nan"),
            float("inf"),
            "4",
            None,
            True,
        ):
            with self.subTest(family_power=family_power):
                with self.assertRaisesRegex(
                    ValueError, "family_power.*exactly 4"
                ):
                    self.adapter(
                        ("H", h),
                        ("H2", h2),
                        sensitivity_alpha=0.25,
                        family_power=family_power,
                    )

    def test_rpa_sensitive_adapter_rejects_bool_and_string_alpha(self):
        h, h2 = self.rpa_sensitive_pairs()
        for alpha in (True, False, "0.25"):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    TypeError,
                    "sensitivity_alpha must be a finite real number",
                ):
                    self.adapter(
                        ("H", h),
                        ("H2", h2),
                        sensitivity_alpha=alpha,
                        family_power=4,
                    )

    def test_coefficient_gradient_matches_centered_difference(self):
        candidate = coefficients(self.coefficient, requires_grad=True)
        result = self.adapter().evaluate(candidate)
        result.loss.backward()
        analytic = float(candidate["H"][0].grad[1, 0])

        step = 1.0e-6
        plus = self.coefficient.clone()
        minus = self.coefficient.clone()
        plus[1, 0] += step
        minus[1, 0] -= step
        finite_difference = float(
            (
                self.adapter().evaluate(coefficients(plus)).loss
                - self.adapter().evaluate(coefficients(minus)).loss
            )
            / (2.0 * step)
        )
        self.assertAlmostEqual(analytic, finite_difference, delta=3.0e-7)

    def test_rejects_wrong_duplicate_or_empty_families(self):
        for pairs in (
            (("H", self.h),),
            (("H", self.h), ("H", self.h2)),
            (("H", self.h), ("", self.h2)),
            (("H", self.h), ("H2", self.h2), ("H2", self.h2)),
        ):
            with self.subTest(pairs=tuple(name for name, _ in pairs)):
                with self.assertRaisesRegex(ValueError, "exactly two unique"):
                    self.adapter(*pairs)

    def test_rejects_unequal_frequency_grids_or_weights(self):
        unequal_grid = modified_pair(
            self.h,
            frequencies=[0.5, 0.5, 2.0, 2.0],
        )
        unequal_weight = modified_pair(
            self.h,
            weights=[0.4, 0.4, 0.6, 0.6],
        )
        for pair, message in (
            (unequal_grid, "frequency grids differ"),
            (unequal_weight, "frequency weights differ"),
        ):
            with self.subTest(message=message):
                adapter = self.adapter(("H", self.h), ("H2", pair))
                with self.assertRaisesRegex(ValueError, message):
                    adapter.evaluate(coefficients(self.coefficient))

    def test_rejects_nonfinite_loss_and_excessive_condition(self):
        adapter = self.adapter()
        adapter._family = mock.Mock()
        adapter._family.evaluate.return_value = ProjectedPiFamilyResult(
            loss=torch.tensor(float("nan"), dtype=torch.float64),
            results={},
            max_candidate_condition=1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "must be finite"):
            adapter.evaluate(coefficients(self.coefficient))

        with self.assertRaisesRegex(RuntimeError, "condition number"):
            self.adapter(condition_limit=1.0).evaluate(
                coefficients(self.coefficient)
            )


if __name__ == "__main__":
    unittest.main()
