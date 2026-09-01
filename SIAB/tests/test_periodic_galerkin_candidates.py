import unittest
from unittest import mock

import torch

import common  # noqa: F401 - configures the optimizer import path
import periodic_galerkin_candidates
import periodic_galerkin_fit
from periodic_galerkin_candidates import PeriodicGalerkinFamilyGradientResult
from test_periodic_galerkin_fit import PeriodicGalerkinFitTest


class PeriodicGalerkinCandidatesTest(unittest.TestCase):
    def test_projects_gradient_to_fixed_prefix_stiefel_tangent(self):
        coefficients = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }
        gradient = {
            "C": [
                torch.tensor(
                    [[4.0, 1.0], [3.0, 2.0], [5.0, 6.0]],
                    dtype=torch.float64,
                )
            ]
        }

        projected = periodic_galerkin_candidates.project_fixed_prefix_tangent(
            coefficients,
            fixed_nu={"C": (1,)},
            gradient=gradient,
        )

        expected = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 6.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(projected["C"][0], expected)
        fixed = coefficients["C"][0][:, :1]
        variable = coefficients["C"][0][:, 1:]
        direction = projected["C"][0][:, 1:]
        torch.testing.assert_close(
            fixed.transpose(0, 1).matmul(direction),
            torch.zeros((1, 1), dtype=torch.float64),
        )
        torch.testing.assert_close(
            variable.transpose(0, 1).matmul(direction)
            + direction.transpose(0, 1).matmul(variable),
            torch.zeros((1, 1), dtype=torch.float64),
        )

    def test_evaluates_named_family_gradients_in_declaration_order(self):
        dataset = PeriodicGalerkinFitTest().three_level_dataset()
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        def family_loss(
            _datasets,
            coefficients,
            *,
            occupied_capture_tolerance,
            dataset_families,
            additional_family_evaluators,
        ):
            self.assertEqual(dataset_families, ("C_solid",))
            self.assertEqual(tuple(additional_family_evaluators), ("C_atom",))
            coordinate = coefficients["C"][0][2, 1]
            atom = (coordinate - 1.0) ** 2
            solid = (coordinate + 2.0) ** 2
            return (
                0.5 * (solid + atom),
                0.999,
                3.0,
                {"C_solid": solid, "C_atom": atom},
            )

        with mock.patch.object(
            periodic_galerkin_fit,
            "_global_pi_loss",
            side_effect=family_loss,
        ), mock.patch.object(
            periodic_galerkin_candidates,
            "_global_pi_loss",
            side_effect=family_loss,
        ):
            result = periodic_galerkin_candidates.evaluate_family_gradients(
                (dataset,),
                initial,
                fixed_nu={"C": (1,)},
                dataset_families=("C_solid",),
                additional_family_evaluators={"C_atom": object()},
            )

        self.assertEqual(result.family_order, ("C_solid", "C_atom"))
        self.assertEqual(result.family_losses, {"C_solid": 4.0, "C_atom": 1.0})
        self.assertEqual(result.minimum_occupied_capture, 0.999)
        self.assertEqual(result.maximum_overlap_condition, 3.0)
        self.assertAlmostEqual(result.gradient_norms["C_solid"], 4.0)
        self.assertAlmostEqual(result.gradient_norms["C_atom"], 2.0)
        torch.testing.assert_close(
            result.gradients["C_solid"]["C"][0],
            torch.tensor(
                [[0.0, 0.0], [0.0, 0.0], [0.0, 4.0]],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            result.gradients["C_atom"]["C"][0],
            torch.tensor(
                [[0.0, 0.0], [0.0, 0.0], [0.0, -2.0]],
                dtype=torch.float64,
            ),
        )
        self.assertAlmostEqual(
            result.gradient_cosines["C_solid:C_atom"],
            -1.0,
        )

    def test_rejects_a_zero_family_gradient(self):
        coefficients = {"C": [torch.eye(2, dtype=torch.float64)]}
        gradient = {"C": [torch.zeros((2, 2), dtype=torch.float64)]}

        with self.assertRaisesRegex(RuntimeError, "gradient norm is zero"):
            periodic_galerkin_candidates.normalize_gradient(
                gradient,
                family="C_atom",
            )

    def candidate_gradient_result(self):
        coefficients = {
            "C": [
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float64,
                )
            ]
        }
        atom = {"C": [torch.zeros((4, 3), dtype=torch.float64)]}
        solid = {"C": [torch.zeros((4, 3), dtype=torch.float64)]}
        atom["C"][0][3, 1] = 1.0
        solid["C"][0][3, 2] = 1.0
        return PeriodicGalerkinFamilyGradientResult(
            coefficients=coefficients,
            family_order=("C_atom", "C_solid"),
            family_losses={"C_atom": 1.0, "C_solid": 2.0},
            gradients={"C_atom": atom, "C_solid": solid},
            normalized_gradients={"C_atom": atom, "C_solid": solid},
            gradient_norms={"C_atom": 1.0, "C_solid": 1.0},
            gradient_cosines={"C_atom:C_solid": 0.0},
            minimum_occupied_capture=0.999,
            maximum_overlap_condition=2.0,
        )

    def test_builds_deterministic_three_weight_pareto_bank(self):
        result = self.candidate_gradient_result()

        first = periodic_galerkin_candidates.build_pareto_candidate_bank(
            result,
            fixed_nu={"C": (1,)},
            family_pair=("C_atom", "C_solid"),
            weights=(0.25, 0.50, 0.75),
            trust_radius=0.02,
        )
        second = periodic_galerkin_candidates.build_pareto_candidate_bank(
            result,
            fixed_nu={"C": (1,)},
            family_pair=("C_atom", "C_solid"),
            weights=(0.25, 0.50, 0.75),
            trust_radius=0.02,
        )

        self.assertEqual([item.weight for item in first], [0.25, 0.5, 0.75])
        self.assertEqual(
            [item.coefficients_sha256 for item in first],
            [item.coefficients_sha256 for item in second],
        )
        self.assertEqual(len(set(item.coefficients_sha256 for item in first)), 3)
        fixed = result.coefficients["C"][0][:, :1]
        for item in first:
            torch.testing.assert_close(item.coefficients["C"][0][:, :1], fixed)
            variable = item.coefficients["C"][0][:, 1:]
            torch.testing.assert_close(
                variable.transpose(0, 1).matmul(variable),
                torch.eye(2, dtype=torch.float64),
                rtol=1.0e-13,
                atol=1.0e-13,
            )
            self.assertAlmostEqual(item.trust_radius, 0.02)

    def test_candidate_family_gate_requires_improvement_without_large_tradeoff(self):
        baseline = {"C_atom": 1.0, "C_solid": 2.0}

        accepted = periodic_galerkin_candidates.assess_family_tradeoff(
            baseline,
            {"C_atom": 0.9, "C_solid": 2.02},
            maximum_relative_degradation=0.02,
        )
        rejected = periodic_galerkin_candidates.assess_family_tradeoff(
            baseline,
            {"C_atom": 0.9, "C_solid": 2.05},
            maximum_relative_degradation=0.02,
        )
        unchanged = periodic_galerkin_candidates.assess_family_tradeoff(
            baseline,
            dict(baseline),
            maximum_relative_degradation=0.02,
        )

        self.assertEqual(accepted["gate"], "pass")
        self.assertEqual(rejected["gate"], "fail")
        self.assertIn("C_solid", rejected["degraded_families"])
        self.assertEqual(unchanged["gate"], "fail")
        self.assertEqual(unchanged["failure_reasons"], ["no_family_improved"])

    def test_reuses_prepared_problem_to_evaluate_candidate_family_losses(self):
        dataset = PeriodicGalerkinFitTest().three_level_dataset()
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        def family_loss(_datasets, coefficients, **_kwargs):
            coordinate = coefficients["C"][0][2, 1]
            solid = (coordinate + 2.0) ** 2
            atom = (coordinate - 1.0) ** 2
            return 0.5 * (solid + atom), 0.998, 4.0, {
                "C_solid": solid,
                "C_atom": atom,
            }

        with mock.patch.object(
            periodic_galerkin_candidates,
            "_global_pi_loss",
            side_effect=family_loss,
        ):
            result = periodic_galerkin_candidates.evaluate_family_gradients(
                (dataset,),
                initial,
                fixed_nu={"C": (1,)},
                dataset_families=("C_solid",),
                additional_family_evaluators={"C_atom": object()},
            )
            trial = {
                "C": [
                    torch.tensor(
                        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.2]],
                        dtype=torch.float64,
                    )
                ]
            }
            evaluation = (
                periodic_galerkin_candidates.evaluate_candidate_family_losses(
                    result,
                    trial,
                )
            )

        self.assertAlmostEqual(evaluation["family_losses"]["C_solid"], 4.84)
        self.assertAlmostEqual(evaluation["family_losses"]["C_atom"], 0.64)
        self.assertEqual(evaluation["minimum_occupied_capture"], 0.998)
        self.assertEqual(evaluation["maximum_overlap_condition"], 4.0)


if __name__ == "__main__":
    unittest.main()
