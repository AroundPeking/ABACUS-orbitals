import math
import unittest
from dataclasses import replace

import torch

import common  # noqa: F401
import test_periodic_galerkin_fit as fixtures
from periodic_galerkin_fit import optimize_periodic_galerkin_basis
from periodic_galerkin_optimization import (
    evaluate_periodic_galerkin_coefficient_response,
)
from periodic_galerkin_rpa import periodic_rpa_objective
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


class PeriodicGalerkinRpaFitTest(unittest.TestCase):
    def fixture(self):
        dataset = fixtures.PeriodicGalerkinFitTest().three_level_dataset()
        dataset = replace(dataset, q_count=8, q_weight=0.25)
        initial = {
            "C": [
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
            ]
        }
        return (dataset,), initial

    def run_fit(self, **options):
        datasets, initial = self.fixture()
        return optimize_periodic_galerkin_basis(
            datasets,
            initial,
            fixed_nu={"C": (1,)},
            learning_rate=0.02,
            max_steps=3,
            minimum_steps=0,
            plateau_patience=4,
            **options
        )

    def test_rpa_iteration_improves_objective_and_records_true_pi_error(self):
        result = self.run_fit(objective="rpa")
        self.assertLess(result.best_loss, result.initial_loss)
        self.assertEqual(result.steps_completed, 3)
        self.assertEqual(result.objective, "rpa")
        self.assertEqual(
            result.objective_weights,
            {"pi_weight": 1.0, "trace_log_weight": 1.0, "energy_weight": 1.0},
        )
        for entry in result.history:
            self.assertEqual(entry["objective"], "rpa")
            details = entry["rpa"]
            self.assertAlmostEqual(
                entry["relative_pi_error"] ** 2, details["pi_relative_squared_error"]
            )
            self.assertNotAlmostEqual(entry["loss"], entry["relative_pi_error"] ** 2)
            self.assertEqual(details["q_weight_coverage"], 0.25)
            self.assertFalse(details["complete_q_weight"])
            self.assertTrue(math.isfinite(details["candidate_energy_ha"]))
            self.assertGreaterEqual(details["evaluation_seconds"], 0.0)
        self.assertTrue(
            all(
                math.isfinite(x["previous_step_gradient_norm"])
                for x in result.history[1:]
            )
        )
        torch.testing.assert_close(
            result.coefficients["C"][0][:, 0],
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

    def test_best_checkpoint_reproduces_direct_rpa_objective(self):
        weights = {"pi_weight": 0.5, "trace_log_weight": 2.0, "energy_weight": 3.0}
        result = self.run_fit(objective="rpa", rpa_weights=weights)
        datasets, _ = self.fixture()
        datasets = tuple(prepare_periodic_occupied_reference(d) for d in datasets)
        responses = tuple(
            evaluate_periodic_galerkin_coefficient_response(
                d, result.coefficients
            ).response
            for d in datasets
        )
        expected = periodic_rpa_objective(datasets, responses, **weights)
        self.assertAlmostEqual(result.best_loss, float(expected.loss), places=12)
        self.assertAlmostEqual(
            result.history[result.best_step]["rpa"]["candidate_energy_ha"],
            float(expected.candidate_energy_ha),
            places=12,
        )
        self.assertEqual(result.objective_weights, weights)
        weights["energy_weight"] = 99.0
        self.assertEqual(result.objective_weights["energy_weight"], 3.0)

    def test_default_pi_path_is_unchanged(self):
        implicit = self.run_fit()
        explicit = self.run_fit(objective="pi")
        self.assertEqual(implicit.history, explicit.history)
        self.assertEqual(implicit.best_loss, explicit.best_loss)
        self.assertEqual(implicit.objective, "pi")
        torch.testing.assert_close(
            implicit.coefficients["C"][0],
            explicit.coefficients["C"][0],
            rtol=0.0,
            atol=0.0,
        )

    def test_multiple_q_contributions_sum_to_integrated_energy(self):
        datasets, initial = self.fixture()
        other = replace(datasets[0], selected_iq=2, q_weight=0.75)
        result = optimize_periodic_galerkin_basis(
            datasets + (other,),
            initial,
            fixed_nu={"C": (1,)},
            objective="rpa",
            max_steps=2,
            minimum_steps=0,
            learning_rate=0.02,
            plateau_patience=3,
        )
        self.assertLess(result.best_loss, result.initial_loss)
        for entry in result.history:
            details = entry["rpa"]
            self.assertTrue(details["complete_q_weight"])
            self.assertEqual(len(details["per_q"]), 2)
            for side in ("candidate", "reference"):
                energy = sum(
                    sum(record[side + "_contributions_ha"])
                    for record in details["per_q"]
                )
                self.assertAlmostEqual(energy, details[side + "_energy_ha"], places=14)

    def test_rpa_rejects_silent_atom_family_or_weight_mixing(self):
        for options in (
            {"dataset_families": ("C_solid",)},
            {"additional_family_evaluators": {"C_atom": object()}},
            {"rpa_weights": {"unexpected": 1.0}},
            {"rpa_weights": {"pi_weight": 0.0}},
            {"rpa_weights": {"energy_weight": float("nan")}},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_fit(objective="rpa", **options)
        with self.assertRaises(ValueError):
            self.run_fit(objective="unknown")
        with self.assertRaises(ValueError):
            self.run_fit(objective="pi", rpa_weights={"energy_weight": 1.0})


if __name__ == "__main__":
    unittest.main()
