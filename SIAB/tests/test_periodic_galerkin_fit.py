import unittest
from dataclasses import replace
from unittest import mock

import torch

import common  # noqa: F401 - configures the optimizer import path
import periodic_galerkin_fit
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock
from periodic_galerkin_fit import optimize_periodic_galerkin_basis
from test_periodic_galerkin_sternheimer import PeriodicGalerkinSternheimerTest


class PeriodicGalerkinFitTest(unittest.TestCase):
    def three_level_dataset(self):
        dataset, _, _ = PeriodicGalerkinSternheimerTest().complete_two_level_dataset()
        omega = float(dataset.frequency_ha[0])
        delta_1 = -0.3 / (1.2 + 1.0j * omega)
        delta_2 = -0.2 / (1.9 + 1.0j * omega)
        response = 2.0 * (0.3 * delta_1 + 0.2 * delta_2)
        response = response + response.conjugate()
        record = replace(
            dataset.kpoints[0],
            overlap=torch.eye(3, dtype=torch.complex128),
            hamiltonian_ha=torch.diag(
                torch.tensor([-0.5, 0.7, 1.4], dtype=torch.float64)
            ).to(torch.complex128),
            occupied_projection=torch.tensor(
                [[1.0, 0.0, 0.0]], dtype=torch.complex128
            ),
            source=torch.tensor([[[0.0, 0.3, 0.2]]], dtype=torch.complex128),
            reference_projection=torch.zeros((1, 1, 1, 3), dtype=torch.complex128),
        )
        return replace(
            dataset,
            primitive_count=3,
            primitive_blocks=(
                PeriodicGalerkinPrimitiveBlock("C", 0, 0, 0, 3, 0),
            ),
            reference_response=torch.tensor([[[response]]], dtype=torch.complex128),
            kpoints=(record,),
        )

    def test_optimizes_virtual_radial_direction_without_changing_fixed_column(self):
        dataset = self.three_level_dataset()
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }
        fixed = initial["C"][0][:, 0].clone()
        progress = []

        result = optimize_periodic_galerkin_basis(
            (dataset,),
            initial,
            fixed_nu={"C": (1,)},
            learning_rate=0.05,
            max_steps=120,
            minimum_steps=30,
            plateau_patience=40,
            plateau_relative_improvement=1.0e-8,
            progress_callback=progress.append,
        )

        self.assertLess(result.best_loss, 0.01 * result.initial_loss)
        self.assertTrue(torch.equal(result.coefficients["C"][0][:, 0], fixed))
        self.assertGreater(abs(float(result.coefficients["C"][0][2, 1])), 0.01)
        self.assertIn(result.stop_reason, ("plateau", "maximum_steps"))
        self.assertEqual(tuple(progress), result.history)

    def test_backtracks_at_fixed_occupied_capture_boundary(self):
        dataset = self.three_level_dataset()
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        def constrained_loss(_datasets, coefficients):
            coordinate = coefficients["C"][0][2, 1]
            if abs(float(coordinate.detach())) > 0.1:
                raise RuntimeError(
                    "candidate basis does not capture the fixed occupied manifold"
                )
            loss = (coordinate - 1.0) ** 2
            return loss, 1.0 - float(coordinate.detach()) ** 2, 1.0

        with mock.patch.object(
            periodic_galerkin_fit,
            "_global_pi_loss",
            side_effect=constrained_loss,
        ):
            result = optimize_periodic_galerkin_basis(
                (dataset,),
                initial,
                fixed_nu={"C": (1,)},
                learning_rate=0.2,
                max_steps=20,
                minimum_steps=0,
                plateau_patience=20,
                plateau_relative_improvement=1.0e-8,
            )

        coordinate = float(result.coefficients["C"][0][2, 1])
        self.assertLessEqual(abs(coordinate), 0.1 + 1.0e-12)
        self.assertLess(result.best_loss, result.initial_loss)
        self.assertIn(
            result.stop_reason,
            ("occupied_capture_boundary", "maximum_steps"),
        )

    def test_rejects_fixed_prefix_larger_than_candidate_channel(self):
        dataset = self.three_level_dataset()
        initial = {"C": [torch.eye(3, 2, dtype=torch.float64)]}

        with self.assertRaisesRegex(ValueError, "fixed_nu"):
            optimize_periodic_galerkin_basis(
                (dataset,),
                initial,
                fixed_nu={"C": (3,)},
                max_steps=1,
                minimum_steps=0,
                plateau_patience=1,
            )


if __name__ == "__main__":
    unittest.main()
