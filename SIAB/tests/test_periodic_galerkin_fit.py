import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch

import common  # noqa: F401 - configures the optimizer import path
import periodic_galerkin_basis
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

    def test_best_callback_receives_independent_improving_snapshots(self):
        dataset = self.three_level_dataset()
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }
        checkpoints = []

        def record_best(step, loss, coefficients):
            checkpoints.append((step, loss, coefficients))
            coefficients["C"][0].fill_(99.0)

        result = optimize_periodic_galerkin_basis(
            (dataset,),
            initial,
            fixed_nu={"C": (1,)},
            learning_rate=0.05,
            max_steps=8,
            minimum_steps=0,
            plateau_patience=8,
            plateau_relative_improvement=1.0e-8,
            best_callback=record_best,
        )

        self.assertGreater(len(checkpoints), 1)
        self.assertEqual(checkpoints[-1][0], result.best_step)
        self.assertEqual(checkpoints[-1][1], result.best_loss)
        self.assertTrue(
            torch.equal(
                checkpoints[-1][2]["C"][0],
                torch.full_like(checkpoints[-1][2]["C"][0], 99.0),
            )
        )
        self.assertFalse(
            torch.equal(
                result.coefficients["C"][0],
                checkpoints[-1][2]["C"][0],
            )
        )

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

        tolerances = []

        def constrained_loss(
            _datasets,
            coefficients,
            *,
            occupied_capture_tolerance,
        ):
            coordinate = coefficients["C"][0][2, 1]
            capture = 0.9 - float(coordinate.detach()) ** 2
            tolerances.append(occupied_capture_tolerance)
            if capture < 1.0 - occupied_capture_tolerance:
                raise RuntimeError(
                    "candidate basis does not capture the fixed occupied manifold"
                )
            loss = (coordinate - 1.0) ** 2
            return loss, capture, 1.0, {"periodic": loss}

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
        self.assertLessEqual(abs(coordinate), (1.0e-8) ** 0.5 + 1.0e-12)
        self.assertLess(result.best_loss, result.initial_loss)
        self.assertAlmostEqual(result.initial_minimum_occupied_capture, 0.9)
        self.assertAlmostEqual(result.occupied_capture_floor, 0.9 - 1.0e-8)
        self.assertGreater(tolerances[0], 0.999999999)
        self.assertTrue(
            all(
                abs(value - (0.1 + 1.0e-8)) < 1.0e-14
                for value in tolerances[1:]
            )
        )
        self.assertIn(
            result.stop_reason,
            ("occupied_capture_boundary", "maximum_steps"),
        )

    def test_fixed_prefix_reference_sets_floor_from_immutable_basis(self):
        dataset = self.three_level_dataset()
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        def smooth_loss(
            _datasets,
            coefficients,
            *,
            occupied_capture_tolerance,
        ):
            coordinate = coefficients["C"][0][2, 1]
            capture = 0.9 - 0.01 * float(coordinate.detach()) ** 2
            if capture < 1.0 - occupied_capture_tolerance:
                raise RuntimeError(
                    "candidate basis does not capture the fixed occupied manifold"
                )
            loss = (coordinate - 1.0) ** 2
            return loss, capture, 1.0, {"periodic": loss}

        with mock.patch.object(
            periodic_galerkin_fit,
            "_minimum_occupied_capture",
            return_value=0.8,
        ) as fixed_capture, mock.patch.object(
            periodic_galerkin_fit,
            "_global_pi_loss",
            side_effect=smooth_loss,
        ):
            result = optimize_periodic_galerkin_basis(
                (dataset,),
                initial,
                fixed_nu={"C": (1,)},
                learning_rate=0.1,
                max_steps=1,
                minimum_steps=0,
                plateau_patience=1,
                plateau_relative_improvement=1.0e-8,
                occupied_capture_reference="fixed_prefix",
            )

        fixed_capture.assert_called_once()
        self.assertEqual(result.occupied_capture_reference, "fixed_prefix")
        self.assertAlmostEqual(result.reference_minimum_occupied_capture, 0.8)
        self.assertAlmostEqual(result.initial_minimum_occupied_capture, 0.9)
        self.assertAlmostEqual(result.occupied_capture_floor, 0.8 - 1.0e-8)
        self.assertLess(result.best_loss, result.initial_loss)

    def test_fixed_prefix_capture_is_evaluated_in_mother_space_metric(self):
        dataset = periodic_galerkin_fit.prepare_periodic_occupied_reference(
            self.three_level_dataset()
        )
        fixed = {
            "C": [
                torch.tensor(
                    [[1.0], [0.0], [0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        capture = periodic_galerkin_fit._minimum_occupied_capture(
            (dataset,),
            fixed,
        )

        self.assertAlmostEqual(capture, 1.0)

    def test_global_loss_uses_block_contraction(self):
        dataset = self.three_level_dataset()
        response = dataset.reference_response.clone()
        result = SimpleNamespace(
            response=response,
            minimum_occupied_capture=0.95,
            maximum_overlap_condition=3.0,
        )
        with mock.patch.object(
            periodic_galerkin_fit,
            "evaluate_periodic_galerkin_coefficient_response",
            return_value=result,
        ) as evaluator:
            loss, capture, condition, family_losses = periodic_galerkin_fit._global_pi_loss(
                (dataset,),
                {"C": [torch.eye(3, 2, dtype=torch.float64)]},
                occupied_capture_tolerance=0.2,
            )

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(capture, 0.95)
        self.assertEqual(condition, 3.0)
        self.assertEqual(float(family_losses["periodic"]), 0.0)
        self.assertEqual(evaluator.call_args.kwargs["contraction_backend"], "block")
        self.assertEqual(
            evaluator.call_args.kwargs["occupied_capture_tolerance"], 0.2
        )

    def test_global_loss_normalizes_atom_and_solid_families_separately(self):
        solid_q1 = self.three_level_dataset()
        solid_q2 = replace(solid_q1, q_weight=2.0)
        solid_q1 = replace(
            solid_q1,
            reference_response=torch.ones((1, 1, 1), dtype=torch.complex128),
        )
        solid_q2 = replace(
            solid_q2,
            reference_response=10.0
            * torch.ones((1, 1, 1), dtype=torch.complex128),
        )
        responses = iter(
            (
                torch.zeros((1, 1, 1), dtype=torch.complex128),
                9.0 * torch.ones((1, 1, 1), dtype=torch.complex128),
            )
        )

        def evaluate(dataset, _coefficients, **_kwargs):
            response = next(responses)
            return SimpleNamespace(
                response=response,
                minimum_occupied_capture=0.97,
                maximum_overlap_condition=4.0,
            )

        atom = SimpleNamespace(
            evaluate=lambda _coefficients: SimpleNamespace(
                loss=torch.tensor(0.25, dtype=torch.float64),
                max_candidate_condition=7.0,
            )
        )
        with mock.patch.object(
            periodic_galerkin_fit,
            "evaluate_periodic_galerkin_coefficient_response",
            side_effect=evaluate,
        ):
            loss, capture, condition, family_losses = (
                periodic_galerkin_fit._global_pi_loss(
                    (solid_q1, solid_q2),
                    {"C": [torch.eye(3, 2, dtype=torch.float64)]},
                    dataset_families=("C_solid", "C_solid"),
                    additional_family_evaluators={"C_atom": atom},
                )
            )

        expected_solid = (1.0 + 2.0 * 1.0) / (1.0 + 2.0 * 100.0)
        self.assertAlmostEqual(float(family_losses["C_solid"]), expected_solid)
        self.assertAlmostEqual(float(family_losses["C_atom"]), 0.25)
        self.assertAlmostEqual(float(loss), 0.5 * (expected_solid + 0.25))
        self.assertEqual(capture, 0.97)
        self.assertEqual(condition, 7.0)

    def test_global_loss_rejects_invalid_or_duplicate_family_names(self):
        dataset = self.three_level_dataset()
        coefficients = {"C": [torch.eye(3, 2, dtype=torch.float64)]}
        with self.assertRaisesRegex(ValueError, "dataset_families"):
            periodic_galerkin_fit._global_pi_loss(
                (dataset,),
                coefficients,
                dataset_families=("C_solid", "extra"),
            )
        with self.assertRaisesRegex(ValueError, "family name"):
            periodic_galerkin_fit._global_pi_loss(
                (dataset,),
                coefficients,
                dataset_families=("C_solid",),
                additional_family_evaluators={"C_solid": object()},
            )

    def test_real_coefficient_gradient_includes_complex_response_components(self):
        dataset = self.three_level_dataset()
        reference = torch.tensor([[[0.4 - 0.7j]]], dtype=torch.complex128)
        dataset = replace(dataset, reference_response=reference)

        def complex_response(_dataset, coefficients, **_kwargs):
            coordinate = coefficients["C"][0][0, 0].to(torch.complex128)
            response = ((1.2 + 0.8j) * coordinate).reshape(1, 1, 1)
            return SimpleNamespace(
                response=response,
                minimum_occupied_capture=1.0,
                maximum_overlap_condition=1.0,
            )

        coordinate = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
        with mock.patch.object(
            periodic_galerkin_fit,
            "evaluate_periodic_galerkin_coefficient_response",
            side_effect=complex_response,
        ):
            loss, _, _, _ = periodic_galerkin_fit._global_pi_loss(
                (dataset,),
                {"C": [coordinate]},
            )
            loss.backward()
            analytic = float(coordinate.grad)

            epsilon = 1.0e-6
            losses = []
            for displacement in (-epsilon, epsilon):
                trial = torch.tensor(
                    [[0.3 + displacement]], dtype=torch.float64
                )
                trial_loss, _, _, _ = periodic_galerkin_fit._global_pi_loss(
                    (dataset,),
                    {"C": [trial]},
                )
                losses.append(float(trial_loss))

        finite_difference = (losses[1] - losses[0]) / (2.0 * epsilon)
        self.assertAlmostEqual(analytic, finite_difference, places=8)

    def test_optimizer_prepares_constant_block_slices_once_per_kpoint(self):
        dataset = self.three_level_dataset()
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        serial = optimize_periodic_galerkin_basis(
            (dataset,),
            initial,
            fixed_nu={"C": (1,)},
            learning_rate=0.01,
            max_steps=1,
            minimum_steps=0,
            plateau_patience=1,
            plateau_relative_improvement=1.0e-8,
            block_cache_workers=1,
        )
        with mock.patch.object(
            periodic_galerkin_fit,
            "prepare_periodic_block_contraction_record",
            wraps=periodic_galerkin_basis.prepare_periodic_block_contraction_record,
        ) as prepare:
            parallel = optimize_periodic_galerkin_basis(
                (dataset,),
                initial,
                fixed_nu={"C": (1,)},
                learning_rate=0.01,
                max_steps=1,
                minimum_steps=0,
                plateau_patience=1,
                plateau_relative_improvement=1.0e-8,
            )

        self.assertEqual(prepare.call_count, len(dataset.kpoints))
        self.assertEqual(parallel.history, serial.history)
        self.assertTrue(
            torch.equal(
                parallel.coefficients["C"][0],
                serial.coefficients["C"][0],
            )
        )

    def test_optimizer_accepts_parallel_block_cache_preparation(self):
        dataset = self.three_level_dataset()
        dataset = replace(dataset, kpoints=dataset.kpoints * 2)
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        with mock.patch.object(
            periodic_galerkin_fit,
            "prepare_periodic_block_contraction_record",
            wraps=periodic_galerkin_basis.prepare_periodic_block_contraction_record,
        ) as prepare:
            optimize_periodic_galerkin_basis(
                (dataset,),
                initial,
                fixed_nu={"C": (1,)},
                learning_rate=0.01,
                max_steps=1,
                minimum_steps=0,
                plateau_patience=1,
                plateau_relative_improvement=1.0e-8,
                block_cache_workers=2,
            )

        self.assertEqual(prepare.call_count, len(dataset.kpoints))

    def test_optimizer_records_balanced_family_losses(self):
        dataset = self.three_level_dataset()
        initial = {
            "C": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        class AtomicEvaluator:
            def evaluate(self, coefficients):
                coordinate = coefficients["C"][0][2, 1]
                return SimpleNamespace(
                    loss=(coordinate - 0.5) ** 2 + 0.1,
                    max_candidate_condition=2.0,
                )

        result = optimize_periodic_galerkin_basis(
            (dataset,),
            initial,
            fixed_nu={"C": (1,)},
            dataset_families=("C_solid",),
            additional_family_evaluators={"C_atom": AtomicEvaluator()},
            learning_rate=0.02,
            max_steps=3,
            minimum_steps=0,
            plateau_patience=3,
            plateau_relative_improvement=1.0e-8,
        )

        self.assertEqual(set(result.initial_family_losses), {"C_atom", "C_solid"})
        self.assertEqual(set(result.best_family_losses), {"C_atom", "C_solid"})
        self.assertTrue(
            all(
                set(record["family_losses"]) == {"C_atom", "C_solid"}
                for record in result.history
            )
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
