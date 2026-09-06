import math
import inspect
import unittest
from dataclasses import fields, is_dataclass, replace
from unittest import mock

import torch

import common  # noqa: F401
import test_periodic_galerkin_fit as fixtures
import periodic_galerkin_fit as fit
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock
from periodic_galerkin_fit import optimize_periodic_galerkin_basis
from periodic_galerkin_optimization import (
    evaluate_periodic_galerkin_coefficient_response,
)
from periodic_galerkin_rpa import periodic_rpa_objective, prepare_periodic_rpa_reference
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


class PeriodicGalerkinRpaFitTest(unittest.TestCase):
    def assert_acceleration_api(self):
        options = inspect.signature(optimize_periodic_galerkin_basis).parameters
        self.assertIn("frequency_batch_size", options)
        self.assertIn("cache_rpa_reference", options)
        self.assertIsNone(options["frequency_batch_size"].default)
        self.assertIs(options["cache_rpa_reference"].default, False)

    def full_radial_fixture(self):
        dataset = fixtures.PeriodicGalerkinFitTest().three_level_dataset()
        blocks = []
        for l in range(3):
            for m in range(-l, l + 1):
                blocks.append(PeriodicGalerkinPrimitiveBlock("C", 0, l, m, 3, 3 * len(blocks)))
        count = 3 * len(blocks)
        frequencies = torch.tensor([0.08, 0.2, 0.5, 1.0, 2.0], dtype=torch.float64)
        energies = torch.linspace(0.7, 2.5, count, dtype=torch.float64)
        energies[0] = -0.5
        generator = torch.Generator().manual_seed(9347)
        source = 0.1 * torch.complex(
            torch.randn((1, 2, count), generator=generator, dtype=torch.float64),
            torch.randn((1, 2, count), generator=generator, dtype=torch.float64),
        )
        source[:, :, 0] = 0
        occupied = torch.zeros((1, count), dtype=torch.complex128)
        occupied[0, 0] = 1
        record = replace(
            dataset.kpoints[0],
            occupation=torch.tensor([2.0], dtype=torch.float64),
            source_eigenvalue_ha=torch.tensor([-0.5], dtype=torch.float64),
            overlap=torch.eye(count, dtype=torch.complex128),
            hamiltonian_ha=torch.diag(energies).to(torch.complex128),
            occupied_projection=occupied,
            source=source,
            reference_projection=torch.zeros((5, 1, 2, count), dtype=torch.complex128),
        )
        # Independent all-virtual-state sum for a diagonal finite Hamiltonian.
        transition = source[0, :, 1:]
        raw = []
        for omega in frequencies:
            half = -record.k_weight * record.occupation[0] * (
                (transition / (energies[1:] + 0.5 + 1j * omega)[None, :])
                @ transition.transpose(0, 1).conj()
            )
            raw.append(half + half.transpose(0, 1).conj())
        first = replace(
            dataset,
            q_count=2,
            selected_iq=1,
            q_weight=0.25,
            primitive_count=count,
            raw_auxiliary_dimension=2,
            whitened_auxiliary_rank=2,
            frequency_ha=frequencies,
            frequency_weights_ha=torch.tensor([0.1, 0.2, 0.3, 0.6, 1.0], dtype=torch.float64),
            coulomb_metric=torch.eye(2, dtype=torch.complex128),
            coulomb_whitening=torch.eye(2, dtype=torch.complex128),
            reference_response=torch.stack(raw),
            primitive_blocks=tuple(blocks),
            kpoints=(record,),
        )
        second = replace(
            first,
            selected_iq=2,
            qpoint=(0.25, 0.0, 0.0),
            q_weight=0.75,
            whitened_auxiliary_rank=1,
            coulomb_whitening=first.coulomb_whitening[:, :1],
            reference_response=first.reference_response[:, :1, :1] * 1.44,
            kpoints=(replace(
                record, source=source[:, :1, :] * 1.2,
                reference_projection=record.reference_projection[:, :, :1, :],
            ),),
        )
        initial = {"C": [
            torch.tensor([[1.0, 0.04], [0.0, 1.0], [0.07 + 0.03 * l, 0.2]],
                         dtype=torch.float64)
            for l in range(3)
        ]}
        return (first, second), initial

    def run_full_radial_fit(self, datasets, initial, **options):
        return optimize_periodic_galerkin_basis(
            datasets, initial, fixed_nu={"C": (0, 0, 0)}, objective="rpa",
            learning_rate=0.01, max_steps=4, minimum_steps=0, plateau_patience=5,
            occupied_capture_degradation_tolerance=0.1, **options
        )

    def assert_tree_close(self, actual, expected, exact=False):
        if is_dataclass(expected):
            self.assertIs(type(actual), type(expected))
            for field in fields(expected):
                self.assert_tree_close(getattr(actual, field.name), getattr(expected, field.name), exact)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree_close(actual[key], expected[key], exact)
        elif isinstance(expected, (list, tuple)):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_tree_close(left, right, exact)
        elif isinstance(expected, (torch.Tensor, float)):
            torch.testing.assert_close(actual, expected, rtol=0 if exact else 2e-10,
                                       atol=0 if exact else 2e-12)
        else:
            self.assertEqual(actual, expected)

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

    def test_acceleration_defaults_preserve_pi_and_rpa_fits_exactly(self):
        self.assert_acceleration_api()
        for objective in ("pi", "rpa"):
            with self.subTest(objective=objective), \
                    mock.patch.object(fit, "prepare_periodic_rpa_reference", side_effect=AssertionError), \
                    mock.patch.object(fit.time, "perf_counter", return_value=10.0):
                implicit = self.run_fit(objective=objective)
                explicit = self.run_fit(
                    objective=objective, frequency_batch_size=None, cache_rpa_reference=False
                )
            self.assert_tree_close(explicit, implicit, exact=True)

    def test_full_radial_spd_fit_improves_with_no_fixed_columns(self):
        self.assert_acceleration_api()
        datasets, initial = self.full_radial_fixture()
        snapshots = [channel.clone() for channel in initial["C"]]
        result = self.run_full_radial_fit(
            datasets, initial, frequency_batch_size=3, cache_rpa_reference=True
        )
        self.assertLess(result.best_loss, result.initial_loss)
        self.assertEqual(result.steps_completed, 4)
        self.assertEqual(result.occupied_capture_reference, "initial_candidate")
        self.assertEqual(result.history[0]["rpa"]["q_weight_coverage"], 1.0)
        for entry in result.history[1:]:
            self.assertTrue(math.isfinite(entry["previous_step_gradient_norm"]))
            self.assertGreater(entry["previous_step_gradient_norm"], 0)
        for saved, unchanged, fitted in zip(snapshots, initial["C"], result.coefficients["C"]):
            torch.testing.assert_close(unchanged, saved, rtol=0, atol=0)
            frame = torch.linalg.qr(saved, mode="reduced")[0]
            initial_projector = frame @ frame.transpose(0, 1)
            fitted_projector = fitted @ fitted.transpose(0, 1)
            self.assertGreater(float((fitted_projector - initial_projector).abs().max()), 1e-5)
            self.assertEqual(fitted.shape, (3, 2))

    def test_cached_batched_full_radial_fit_matches_scalar_values_coefficients_and_gradnorms(self):
        self.assert_acceleration_api()
        datasets, initial = self.full_radial_fixture()
        weights = {"pi_weight": 0.5, "trace_log_weight": 2.0, "energy_weight": 3.0}
        with mock.patch.object(fit.time, "perf_counter", return_value=10.0):
            expected = self.run_full_radial_fit(datasets, initial, rpa_weights=weights)
            for batch, cached in ((None, True), (3, False), (3, True), (8, True)):
                with self.subTest(batch=batch, cache=cached):
                    actual = self.run_full_radial_fit(
                        datasets, initial, frequency_batch_size=batch,
                        cache_rpa_reference=cached, rpa_weights=weights,
                    )
                    self.assert_tree_close(actual, expected, exact=batch is None)

    def test_global_rpa_loss_passes_batch_and_cache_and_preserves_all_column_gradients(self):
        self.assert_acceleration_api()
        datasets, initial = self.full_radial_fixture()
        datasets = tuple(prepare_periodic_occupied_reference(d) for d in datasets)
        cache = prepare_periodic_rpa_reference(datasets)
        weights = {"pi_weight": 0.5, "trace_log_weight": 2.0, "energy_weight": 0.0}
        gradients = []
        results = []
        with mock.patch.object(fit.time, "perf_counter", return_value=10.0):
            for options in ({}, {"frequency_batch_size": 3, "reference_cache": cache}):
                coefficients = {"C": [c.clone().requires_grad_() for c in initial["C"]]}
                result = fit._global_rpa_loss(
                    datasets, coefficients, occupied_capture_tolerance=0.5,
                    weights=weights, **options
                )
                result[0].backward()
                gradients.append([c.grad for c in coefficients["C"]])
                results.append(result)
        self.assert_tree_close(results[1], results[0])
        self.assert_tree_close(gradients[1], gradients[0])
        for gradient in gradients[1]:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertTrue(bool((gradient.abs().sum(dim=0) > 1e-9).all()))

    def test_reference_cache_is_prepared_once_after_all_dataset_replacements(self):
        self.assert_acceleration_api()
        datasets, initial = self.full_radial_fixture()
        prepared_calls = []

        def prepare(final_datasets):
            for original, final in zip(datasets, final_datasets):
                self.assertIsNot(final, original)
                for record in final.kpoints:
                    self.assertIsNotNone(record.occupied_projection_normalization)
                    self.assertIsNotNone(record.block_contraction_cache)
            cache = prepare_periodic_rpa_reference(final_datasets)
            prepared_calls.append((final_datasets, cache))
            return cache

        with mock.patch.object(fit, "prepare_periodic_rpa_reference", side_effect=prepare) as preparation, \
                mock.patch.object(fit, "periodic_rpa_objective", wraps=periodic_rpa_objective) as objective, \
                mock.patch.object(fit, "evaluate_periodic_galerkin_coefficient_response",
                                  wraps=evaluate_periodic_galerkin_coefficient_response) as evaluate:
            result = self.run_full_radial_fit(
                datasets, initial, frequency_batch_size=3, cache_rpa_reference=True
            )
        preparation.assert_called_once()
        self.assertGreaterEqual(objective.call_count, result.steps_completed + 1)
        final_datasets, cache = prepared_calls[0]
        for call in objective.call_args_list:
            self.assertIs(call[0][0], final_datasets)
            self.assertIs(call[1]["reference_cache"], cache)
        for call in evaluate.call_args_list:
            self.assertEqual(call[1]["frequency_batch_size"], 3)

    def test_pi_objective_rejects_nondefault_acceleration_options(self):
        self.assert_acceleration_api()
        for options in ({"frequency_batch_size": 1}, {"cache_rpa_reference": True},
                        {"frequency_batch_size": 3, "cache_rpa_reference": True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "objective=rpa"):
                self.run_fit(objective="pi", **options)

    def test_invalid_acceleration_options_fail_before_dataset_preparation(self):
        self.assert_acceleration_api()
        options = [
            {"frequency_batch_size": value} for value in (0, -1, True, 1.5, "2")
        ] + [
            {"cache_rpa_reference": value} for value in (None, 0, 1, "true")
        ]
        with mock.patch.object(fit, "prepare_periodic_occupied_reference", side_effect=AssertionError):
            for option in options:
                with self.subTest(option=option), self.assertRaisesRegex(ValueError, next(iter(option))):
                    self.run_fit(objective="rpa", **option)

    def test_coefficient_guard_exception_api(self):
        error_class = getattr(fit, "CandidateGuardError", None)
        self.assertIsNotNone(error_class)
        self.assertTrue(issubclass(error_class, RuntimeError))

    def assert_guard_api(self):
        options = inspect.signature(optimize_periodic_galerkin_basis).parameters
        self.assertIn("coefficient_guard", options)
        self.assertIsNone(options["coefficient_guard"].default)

    def test_coefficient_guard_default_preserves_existing_history(self):
        self.assert_guard_api()
        with mock.patch.object(fit.time, "perf_counter", return_value=10.0):
            for objective in ("pi", "rpa"):
                with self.subTest(objective=objective):
                    implicit = self.run_fit(objective=objective)
                    explicit = self.run_fit(objective=objective, coefficient_guard=None)
                    self.assert_tree_close(explicit, implicit, exact=True)
                    self.assertTrue(all("coefficient_guard" not in x for x in explicit.history))

    def test_coefficient_guard_runs_no_grad_before_each_loss_and_snapshots_diagnostics(self):
        self.assert_guard_api()
        for objective, loss_name in (("pi", "_global_pi_loss"), ("rpa", "_global_rpa_loss")):
            events = []
            diagnostic = {"gate": True, "nested": {"call": 0}}
            original_loss = getattr(fit, loss_name)

            def guard(coefficients):
                self.assertFalse(torch.is_grad_enabled())
                diagnostic["nested"]["call"] += 1
                events.append(("guard", id(coefficients)))
                return diagnostic

            def evaluate(datasets, coefficients, **options):
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(events[-1], ("guard", id(coefficients)))
                events.append(("loss", id(coefficients)))
                return original_loss(datasets, coefficients, **options)

            with self.subTest(objective=objective), \
                    mock.patch.object(fit, loss_name, side_effect=evaluate) as loss:
                result = self.run_fit(objective=objective, coefficient_guard=guard)
            self.assertEqual(loss.call_count, len(result.history))
            self.assertEqual(len(events), 2 * len(result.history))
            for index, entry in enumerate(result.history):
                self.assertEqual(entry["coefficient_guard"], {"gate": True, "nested": {"call": index + 1}})
            diagnostic["nested"]["call"] = -1
            self.assertEqual(result.history[0]["coefficient_guard"]["nested"]["call"], 1)

    def test_coefficient_guard_initial_rejection_skips_loss_and_propagates(self):
        self.assert_guard_api()
        for raises in (False, True):
            def guard(coefficients):
                if raises:
                    raise fit.CandidateGuardError("initial coefficient rejection")
                return {"gate": False, "reason": "initial coefficient rejection"}

            with self.subTest(raises=raises), \
                    mock.patch.object(fit, "_global_rpa_loss", side_effect=AssertionError) as loss, \
                    self.assertRaises(fit.CandidateGuardError):
                self.run_fit(objective="rpa", coefficient_guard=guard)
            loss.assert_not_called()

    def test_coefficient_guard_rejections_backtrack_before_response_evaluation(self):
        self.assert_guard_api()
        accepted, rejected = [], []

        def guard(coefficients):
            coordinate = float(coefficients["C"][0][2, 1])
            if coordinate > 0.006:
                rejected.append(coordinate)
                raise fit.CandidateGuardError("coordinate exceeds candidate bound")
            accepted.append(coordinate)
            return {"gate": True, "coordinate": coordinate}

        with mock.patch.object(fit, "_global_rpa_loss", wraps=fit._global_rpa_loss) as loss, \
                mock.patch.object(fit, "prepare_periodic_rpa_reference",
                                  wraps=prepare_periodic_rpa_reference) as prepare:
            result = self.run_fit(
                objective="rpa", coefficient_guard=guard, maximum_backtracks=8,
                frequency_batch_size=1, cache_rpa_reference=True,
            )
        prepare.assert_called_once()
        self.assertGreater(len(rejected), 0)
        self.assertEqual(loss.call_count, len(accepted))
        self.assertGreater(result.total_backtracks, 0)
        self.assertGreater(result.steps_completed, 0)
        self.assertLess(result.best_loss, result.initial_loss)
        self.assertLessEqual(float(result.coefficients["C"][0][2, 1]), 0.006)
        self.assertTrue(all(x["coefficient_guard"]["gate"] for x in result.history))

    def test_coefficient_guard_trial_exhaustion_has_distinct_boundary_reason(self):
        self.assert_guard_api()
        calls = []

        def guard(coefficients):
            coordinate = float(coefficients["C"][0][2, 1])
            calls.append(coordinate)
            return {"gate": coordinate == 0.0, "coordinate": coordinate}

        with mock.patch.object(fit, "_global_rpa_loss", wraps=fit._global_rpa_loss) as loss:
            result = self.run_fit(
                objective="rpa", coefficient_guard=guard, maximum_backtracks=2
            )
        self.assertEqual(result.stop_reason, "candidate_guard_boundary")
        self.assertEqual(result.steps_completed, 0)
        self.assertEqual(result.best_step, 0)
        self.assertEqual(result.best_loss, result.initial_loss)
        self.assertEqual(len(calls), 4)
        self.assertEqual(loss.call_count, 1)
        self.assertEqual(len(result.history), 1)
        _, initial = self.fixture()
        torch.testing.assert_close(result.coefficients["C"][0], initial["C"][0], rtol=0, atol=0)

    def test_coefficient_guard_does_not_swallow_unrelated_runtime_errors(self):
        self.assert_guard_api()
        calls = []

        def guard(coefficients):
            calls.append(None)
            if len(calls) > 1:
                raise RuntimeError("unexpected callback error")
            return {"gate": True}

        with self.assertRaisesRegex(RuntimeError, "unexpected callback error"):
            self.run_fit(objective="rpa", coefficient_guard=guard)
        self.assertEqual(len(calls), 2)

        original_loss = fit._global_rpa_loss
        calls.clear()

        def failing_loss(*args, **kwargs):
            calls.append(None)
            if len(calls) > 1:
                raise RuntimeError("unexpected response error")
            return original_loss(*args, **kwargs)

        with mock.patch.object(fit, "_global_rpa_loss", side_effect=failing_loss), \
                self.assertRaisesRegex(RuntimeError, "unexpected response error"):
            self.run_fit(objective="rpa", coefficient_guard=lambda c: {"gate": True})
        self.assertEqual(len(calls), 2)

    def test_coefficient_guard_rejects_noncallable_before_preparation(self):
        self.assert_guard_api()
        with mock.patch.object(fit, "prepare_periodic_occupied_reference", side_effect=AssertionError):
            for invalid in (False, 1, "guard", {}):
                with self.subTest(guard=invalid), self.assertRaisesRegex(ValueError, "coefficient_guard"):
                    self.run_fit(objective="rpa", coefficient_guard=invalid)

    def test_coefficient_guard_requires_serializable_diagnostics_and_boolean_gate(self):
        self.assert_guard_api()
        for invalid in (None, True, {}, {"gate": 1}, {"gate": True, "value": object()},
                        {"gate": True, "value": torch.tensor(0.1)},
                        {"gate": True, "value": float("nan")}):
            with self.subTest(diagnostics=invalid), \
                    mock.patch.object(fit, "_global_rpa_loss", side_effect=AssertionError), \
                    self.assertRaisesRegex(ValueError, "coefficient_guard"):
                self.run_fit(objective="rpa", coefficient_guard=lambda c: invalid)


if __name__ == "__main__":
    unittest.main()
