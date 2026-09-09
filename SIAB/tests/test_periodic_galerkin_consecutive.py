"""Pure-NumPy controller tests; no SCF, torch, or native builds are needed."""

import copy
import importlib
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np


OPT_DIR = Path(__file__).resolve().parents[1] / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))


class EuclideanProblem:
    def __init__(self, size=12):
        self.initial = np.zeros(size, dtype=np.float64)
        self.target = np.linspace(0., 1., size)
        self.b = np.eye(1, size)[0]
        self.gradients = []
        self.measured = []
        self.evaluated = []
        self.checkpoints = []

    def pbe(self, x):
        return float(x[0])

    def record(self, x):
        objective = float(.5 * np.dot(x - self.target, x - self.target))
        return dict(objective=objective, loss=objective, pbe=self.pbe(x))

    def gradient(self, x):
        self.gradients.append(x.copy())
        return x - self.target

    def measure(self, x, trial_id):
        self.measured.append((trial_id, x.copy()))
        return dict(gate="pass", pbe=self.pbe(x))

    def evaluate(self, x, trial_id):
        self.evaluated.append((trial_id, x.copy()))
        return self.record(x)

    def run(self, run_consecutive, **options):
        callbacks = dict(gradient=self.gradient, retract=lambda x, s: x + s,
                         project=lambda x, v: v.copy(), measure=self.measure,
                         evaluate=self.evaluate, checkpoint=self.checkpoints.append)
        callbacks.update(options)
        return run_consecutive(self.initial, self.record(self.initial), self.b, **callbacks)


class ConsecutiveTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("periodic_galerkin_consecutive"),
                             "the reusable consecutive optimizer module is missing")
        self.run_consecutive = importlib.import_module(
            "periodic_galerkin_consecutive").run_consecutive

    def test_successive_acceptances_refresh_full_248_gradient_and_grow_radius(self):
        problem = EuclideanProblem(248)
        result = problem.run(self.run_consecutive, max_steps=4)
        self.assertEqual(result["status"], "max_steps")
        self.assertEqual(result["counts"]["accepted_steps"], 4)
        self.assertEqual(len(problem.checkpoints), 4)
        self.assertGreaterEqual(len(problem.gradients), 4)
        np.testing.assert_array_equal(problem.gradients[0], problem.initial)
        for index in range(1, 4):
            np.testing.assert_array_equal(problem.gradients[index],
                                          problem.checkpoints[index - 1]["x"])
            self.assertFalse(np.array_equal(problem.gradients[index - 1],
                                            problem.gradients[index]))
        self.assertEqual(np.count_nonzero(result["x"][1:]), 247)
        self.assertEqual([r["trial_id"] for r in result["history"]], [0, 1, 2, 3])
        self.assertEqual(result["accepted_steps"], 4)
        self.assertEqual(result["trials"], 4)
        np.testing.assert_allclose([r["radius"] for r in result["history"]],
                                   [.02, .03, .04, .04])
        self.assertTrue(all(r["accepted"] for r in result["history"]))
        self.assertTrue(all(r["objective_gain"] > 0 for r in result["history"]))

    def test_small_positive_gains_do_not_stop_gradient_refresh(self):
        problem = EuclideanProblem()
        result = problem.run(self.run_consecutive, max_steps=3, initial_radius=1e-5,
                             min_radius=1e-7, gain_tolerance=.1)
        self.assertEqual(result["counts"]["accepted_steps"], 3)
        self.assertTrue(all(row["small_gain"] for row in result["history"]))
        self.assertGreaterEqual(len(problem.gradients), 3)

    def test_actual_positive_and_negative_bound_failures_never_evaluate(self):
        for shift in (.011, -.011):
            with self.subTest(shift=shift):
                problem = EuclideanProblem()
                result = problem.run(self.run_consecutive, max_trials=2,
                    measure=lambda x, i: dict(gate="pass", pbe=shift))
                self.assertEqual(result["status"], "max_trials")
                self.assertEqual(problem.evaluated, [])
                self.assertEqual(problem.checkpoints, [])
                self.assertTrue(all(r["reason"] == "pbe_limit_exceeded"
                                    for r in result["history"]))
                np.testing.assert_array_equal(result["x"], problem.initial)
                np.testing.assert_allclose([r["radius"] for r in result["history"]],
                                           [.02, .01])

    def test_failed_missing_or_nonfinite_actual_pbe_never_evaluates(self):
        measurements = [dict(gate="fail", pbe=0.), dict(gate="cheap_reject"),
                        dict(gate="pass"), dict(gate="pass", pbe=float("nan")),
                        dict(gate="pass", pbe=float("inf")), dict(pbe=0.)]
        for measurement in measurements:
            with self.subTest(measurement=measurement):
                problem = EuclideanProblem()
                result = problem.run(self.run_consecutive, max_trials=1,
                                     measure=lambda x, i: measurement)
                self.assertEqual(result["counts"]["accepted_steps"], 0)
                self.assertEqual(problem.evaluated, [])
                self.assertEqual(problem.checkpoints, [])

    def test_initial_point_must_have_finite_feasible_record(self):
        problem = EuclideanProblem()
        for change in (dict(pbe=.0101), dict(pbe=-.0101), dict(pbe=np.nan),
                       dict(objective=np.inf), dict(loss=-1.), dict(gate="fail")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_consecutive(problem.initial,
                    dict(problem.record(problem.initial), **change), problem.b,
                    gradient=problem.gradient, project=lambda x, v: v,
                    retract=lambda x, s: x + s, measure=problem.measure,
                    evaluate=problem.evaluate, checkpoint=problem.checkpoints.append)
        self.assertEqual(problem.gradients, [])
        self.assertEqual(problem.measured, [])

    def test_evaluation_requires_strict_improvement_and_nonincreasing_loss(self):
        problem = EuclideanProblem()
        center = problem.record(problem.initial)
        records = [dict(objective=center["objective"], loss=0.),
                   dict(objective=center["objective"] + 1., loss=0.),
                   dict(objective=0., loss=center["loss"] + .01),
                   dict(objective=np.nan, loss=0.), dict(objective=np.inf, loss=0.),
                   dict(objective=0., loss=np.inf), dict(objective=0., loss=-1.),
                   dict(loss=0.), dict(objective=0., loss=0., gate="fail"),
                   dict(objective=0., loss=0., pbe=np.nan)]
        for record in records:
            with self.subTest(record=record):
                result = problem.run(self.run_consecutive, max_trials=1,
                                     evaluate=lambda x, i: record)
                self.assertEqual(result["counts"]["accepted_steps"], 0)
                self.assertEqual(result["record"], center)
                np.testing.assert_array_equal(result["best"]["x"], problem.initial)
        self.assertEqual(problem.checkpoints, [])

    def test_evaluate_may_omit_pbe_and_measurement_remains_authoritative(self):
        problem = EuclideanProblem()
        result = problem.run(self.run_consecutive, max_steps=1,
            measure=lambda x, i: dict(gate="pass", pbe=.003),
            evaluate=lambda x, i: dict(objective=0., loss=0.))
        self.assertEqual(result["record"]["pbe"], .003)
        self.assertEqual(result["counts"]["accepted_steps"], 1)

    def test_failed_actual_secant_changes_next_proposal_in_ambient_coordinates(self):
        problem = EuclideanProblem(2)
        points = []

        def measure(x, trial_id):
            points.append(x.copy())
            return dict(gate="fail", pbe=.02) if trial_id == 0 else dict(gate="pass", pbe=0.)

        result = problem.run(self.run_consecutive, max_trials=2, measure=measure)
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0][0], 0.)
        self.assertLess(points[1][0], 0.)
        self.assertGreater(points[1][1], 0.)
        first = result["history"][0]
        self.assertAlmostEqual(first["predicted_pbe"], 0.)
        self.assertAlmostEqual(first["pbe_prediction_error"], .02)
        self.assertTrue(first["secant_updated"])
        self.assertEqual(result["counts"]["secant_updates"], 2)
        np.testing.assert_allclose(first["pbe_gradient_after"], [1., 1.])
        np.testing.assert_allclose(result["pbe_gradient"] @ points[1], 0., atol=1e-16)
        self.assertEqual(len(problem.gradients), 1)

    def test_secant_uses_retracted_displacement_not_proposed_step(self):
        problem = EuclideanProblem(3)
        problem.initial = np.array([1., 0., 0.])
        problem.target = np.array([1., 1., 0.])
        problem.b = np.array([3., 0., 1.])
        problem.pbe = lambda x: float(.1 * x[1])
        result = problem.run(self.run_consecutive, max_trials=1,
            project=lambda x, v: v - x * (x @ v),
            retract=lambda x, s: (x + s) / np.linalg.norm(x + s))
        trial = problem.measured[0][1]
        dx = trial - problem.initial
        error = problem.pbe(trial) - problem.pbe(problem.initial) - problem.b @ dx
        expected = problem.b + error * dx / (dx @ dx)
        np.testing.assert_allclose(result["pbe_gradient"], expected, atol=1e-13)

    def test_inward_normal_is_signed_and_capped_for_both_sides(self):
        for sign in (1., -1.):
            with self.subTest(sign=sign):
                problem = EuclideanProblem(2)
                problem.pbe = lambda x: float(sign * .0095 + .01 * x[0])
                problem.b = np.array([.01, 0.])
                result = problem.run(self.run_consecutive, max_steps=1)
                step = result["x"] - problem.initial
                self.assertLess(sign * step[0], 0.)
                self.assertLessEqual(abs(step[0]), .3 * .02 + 1e-15)
                self.assertLessEqual(np.linalg.norm(step), .02 + 1e-15)
                self.assertLess(abs(result["record"]["pbe"]), .0095)

    def test_normal_correction_is_reduced_if_it_would_destroy_descent(self):
        problem = EuclideanProblem(2)
        problem.target = np.array([100., 1.])
        problem.pbe = lambda x: float(.0095 + .01 * x[0])
        problem.b = np.array([.01, 0.])
        result = problem.run(self.run_consecutive, max_steps=1, max_trials=2)
        self.assertEqual(result["counts"]["accepted_steps"], 1)
        step = result["x"] - problem.initial
        self.assertLess(step[0], 0.)
        self.assertLess(abs(step[0]), .3 * .02)
        self.assertLess((-problem.target) @ step, 0.)

    def test_nonlinear_feasible_constraint_makes_repeated_progress(self):
        problem = EuclideanProblem(2)
        problem.target = np.array([0., .5])
        problem.pbe = lambda x: float(.007 + .4 * x[0] + 1.5 * x[1] ** 2)
        problem.b = np.array([.4, 0.])
        result = problem.run(self.run_consecutive, max_steps=8, max_trials=60,
                             initial_radius=.04, max_radius=.08)
        self.assertGreaterEqual(result["counts"]["accepted_steps"], 3)
        self.assertLess(result["record"]["objective"],
                        .8 * problem.record(problem.initial)["objective"])
        self.assertTrue(all(abs(state["record"]["pbe"]) <= .010
                            for state in problem.checkpoints))
        self.assertTrue(any(abs(row["pbe_prediction_error"] or 0.) > 1e-5
                            for row in result["history"]))
        for row in result["history"]:
            if row["accepted"]:
                self.assertLess(row["predicted_objective_delta"], 0.)
                self.assertGreater(row["objective_gain"], 0.)
                self.assertIn("gradient_norm", row)
                self.assertIn("projected_gradient_norm", row)
                self.assertIn("feasible_gradient_norm", row)

    def test_sphere_projection_is_refreshed_and_retractions_preserve_shape(self):
        problem = EuclideanProblem(4)
        problem.initial = np.array([1., 0., 0., 0.])
        problem.target = np.array([0., 1., .4, 0.])
        problem.b = np.array([0., 0., 0., 1.])
        problem.pbe = lambda x: float(x[3])
        projected_at = []

        def project(x, vector):
            self.assertEqual(x.shape, (4,))
            self.assertEqual(x.dtype, np.dtype("float64"))
            self.assertEqual(vector.shape, x.shape)
            projected_at.append(x.copy())
            return vector - x * np.dot(x, vector)

        def retract(x, step):
            self.assertAlmostEqual(float(x @ step), 0., places=13)
            return (x + step) / np.linalg.norm(x + step)

        result = problem.run(self.run_consecutive, max_steps=5, project=project, retract=retract)
        self.assertEqual(result["counts"]["accepted_steps"], 5)
        for state in problem.checkpoints:
            self.assertEqual(state["x"].shape, (4,))
            self.assertAlmostEqual(np.linalg.norm(state["x"]), 1., places=14)
        for center in problem.gradients[:5]:
            self.assertGreaterEqual(sum(np.array_equal(center, x) for x in projected_at), 2)

    def test_budget_and_direction_stops_never_claim_stationarity(self):
        cases = [(dict(max_steps=0), "max_steps"), (dict(max_trials=0), "max_trials"),
                 (dict(gradient=lambda x: np.zeros_like(x)), "feasible_direction_unresolved"),
                 (dict(project=lambda x, v: np.zeros_like(v)), "feasible_direction_unresolved")]
        for options, status in cases:
            with self.subTest(options=options):
                problem = EuclideanProblem()
                result = problem.run(self.run_consecutive, **options)
                self.assertEqual(result["status"], status)
                self.assertNotIn("converged", result)
                self.assertNotIn("stationary", result)
                self.assertEqual(problem.measured, [])
        problem = EuclideanProblem()
        result = problem.run(self.run_consecutive, min_radius=.015,
                             measure=lambda x, i: dict(gate="cheap_reject"))
        self.assertEqual(result["status"], "feasible_direction_unresolved")
        self.assertEqual(result["stop_reason"], "min_radius")
        self.assertEqual(result["counts"]["trials"], 1)

    def test_retraction_that_reverses_descent_is_not_measured(self):
        problem = EuclideanProblem()
        result = problem.run(self.run_consecutive, max_trials=2, retract=lambda x, s: x - s)
        self.assertEqual(problem.measured, [])
        self.assertEqual(result["status"], "max_trials")
        self.assertTrue(all(row["reason"] == "non_descent_retraction"
                            for row in result["history"]))

    def test_invalid_gradient_projection_and_retraction_raise(self):
        for callback in ("gradient", "project", "retract"):
            for bad in (np.zeros((2, 6)), np.zeros(11), np.full(12, np.nan),
                        np.full(12, np.inf), np.ones(12, dtype=complex) * (1 + 1j)):
                with self.subTest(callback=callback, bad=bad), self.assertRaises(ValueError):
                    EuclideanProblem().run(self.run_consecutive,
                                           **{callback: lambda *args: bad})

    def test_nonfinite_initial_vector_and_surrogate_raise(self):
        for field in ("initial", "b"):
            for bad in (np.full(12, np.nan), np.zeros((3, 4)), np.zeros(0)):
                problem = EuclideanProblem()
                setattr(problem, field, bad)
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    self.run_consecutive(problem.initial, dict(objective=1., loss=1., pbe=0.),
                        problem.b, gradient=problem.gradient, project=lambda x, v: v,
                        retract=lambda x, s: x + s, measure=problem.measure,
                        evaluate=problem.evaluate, checkpoint=problem.checkpoints.append)

    def test_invalid_options_raise_before_callbacks(self):
        for options in (dict(max_steps=-1), dict(max_trials=1.5), dict(max_steps=True),
                        dict(initial_radius=np.nan), dict(initial_radius=0.),
                        dict(max_radius=.001), dict(min_radius=.03),
                        dict(pbe_limit=-1.), dict(pbe_target=.011), dict(pbe_target=-1.),
                        dict(gradient_tolerance=np.inf), dict(gain_tolerance=-1.)):
            problem = EuclideanProblem()
            with self.subTest(options=options), self.assertRaises(ValueError):
                problem.run(self.run_consecutive, **options)
            self.assertEqual(problem.gradients, [])

    def test_rejected_point_cannot_overwrite_checkpoint_or_best(self):
        problem = EuclideanProblem()
        records = []

        def evaluate(x, trial_id):
            record = problem.record(x)
            record["metadata"] = {"ids": [trial_id]}
            if trial_id > 0:
                record["objective"] += 10.
            records.append(record)
            return record

        result = problem.run(self.run_consecutive, max_trials=3, evaluate=evaluate)
        self.assertEqual(len(problem.checkpoints), 1)
        saved = problem.checkpoints[0]
        np.testing.assert_array_equal(result["x"], saved["x"])
        np.testing.assert_array_equal(result["best"]["x"], saved["x"])
        self.assertEqual(saved["counts"]["trials"], 1)
        self.assertEqual(len(saved["history"]), 1)
        self.assertEqual(result["counts"]["trials"], 3)
        records[0]["metadata"]["ids"].append(9)
        self.assertEqual(saved["record"]["metadata"]["ids"], [0])
        saved_before = copy.deepcopy(saved)
        result["x"][:] = 50.
        result["record"]["metadata"]["ids"].append(8)
        np.testing.assert_array_equal(result["best"]["x"], saved_before["x"])
        np.testing.assert_array_equal(saved["x"], saved_before["x"])
        self.assertEqual(result["best"]["record"]["metadata"]["ids"], [0])

    def test_mutating_checkpoint_and_callback_arguments_cannot_change_center(self):
        problem = EuclideanProblem()

        def checkpoint(state):
            state["x"][:] = 999.
            state["record"]["objective"] = -1.
            state["best"]["x"][:] = 888.
            state["counts"]["trials"] = 999
            state["history"].clear()
            state["pbe_gradient"][:] = 999.

        def gradient(x):
            value = problem.gradient(x)
            x[:] = 99.
            return value

        result = problem.run(self.run_consecutive, max_steps=3, checkpoint=checkpoint,
                             gradient=gradient)
        self.assertEqual(result["counts"]["accepted_steps"], 3)
        self.assertEqual(result["counts"]["trials"], 3)
        self.assertEqual(len(result["history"]), 3)
        self.assertTrue(np.all(result["x"] < 1.))
        np.testing.assert_array_equal(problem.initial, np.zeros(12))


class JointProblem(EuclideanProblem):
    def __init__(self, loss_weight=.003):
        super().__init__(3)
        self.target = np.array([0., 1., 0.])
        self.loss_target = np.array([0., -1., 2.])
        self.loss_weight = loss_weight
        self.loss_gradients = []

    def record(self, x):
        result = super().record(x)
        residual = x - self.loss_target
        result["loss"] = float(.5 * self.loss_weight * (residual @ residual))
        return result

    def loss_gradient(self, x):
        self.loss_gradients.append(x.copy())
        return self.loss_weight * (x - self.loss_target)


class JointConsecutiveTest(unittest.TestCase):
    def setUp(self):
        self.run_consecutive = importlib.import_module(
            "periodic_galerkin_consecutive").run_consecutive
        self.assertIn("loss_gradient", inspect.signature(self.run_consecutive).parameters,
                      "the optional exact loss-gradient callback is missing")

    def test_energy_only_stalls_but_common_descent_accepts_multiple_points(self):
        energy_only = JointProblem()
        baseline = energy_only.run(self.run_consecutive, max_trials=5)
        self.assertEqual(baseline["accepted_steps"], 0)
        self.assertTrue(all(row["reason"] == "loss_increased" for row in baseline["history"]))
        self.assertTrue(all(row["objective_gain"] > 0. for row in baseline["history"]))
        problem = JointProblem()
        result = problem.run(self.run_consecutive, max_steps=5,
                             loss_gradient=problem.loss_gradient)
        self.assertEqual(result["accepted_steps"], 5)
        previous = problem.record(problem.initial)
        for state in problem.checkpoints:
            self.assertLess(state["record"]["objective"], previous["objective"])
            self.assertLess(state["record"]["loss"], previous["loss"])
            self.assertLessEqual(abs(state["record"]["pbe"]), .010)
            previous = state["record"]
        for row in result["history"]:
            self.assertLess(row["predicted_step_loss_delta"], 0.)
            self.assertLess(row["predicted_loss_delta"], 0.)
            self.assertGreater(row["loss_gradient_norm"], 0.)

    def test_analytic_mixture_uses_normalized_gradients_and_is_scale_independent(self):
        expected_u = np.array([0., -1., 0.])
        expected_v = np.array([0., 1., -2.]) / np.sqrt(5.)
        difference = expected_u - expected_v
        weight = np.clip(-(expected_v @ difference) / (difference @ difference), 0., 1.)
        combination = weight * expected_u + (1. - weight) * expected_v
        for loss_weight in (1e-7, .003, 1e4):
            with self.subTest(loss_weight=loss_weight):
                problem = JointProblem(loss_weight)
                result = problem.run(self.run_consecutive, max_steps=1,
                                     loss_gradient=problem.loss_gradient)
                np.testing.assert_allclose(result["x"], -.02 * combination / np.linalg.norm(combination))
                row = result["history"][0]
                self.assertAlmostEqual(row["joint_mixing_weight"], weight)
                self.assertAlmostEqual(row["joint_combination_norm"], np.linalg.norm(combination))
                self.assertAlmostEqual(row["joint_angle_degrees"],
                                       np.degrees(np.arccos(expected_u @ expected_v)))
                self.assertAlmostEqual(row["feasible_loss_gradient_norm"], loss_weight * np.sqrt(5.))

    def test_parallel_gradients_do_not_divide_by_zero(self):
        problem = JointProblem()
        problem.loss_target = problem.target.copy()
        result = problem.run(self.run_consecutive, max_steps=3, loss_gradient=problem.loss_gradient)
        self.assertEqual(result["accepted_steps"], 3)
        self.assertAlmostEqual(result["history"][0]["joint_angle_degrees"], 0.)
        self.assertAlmostEqual(result["history"][0]["joint_combination_norm"], 1.)

    def test_antiparallel_or_tiny_combination_is_unresolved_not_convergence(self):
        for offset in (0., 1e-12):
            with self.subTest(offset=offset):
                problem = JointProblem()
                problem.loss_target = np.array([0., -1., offset])
                result = problem.run(self.run_consecutive, loss_gradient=problem.loss_gradient)
                self.assertEqual(result["status"], "unresolved_joint_direction")
                self.assertEqual(result["stop_reason"], "unresolved_joint_direction")
                self.assertEqual(result["trials"], 0)
                self.assertEqual(problem.measured, [])
                self.assertEqual(problem.checkpoints, [])
                np.testing.assert_array_equal(result["best"]["x"], problem.initial)
                self.assertNotIn("converged", result)
                self.assertNotIn("stationary", result)

    def test_both_gradients_refresh_in_order_only_at_new_centers(self):
        problem = JointProblem()
        cached = {}
        events = []

        def gradient(x):
            events.append("energy")
            cached["x"] = x.copy()
            cached["loss_gradient"] = problem.loss_weight * (x - problem.loss_target)
            return problem.gradient(x)

        def loss_gradient(x):
            events.append("loss")
            np.testing.assert_array_equal(x, cached["x"])
            problem.loss_gradients.append(x.copy())
            x[:] = 999.
            return cached["loss_gradient"]

        def measure(x, trial_id):
            events.append("measure")
            return dict(gate="cheap_reject") if trial_id % 2 == 0 else problem.measure(x, trial_id)

        result = problem.run(self.run_consecutive, max_steps=3, gradient=gradient,
                             loss_gradient=loss_gradient, measure=measure)
        self.assertEqual(result["accepted_steps"], 3)
        self.assertEqual(result["trials"], 6)
        self.assertEqual(events, ["energy", "loss", "measure", "measure"] * 3)
        self.assertEqual(result["counts"]["gradient_calls"], 3)
        self.assertEqual(result["counts"]["loss_gradient_calls"], 3)
        for index, (energy_x, loss_x) in enumerate(zip(problem.gradients, problem.loss_gradients)):
            center = problem.initial if index == 0 else problem.checkpoints[index - 1]["x"]
            np.testing.assert_array_equal(energy_x, center)
            np.testing.assert_array_equal(loss_x, center)
        np.testing.assert_array_equal(cached["loss_gradient"],
            problem.loss_weight * (cached["x"] - problem.loss_target))

    def test_unresolved_combination_retains_diagnostic_without_new_trial(self):
        for offset in (0., 1e-12):
            with self.subTest(offset=offset):
                problem = JointProblem()
                problem.loss_target = np.array([0., -1., offset])
                result = problem.run(self.run_consecutive, loss_gradient=problem.loss_gradient)
                self.assertIn("direction_diagnostic", result)
                diagnostic = result["direction_diagnostic"]
                self.assertEqual(diagnostic["reason"], "joint_combination_tiny")
                self.assertAlmostEqual(diagnostic["joint_angle_degrees"], 180.)
                self.assertAlmostEqual(diagnostic["joint_mixing_weight"], .5)
                self.assertLessEqual(diagnostic["joint_combination_norm"],
                                     diagnostic["gradient_tolerance"])
                for key in ("gradient_norm", "projected_gradient_norm", "feasible_gradient_norm",
                            "pbe_gradient_norm", "loss_gradient_norm",
                            "projected_loss_gradient_norm", "feasible_loss_gradient_norm"):
                    self.assertTrue(np.isfinite(diagnostic[key]))
                    self.assertGreater(diagnostic[key], 0.)
                self.assertEqual(result["trials"], 0)
                self.assertEqual(result["history"], [])
                self.assertEqual(result["counts"]["gradient_calls"], 1)
                self.assertEqual(result["counts"]["loss_gradient_calls"], 1)
                self.assertEqual(problem.measured, [])
                self.assertEqual(problem.evaluated, [])
                self.assertEqual(problem.checkpoints, [])

    def test_non_descending_joint_direction_retains_slope_evidence(self):
        problem = JointProblem()
        problem.loss_target = np.array([0., -1., 1e-12])
        result = problem.run(self.run_consecutive, loss_gradient=problem.loss_gradient,
                             gradient_tolerance=0.)
        self.assertEqual(result["status"], "unresolved_joint_direction")
        self.assertIn("direction_diagnostic", result)
        diagnostic = result["direction_diagnostic"]
        self.assertEqual(diagnostic["reason"], "non_descending_common_direction")
        self.assertGreater(diagnostic["joint_combination_norm"], diagnostic["gradient_tolerance"])
        self.assertEqual(diagnostic["objective_direction_slope"], 0.)
        self.assertLess(diagnostic["loss_direction_slope"], 0.)
        self.assertEqual(result["trials"], 0)
        self.assertEqual(result["history"], [])
        self.assertEqual(result["counts"]["gradient_calls"], 1)
        self.assertEqual(result["counts"]["loss_gradient_calls"], 1)
        self.assertEqual(problem.measured, [])

    def test_default_none_does_not_add_direction_diagnostic(self):
        for options in (dict(max_steps=1), dict(gradient=lambda x: np.zeros_like(x))):
            problem = EuclideanProblem()
            result = problem.run(self.run_consecutive, loss_gradient=None, **options)
            self.assertNotIn("direction_diagnostic", result)
            for state in problem.checkpoints:
                self.assertNotIn("direction_diagnostic", state)

    def test_inward_normal_reserves_half_of_descent_for_both_objectives(self):
        for energy_normal, loss_normal in ((100., 10000.), (10000., 100.)):
            for sign in (-1., 1.):
                with self.subTest(energy_normal=energy_normal, loss_normal=loss_normal, sign=sign):
                    problem = JointProblem()
                    problem.target[0] = sign * energy_normal
                    problem.loss_target[0] = sign * loss_normal
                    problem.pbe = lambda x: float(sign * .0095 + .01 * x[0])
                    problem.b = np.array([.01, 0., 0.])
                    result = problem.run(self.run_consecutive, max_steps=1, max_trials=2,
                                         loss_gradient=problem.loss_gradient)
                    self.assertEqual(result["accepted_steps"], 1)
                    step = result["x"] - problem.initial
                    self.assertLess(sign * step[0], 0.)
                    self.assertLess(abs(step[0]), .3 * .02)
                    tangent_step = step.copy()
                    tangent_step[0] = 0.
                    for full_gradient in (-problem.target, -problem.loss_weight * problem.loss_target):
                        self.assertLess(full_gradient @ tangent_step, 0.)
                        self.assertLessEqual(full_gradient @ step, .5 * (full_gradient @ tangent_step))

    def test_tiny_tangent_loss_uses_energy_direction_without_harmful_normal(self):
        for normal, tangent in ((0., 0.), (-100., 0.), (-100., -1e-12), (100., 0.)):
            with self.subTest(normal=normal, tangent=tangent):
                problem = EuclideanProblem(3)
                problem.target = np.array([0., 1., 0.])
                problem.pbe = lambda x: float(.0095 + .01 * x[0])
                problem.b = np.array([.01, 0., 0.])
                full_loss = np.array([normal, tangent, 0.])
                original_record = problem.record

                def record(x):
                    result = original_record(x)
                    result["loss"] = float(10. + full_loss @ x)
                    return result

                problem.record = record
                result = problem.run(self.run_consecutive, max_steps=1,
                                     loss_gradient=lambda x: full_loss.copy())
                self.assertEqual(result["accepted_steps"], 1)
                row = result["history"][0]
                self.assertIsNone(row["joint_angle_degrees"])
                self.assertEqual(row["joint_mixing_weight"], 1.)
                self.assertLessEqual(row["predicted_step_loss_delta"], 0.)
                self.assertGreater(result["x"][1], 0.)
                self.assertEqual(result["x"][2], 0.)
                if normal < 0.:
                    self.assertLessEqual(abs(result["x"][0]), 1e-15)

    def test_actual_pbe_still_blocks_joint_descent_before_evaluate(self):
        for measurement in (dict(gate="pass", pbe=.011), dict(gate="pass", pbe=-.011),
                            dict(gate="pass", pbe=np.nan), dict(gate="fail", pbe=0.)):
            with self.subTest(measurement=measurement):
                problem = JointProblem()
                result = problem.run(self.run_consecutive, max_trials=1,
                    loss_gradient=problem.loss_gradient, measure=lambda x, i: measurement)
                self.assertEqual(result["accepted_steps"], 0)
                self.assertEqual(problem.evaluated, [])
                self.assertEqual(problem.checkpoints, [])
                self.assertLess(result["history"][0]["predicted_step_loss_delta"], 0.)

    def test_actual_loss_gate_remains_strict_despite_predicted_joint_descent(self):
        problem = JointProblem()
        original_record = problem.record

        def record(x):
            result = original_record(x)
            result["loss"] += 1000. * float(x @ x)
            return result

        problem.record = record
        result = problem.run(self.run_consecutive, max_trials=2,
            loss_gradient=lambda x: problem.loss_gradient(x) + 2000. * x)
        self.assertEqual(result["accepted_steps"], 0)
        self.assertEqual(problem.checkpoints, [])
        self.assertEqual(len(problem.gradients), 1)
        self.assertEqual(len(problem.loss_gradients), 1)
        self.assertTrue(all(row["reason"] == "loss_increased" for row in result["history"]))
        self.assertTrue(all(row["predicted_step_loss_delta"] < 0. for row in result["history"]))

    def test_joint_gradients_are_projected_on_manifold_and_current_pbe_tangent(self):
        problem = JointProblem()
        problem.initial = np.array([1., 0., 0.])
        problem.pbe = lambda x: 0.
        problem.b = np.zeros(3)
        projected = []

        def project(x, v):
            projected.append((x.copy(), v.copy()))
            return v - x * (x @ v)

        result = problem.run(self.run_consecutive, max_steps=3,
            loss_gradient=problem.loss_gradient, project=project,
            retract=lambda x, s: (x + s) / np.linalg.norm(x + s))
        self.assertEqual(result["accepted_steps"], 3)
        self.assertAlmostEqual(np.linalg.norm(result["x"]), 1., places=14)
        for center in problem.gradients:
            loss_g = problem.loss_weight * (center - problem.loss_target)
            self.assertTrue(any(np.array_equal(x, center) and np.array_equal(v, loss_g)
                                for x, v in projected))

    def test_invalid_loss_gradient_and_projection_raise_before_measurement(self):
        for bad in (np.zeros((1, 3)), np.zeros(2), np.full(3, np.nan),
                    np.full(3, np.inf), np.ones(3) * (1. + 1j)):
            problem = JointProblem()
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                problem.run(self.run_consecutive, loss_gradient=lambda x: bad)
            self.assertEqual(problem.measured, [])
        problem = JointProblem()
        for invalid in (False, np.ones(3)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                problem.run(self.run_consecutive, loss_gradient=invalid)
        self.assertEqual(problem.gradients, [])

        def project(x, v):
            return np.full_like(v, np.nan) if np.array_equal(v, -problem.loss_weight * problem.loss_target) else v

        with self.assertRaises(ValueError):
            problem.run(self.run_consecutive, loss_gradient=problem.loss_gradient, project=project)
        self.assertEqual(problem.measured, [])

    def test_explicit_none_preserves_default_results(self):
        default = EuclideanProblem().run(self.run_consecutive, max_steps=4)
        explicit = EuclideanProblem().run(self.run_consecutive, max_steps=4, loss_gradient=None)
        np.testing.assert_equal(default, explicit)



class InequalityProposalTest(unittest.TestCase):
    def setUp(self):
        self.run_consecutive = importlib.import_module(
            "periodic_galerkin_consecutive").run_consecutive

    def problem(self, sign=1., offset=.008):
        p = JointProblem(loss_weight=.1)
        p.target = np.array([sign, .2, 0.])
        p.loss_target = np.array([sign, 0., .2])
        p.pbe = lambda x: float(sign * offset + x[0])
        return p

    def test_interior_releases_normal_component_and_caps_both_signed_boundaries(self):
        for sign in (1., -1.):
            p = self.problem(sign)
            result = p.run(self.run_consecutive, pbe_proposal="inequality",
                loss_gradient=p.loss_gradient, max_steps=1, initial_radius=.02)
            row = result["history"][0]
            self.assertEqual(result["accepted_steps"], 1)
            self.assertAlmostEqual(result["x"][0], sign * .001, places=14)
            self.assertAlmostEqual(result["record"]["pbe"], sign * .009, places=14)
            self.assertEqual(row["pbe_proposal_used"], "interior_inequality")
            self.assertLess(row["step_norm"], row["radius"])
            self.assertEqual(row["normal_correction_norm"], 0.)
            self.assertLess(row["predicted_step_loss_delta"], 0.)

    def test_inward_direction_can_cross_zero_but_is_limited_by_opposite_boundary(self):
        p = self.problem(-1., offset=-.009)
        result = p.run(self.run_consecutive, pbe_proposal="inequality",
            loss_gradient=p.loss_gradient, max_steps=1, initial_radius=.04)
        self.assertEqual(result["accepted_steps"], 1)
        self.assertAlmostEqual(result["x"][0], -.0095, places=14)
        self.assertAlmostEqual(result["record"]["pbe"], -.0005, places=14)

    def test_interior_pure_normal_gradient_is_not_wrongly_declared_unresolved(self):
        p = self.problem()
        p.target = np.array([1., 0., 0.])
        p.loss_target = p.target.copy()
        result = p.run(self.run_consecutive, pbe_proposal="inequality",
            loss_gradient=p.loss_gradient, max_steps=1, initial_radius=.001)
        self.assertEqual(result["accepted_steps"], 1)
        self.assertAlmostEqual(result["x"][0], .001, places=14)

    def test_no_pbe_slope_does_not_shorten_radius(self):
        p = JointProblem()
        result = p.run(self.run_consecutive, pbe_proposal="inequality",
            loss_gradient=p.loss_gradient, max_steps=1, initial_radius=.001)
        self.assertAlmostEqual(result["history"][0]["step_norm"], .001, places=14)
        self.assertEqual(result["history"][0]["pbe_proposal_used"], "interior_inequality")

    def test_boundary_or_tiny_model_margin_uses_existing_tangent_fallback(self):
        for offset in (.010, .010 - 1e-8):
            p = self.problem(offset=offset)
            result = p.run(self.run_consecutive, pbe_proposal="inequality",
                loss_gradient=p.loss_gradient, max_trials=1, initial_radius=.001)
            self.assertEqual(len(p.measured), 1)
            row = result["history"][0]
            self.assertEqual(row["pbe_proposal_used"], "tangent_fallback")
            self.assertLessEqual(row["predicted_pbe"], offset)
            self.assertLess(row["predicted_step_objective_delta"], 0.)
            self.assertLessEqual(row["predicted_step_loss_delta"], 0.)

    def test_inaccurate_model_actual_pbe_violation_never_evaluates(self):
        for sign in (1., -1.):
            p = self.problem(sign)
            result = p.run(self.run_consecutive, pbe_proposal="inequality",
                loss_gradient=p.loss_gradient, max_trials=1,
                measure=lambda x, i: dict(gate="pass", pbe=sign * .011))
            self.assertEqual(p.evaluated, [])
            self.assertEqual(p.checkpoints, [])
            self.assertEqual(result["history"][0]["reason"], "pbe_limit_exceeded")
            self.assertEqual(result["counts"]["secant_updates"], 1)

    def test_half_slack_does_not_replace_actual_nonincreasing_loss_gate(self):
        p = self.problem()
        result = p.run(self.run_consecutive, pbe_proposal="inequality",
            loss_gradient=p.loss_gradient, max_trials=1,
            evaluate=lambda x, i: dict(objective=0., loss=999.))
        self.assertEqual(result["accepted_steps"], 0)
        self.assertEqual(result["history"][0]["reason"], "loss_increased")

    def test_acceptances_refresh_gradients_once_and_keep_original_pbe_baseline(self):
        p = self.problem()
        result = p.run(self.run_consecutive, pbe_proposal="inequality",
            loss_gradient=p.loss_gradient, max_steps=3, initial_radius=.0001)
        self.assertEqual(result["accepted_steps"], 3)
        self.assertEqual(len(p.gradients), 3)
        self.assertEqual(len(p.loss_gradients), 3)
        self.assertGreater(result["record"]["pbe"], .008)
        self.assertLess(result["record"]["pbe"], .010)

    def test_capped_rejection_backtracks_actual_step_without_repeating_candidate(self):
        p = self.problem()
        p.initial = np.zeros(1)
        p.b = np.ones(1)
        p.record = lambda x: dict(objective=float(1-x[0]),
            loss=float(1-x[0]+2000*x[0]**2), pbe=float(.008+x[0]))
        result = p.run(self.run_consecutive, pbe_proposal="inequality",
            gradient=lambda x: -np.ones(1), loss_gradient=lambda x: np.array([-1+4000*x[0]]),
            max_steps=1, max_trials=6, initial_radius=.02)
        self.assertEqual(result["accepted_steps"], 1)
        points = [tuple(x) for _, x in p.measured]
        self.assertEqual(len(set(points)), len(points))
        self.assertAlmostEqual(result["history"][0]["radius_next"], .0005, places=14)
        self.assertLessEqual(result["trials"], 3)

    def test_legacy_default_is_exact_and_invalid_mode_fails_before_callbacks(self):
        default = JointProblem()
        explicit = JointProblem()
        a = default.run(self.run_consecutive, loss_gradient=default.loss_gradient, max_steps=2)
        b = explicit.run(self.run_consecutive, loss_gradient=explicit.loss_gradient,
                         pbe_proposal="tangent", max_steps=2)
        np.testing.assert_equal(a, b)
        for invalid in (None, False, "relaxed"):
            p = self.problem()
            with self.assertRaises(ValueError):
                p.run(self.run_consecutive, pbe_proposal=invalid)
            self.assertEqual(p.gradients, [])


if __name__ == "__main__":
    unittest.main()
