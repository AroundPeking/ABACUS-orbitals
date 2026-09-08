"""Pure-NumPy controller tests; no SCF, torch, or native builds are needed."""

import copy
import importlib
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


if __name__ == "__main__":
    unittest.main()
