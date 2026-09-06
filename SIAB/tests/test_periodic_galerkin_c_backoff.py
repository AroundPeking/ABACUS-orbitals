"""Bounded interpolation of an accepted direction, never another optimization."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

import common  # noqa: F401
import periodic_galerkin_fit as fitter
import test_c_optimized_pbe as pbe_fixtures
import test_periodic_galerkin_rpa_fit as fit_fixtures


WORKFLOW = pbe_fixtures.MODULE.parent
sys.path.insert(0, str(WORKFLOW))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CBackoffTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("backoff_c_optimized_direction"), "backoff module missing")
        self.backoff = importlib.import_module("backoff_c_optimized_direction")
        self.fixture = pbe_fixtures.COptimizedPbeTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.initial = {"C": [torch.eye(31, dtype=torch.float64)[:, :n].clone()
                              for n in (3, 3, 2, 0, 0)]}
        self.best = {"C": [c.clone() for c in self.initial["C"]]}
        for c in self.best["C"][:3]:
            c[5, 0] = .15
        self.best = self.backoff.retract_coefficients(self.best)
        self.initial_path = self.root / "original_coefficients.txt"
        self.backoff.write_periodic_optimizer_coefficients(self.initial_path, self.initial)
        self.fixture.coefficient.unlink()
        self.backoff.write_periodic_optimizer_coefficients(self.fixture.coefficient, self.best)
        self.fixture.best_coefficient.write_bytes(self.fixture.coefficient.read_bytes())
        checkpoint = json.loads(self.fixture.metadata.read_text())
        checkpoint["orbital_sha256"] = sha(self.fixture.coefficient)
        self.fixture.metadata.write_text(json.dumps(checkpoint))
        self.freeze = self.root / "freeze.json"
        self.freeze.write_text(json.dumps(dict(status="success")))
        self.index = self.root / "ACTIVE_DATA_CACHE.json"
        self.index.write_text(json.dumps(dict(status="success")))
        parent = json.loads(self.fixture.result.read_text())
        parent.update(initial_sha256=sha(self.initial_path), freeze_sha256=sha(self.freeze),
                      active_cache_index_sha256=sha(self.index), coefficient_sha256=sha(self.fixture.coefficient),
                      configuration=dict(threads=1, frequency_batch_size=3),
                      training_weights=dict(pi_weight=1., trace_log_weight=1., energy_weight=1.))
        parent["initial"]["rpa"] = dict(candidate_energy_ha=-.42, reference_energy_ha=-.51)
        self.fixture.result.write_text(json.dumps(parent))
        self.output = self.root / "backoff"
        self.options = dict(initial=self.initial_path, initial_sha256=sha(self.initial_path),
            parent_result=self.fixture.result, parent_result_sha256=sha(self.fixture.result),
            parent_best_metadata_sha256=sha(self.fixture.metadata), freeze=self.freeze,
            freeze_sha256=sha(self.freeze), active_cache_index=self.index,
            active_cache_index_sha256=sha(self.index), output=self.output, alpha=.25)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def pair(self):
        def record(loss, energy):
            return dict(loss=loss, minimum_occupied_capture=.99999, maximum_overlap_condition=1.,
                coefficient_guard=dict(gate=True, scf_pbe_gate="pending"),
                energy_quantity="frozen_body_RPA_correlation_not_PBE_total",
                rpa=dict(candidate_energy_ha=energy, reference_energy_ha=-.51,
                         q_weight_coverage=1., complete_q_weight=True,
                         per_q=[dict(selected_iq=iq, frequency_ha=list(range(12)))
                                for iq in (1, 22, 43, 6, 27, 23, 11, 55)]),
                rpa_correlation_energy_ev_per_cell=energy * 27.211386245988,
                rpa_correlation_energy_ev_per_c=energy * 27.211386245988 / 2,
                reference_rpa_correlation_energy_ev_per_cell=-.51 * 27.211386245988,
                reference_rpa_correlation_energy_ev_per_c=-.51 * 27.211386245988 / 2)
        return record(.8, -.42), record(.6, -.43)

    def produce(self):
        with mock.patch.object(self.backoff, "load_frozen_c", return_value=((), [], mock.Mock())) as load, \
                mock.patch.object(self.backoff, "evaluate_pair", return_value=self.pair()) as evaluate, \
                mock.patch.object(fitter, "optimize_periodic_galerkin_basis", side_effect=AssertionError("optimization")):
            result = self.backoff.prepare_backoff(**self.options)
        self.assertEqual(load.call_args[1]["active_cache_index_sha256"], self.options["active_cache_index_sha256"])
        self.assertEqual(load.call_args[1]["active_cache_index"], self.index)
        self.assertIs(evaluate.call_args[1]["frequency_batch_size"], 3)
        return result

    def test_retraction_matches_fitter_and_interpolation_endpoints(self):
        datasets, _ = fit_fixtures.PeriodicGalerkinRpaFitTest().fixture()
        scaled = {"C": [c * 2 for c in self.initial["C"]]}
        fixed, variable, _ = fitter._validate_inputs(datasets, scaled, {"C": (0, 0, 0, 0, 0)})
        fitter._retract_variables(fixed, variable)
        expected = fitter._assemble(fixed, variable)
        actual = self.backoff.retract_coefficients(scaled)
        for a, b in zip(actual["C"], expected["C"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(a.requires_grad)
        for alpha, endpoint in ((0., actual), (1., self.backoff.retract_coefficients(self.best))):
            candidate = self.backoff.interpolate_direction(scaled, self.best, alpha)
            for a, b in zip(candidate["C"], endpoint["C"]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        candidate = self.backoff.interpolate_direction(scaled, self.best, .25)
        expected = self.backoff.retract_coefficients({"C": [
            .75 * a + .25 * b for a, b in zip(actual["C"], self.best["C"])]})
        for a, b in zip(candidate["C"], expected["C"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_algebra_rejects_nonfinite_alpha_shape_and_rank_deficiency(self):
        for alpha in (True, float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                self.backoff.interpolate_direction(self.initial, self.best, alpha)
        for changed in ({"C": self.best["C"][:3]}, {"C": [self.best["C"][0][:, :2]] + self.best["C"][1:]},
                        {"C": [torch.full_like(self.best["C"][0], float("nan"))] + self.best["C"][1:]}):
            with self.assertRaises(ValueError):
                self.backoff.interpolate_direction(self.initial, changed, .25)
        with self.assertRaisesRegex(RuntimeError, "rank deficient"):
            self.backoff.retract_coefficients({"C": [torch.zeros((3, 2), dtype=torch.float64)]})

    def test_actual_complex_response_pair_has_no_grad_and_improves_loss(self):
        datasets, initial = fit_fixtures.PeriodicGalerkinRpaFitTest().full_radial_fixture()
        fitted = fit_fixtures.PeriodicGalerkinRpaFitTest().run_full_radial_fit(datasets, initial)
        candidate = self.backoff.interpolate_direction(initial, fitted.coefficients, .25)
        calls = []
        def guard(coeff):
            calls.append(torch.is_grad_enabled())
            return dict(gate=True, scf_pbe_gate="pending")
        original_loss = fitter._global_rpa_loss
        def evaluate(*args, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            values = original_loss(*args, **kwargs)
            self.assertFalse(values[0].requires_grad)
            return values
        with mock.patch.object(self.backoff, "_global_rpa_loss", side_effect=evaluate) as response, \
                mock.patch.object(self.backoff, "prepare_periodic_occupied_reference",
                                  wraps=fitter.prepare_periodic_occupied_reference) as occupied:
            a, b = self.backoff.evaluate_pair(datasets, self.backoff.retract_coefficients(initial), candidate,
                guard=guard, frequency_batch_size=3, weights=dict(pi_weight=1., trace_log_weight=1., energy_weight=1.))
        self.assertEqual(response.call_count, 2)
        self.assertEqual(occupied.call_count, len(datasets))
        self.assertEqual(calls, [False, False])
        self.assertLess(b["loss"], a["loss"])
        self.assertAlmostEqual(b["rpa_correlation_energy_ev_per_c"],
                               b["rpa"]["candidate_energy_ha"] * 27.211386245988 / 2)

    def test_initial_export_archive_and_fresh_timings(self):
        result = self.produce()
        for key, name in (("initial_retracted_coefficient_sha256", "INITIAL_RETRACTED_COEFFICIENTS.txt"),
                          ("initial_retracted_orbital_sha256", "C_3s3p2d_initial_retracted.orb")):
            self.assertEqual(result[key], sha(self.output / name))
        for key in ("cache_load_seconds", "evaluation_seconds", "total_seconds", "peak_rss_kib"):
            self.assertGreaterEqual(result[key], 0)

    def test_initial_energy_mismatch_rejects_even_when_loss_matches(self):
        pair = self.pair()
        pair[0]["rpa"]["candidate_energy_ha"] -= .001
        with mock.patch.object(self.backoff, "load_frozen_c", return_value=((), [], mock.Mock())), \
                mock.patch.object(self.backoff, "evaluate_pair", return_value=pair):
            with self.assertRaisesRegex(ValueError, "initial.*energy"):
                self.backoff.prepare_backoff(**self.options)
        self.assertFalse((self.output / "RESULT.json").exists())

    def test_improved_loss_without_improved_energy_rejects(self):
        initial, _ = self.pair()
        candidate = dict(initial, loss=.6)
        with mock.patch.object(self.backoff, "load_frozen_c", return_value=((), [], mock.Mock())), \
                mock.patch.object(self.backoff, "evaluate_pair", return_value=(initial, candidate)):
            with self.assertRaisesRegex(ValueError, "energy"):
                self.backoff.prepare_backoff(**self.options)
        self.assertFalse((self.output / "RESULT.json").exists())

    def test_distinct_result_and_original_checker_compatibility(self):
        result = self.produce()
        self.assertEqual(result["scope"], "interpolated_retracted_optimized_direction")
        self.assertEqual(result["alpha"], .25)
        self.assertFalse({"steps_completed", "best_step", "best"} & set(result))
        path = self.output / "RESULT.json"
        prepared = self.fixture.prepare(optimizer_result=path, optimizer_result_sha256=sha(path))
        self.assertEqual(prepared["candidate_scope"], result["scope"])
        self.assertEqual(prepared["candidate_alpha"], .25)
        self.assertNotIn("optimizer_steps_completed", prepared)
        self.assertEqual((self.fixture.output / "C_original.orb").read_bytes(),
                         (self.output / "C_3s3p2d_interpolated.orb").read_bytes())
        collected = self.fixture.collect(pbe_fixtures.log_text())
        self.assertEqual(collected["pbe_total_energy_gate"], "pass")
        self.assertEqual(collected["candidate_result_sha256"], sha(path))
        self.assertEqual(collected["physical_release_gate"], "hold")

    def test_producer_rejects_wrong_parent_hashes_alpha_or_unimproved_candidate(self):
        for key in ("initial_sha256", "parent_result_sha256", "parent_best_metadata_sha256",
                    "freeze_sha256", "active_cache_index_sha256"):
            with self.subTest(key=key), mock.patch.object(self.backoff, "load_frozen_c") as load:
                with self.assertRaises((ValueError, RuntimeError)):
                    self.backoff.prepare_backoff(**dict(self.options, **{key: "0" * 64}))
                load.assert_not_called()
        for alpha in (0., 1., float("nan")):
            with self.assertRaises(ValueError):
                self.backoff.prepare_backoff(**dict(self.options, alpha=alpha))
        with mock.patch.object(self.backoff, "load_frozen_c", return_value=((), [], mock.Mock())), \
                mock.patch.object(self.backoff, "evaluate_pair", return_value=(self.pair()[0], self.pair()[0])):
            with self.assertRaisesRegex(ValueError, "improv"):
                self.backoff.prepare_backoff(**self.options)
        self.assertFalse((self.output / "RESULT.json").exists())

    def test_checker_rejects_tampered_backoff_artifacts_and_spoofed_metadata(self):
        self.produce()
        result_path = self.output / "RESULT.json"
        checker = pbe_fixtures.endpoint
        original = json.loads(result_path.read_text())
        for name in ("INTERPOLATED_COEFFICIENTS.txt", "C_3s3p2d_interpolated.orb", "BACKOFF_PROVENANCE.json",
                     "INITIAL_RETRACTED_COEFFICIENTS.txt", "C_3s3p2d_initial_retracted.orb",
                     "ORIGINAL_COEFFICIENTS.txt", "parent/BEST_CHECKPOINT.json", "parent/BEST_ORBITAL_CHECKPOINT.txt"):
            path = self.output / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"changed")
            with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                checker._optimizer_artifacts(result_path, sha(result_path))
            path.write_bytes(saved)
        for change in (dict(alpha=1.), dict(alpha=.5), dict(steps_completed=1), dict(scope="made_up"),
                       dict(candidate=self.pair()[0]), dict(candidate=dict(self.pair()[0], loss=.6)),
                       dict(parent_result_sha256="0" * 64)):
            result_path.write_text(json.dumps(dict(original, **change)))
            with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)):
                checker._optimizer_artifacts(result_path, sha(result_path))


if __name__ == "__main__":
    unittest.main()
