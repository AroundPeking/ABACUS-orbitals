"""Runner stopping policy and accepted-cache routing, without production jobs."""

from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import common  # noqa: F401
import periodic_galerkin_fit as fitter
from periodic_galerkin_fit import optimize_periodic_galerkin_basis
import test_periodic_galerkin_rpa_fit as fixtures


MODULE = (Path(__file__).resolve().parents[1] / "example_C_sternheimer" /
          "periodic_basis_optimization/galerkin_binding_workflow/optimize_c_all_radial_fast.py")
spec = importlib.util.spec_from_file_location("all_radial_runner_under_test", str(MODULE))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
LABELS = (1, 2, 3, 6, 7, 8, 11, 28)
INDICES = (1, 22, 43, 6, 27, 23, 11, 55)
MULTIPLICITIES = (1, 8, 4, 6, 24, 12, 3, 6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AllRadialRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        printed = mock.patch.object(runner, "print", create=True)
        printed.start()
        self.addCleanup(printed.stop)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.initial = {"C": [torch.eye(31, dtype=torch.float64)[:, :n].clone()
                              for n in (3, 3, 2, 0, 0)]}
        self.freeze = self.root / "freeze.json"
        self.index = self.root / "ACTIVE_DATA_CACHE.json"
        frozen, records, self.datasets = [], [], []
        base, _ = fixtures.PeriodicGalerkinRpaFitTest().fixture()
        for label, iq, multiplicity in zip(LABELS, INDICES, MULTIPLICITIES):
            source = self.root / ("source_q" + str(label))
            source.mkdir()
            for name in ("manifest.dat", "status.dat", "acceptance.json"):
                (source / name).write_text("fixture {} {}\n".format(label, name))
            mapping = hashlib.sha256(("mapping" + str(label)).encode()).hexdigest()
            frozen.append(dict(label=label, dataset=str(source), acceptance=str(source / "acceptance.json"),
                               manifest_sha256=sha(source / "manifest.dat"),
                               status_sha256=sha(source / "status.dat"),
                               acceptance_sha256=sha(source / "acceptance.json"),
                               selected_iq=iq, q_weight=multiplicity / 64., physics_hash=base[0].physics_hash))
            cache = self.root / ("cache_q" + str(label))
            cache.mkdir()
            (cache / "COMPLETE.json").write_text("complete fixture {}\n".format(label))
            records.append(dict(label=label, path=str(cache), complete_sha256=sha(cache / "COMPLETE.json"),
                                binding=dict(manifest_sha256=frozen[-1]["manifest_sha256"],
                                             status_sha256=frozen[-1]["status_sha256"],
                                             identifiers=dict(acceptance_sha256=frozen[-1]["acceptance_sha256"],
                                                              mapping_sha256=mapping))))
            record = replace(base[0].kpoints[0], k_weight=1. / 32.,
                             occupied_projection_normalization=torch.eye(1, dtype=torch.complex128))
            self.datasets.append(replace(
                base[0], selected_iq=iq, q_count=64, q_weight=multiplicity / 64.,
                kpoints=(record,) * 64, frequency_ha=torch.arange(12, dtype=torch.float64),
                primitive_count=558, active_primitive_reduction=SimpleNamespace(
                    original_primitive_count=1550, mapping_sha256=mapping)))
        self.freeze.write_text(json.dumps(dict(status="success", datasets=frozen)))
        self.freeze_hash = sha(self.freeze)
        for record in records:
            record["binding"]["identifiers"]["freeze_sha256"] = self.freeze_hash
        self.index.write_text(json.dumps(dict(status="success",
            scope="exact_active_dataset_derivative_not_new_reference", records=records)))
        self.events = []
        self.guard = mock.Mock(return_value=dict(gate=True, scf_pbe_gate="pending"))
        self.physical = mock.patch.object(runner, "prepare_frozen_band_guard", side_effect=self.make_guard)
        self.prepare_guard = self.physical.start()
        self.addCleanup(self.physical.stop)

    def make_guard(self, *args, **kwargs):
        self.events.append("guard")
        return self.guard

    def cache_read(self, path, **kwargs):
        record = next(r for r in json.loads(self.index.read_text())["records"] if r["path"] == str(path))
        self.assertEqual(kwargs["cache_sha256"], record["complete_sha256"])
        self.assertEqual(kwargs["manifest_sha256"], record["binding"]["manifest_sha256"])
        self.assertEqual(kwargs["status_sha256"], record["binding"]["status_sha256"])
        self.assertEqual(kwargs["identifiers"], record["binding"]["identifiers"])
        self.assertIs(kwargs["active_coefficients"], self.initial)
        runner.check(Path(path) / "COMPLETE.json", kwargs["cache_sha256"])
        index = LABELS.index(record["label"])
        self.events.append(record["label"])
        return self.datasets[index]

    def load(self, **kwargs):
        self.assertIn("active_cache_index", inspect.signature(runner.load_frozen_c).parameters,
                      "accepted index reuse API is missing")
        options = dict(active_cache_index=self.index, active_cache_index_sha256=sha(self.index))
        options.update(kwargs)
        return runner.load_frozen_c(self.freeze, self.freeze_hash, self.initial, self.output, **options)

    def settings(self, maximum):
        settings = getattr(runner, "iteration_settings", None)
        self.assertTrue(callable(settings), "explicit runner iteration settings are missing")
        return settings(maximum)

    def test_runner_stopping_settings_and_unchanged_library_defaults(self):
        self.assertEqual(self.settings(20), dict(max_steps=20, minimum_steps=5,
            plateau_patience=2, plateau_relative_improvement=1e-6))
        self.assertEqual(self.settings(3)["minimum_steps"], 3)
        signature = inspect.signature(optimize_periodic_galerkin_basis).parameters
        self.assertEqual(signature["minimum_steps"].default, 200)
        self.assertEqual(signature["plateau_patience"].default, 300)

    def test_actual_rpa_solver_stops_at_five_on_synthetic_plateau(self):
        datasets, initial = fixtures.PeriodicGalerkinRpaFitTest().fixture()
        options = dict(fixed_nu={"C": (1,)}, objective="rpa", learning_rate=1e-12)
        previous = optimize_periodic_galerkin_basis(datasets, initial, max_steps=20,
            minimum_steps=20, plateau_patience=10, **options)
        fast = optimize_periodic_galerkin_basis(datasets, initial, **self.settings(20), **options)
        self.assertEqual(previous.steps_completed, 20)
        self.assertEqual(fast.steps_completed, 5)
        self.assertEqual(fast.stop_reason, "plateau")
        self.assertEqual(len(fast.history), 6)
        self.assertGreater(fast.history[-1]["previous_step_gradient_norm"], 0.)

    def test_archived_21885284_loss_replay_stops_at_nine_not_endpoint_equivalence(self):
        # OPTIMIZATION_HISTORY.jsonl SHA256:
        # c0d15d7ab93968e1e6bd49e6c93880589a3dab380318006699675840c46e930f
        losses = [0.08445846911081573, 0.08329529708706784, 0.08216927338805585,
                  0.0811476511774175, 0.08030393639030978, 0.08025747613277337,
                  0.08024564253838538, 0.08023361344659072, 0.08023358964659615,
                  0.08023357761082278, 0.08023356546486224, 0.08023356470000058,
                  0.08023356469698553, 0.08023356469396717, 0.08023356469320507,
                  0.08023356469244523]
        remaining = iter(losses)
        original = fitter._global_rpa_loss

        def replay(*args, **kwargs):
            values = original(*args, **kwargs)
            loss = values[0] * 0 + next(remaining)
            return (loss,) + values[1:3] + ({"periodic": loss}, values[4])

        datasets, initial = fixtures.PeriodicGalerkinRpaFitTest().fixture()
        with mock.patch.object(fitter, "_global_rpa_loss", side_effect=replay):
            result = optimize_periodic_galerkin_basis(datasets, initial,
                fixed_nu={"C": (1,)}, objective="rpa", **self.settings(20))
        self.assertEqual(result.stop_reason, "plateau")
        self.assertEqual(result.steps_completed, 9)
        self.assertEqual([row["loss"] for row in result.history], losses[:10])

    def test_cache_route_preserves_all_dataset_objects_and_reads_all_before_guard(self):
        with mock.patch.object(runner, "read_periodic_galerkin_dataset", side_effect=AssertionError("producer read")), \
                mock.patch.object(runner, "read_periodic_galerkin_dataset_cache", side_effect=self.cache_read) as read:
            datasets, records, guard = self.load()
        self.assertEqual(read.call_count, 8)
        self.assertEqual(self.events, list(LABELS) + ["guard"])
        self.assertEqual([r["label"] for r in records], list(LABELS))
        self.assertIs(guard, self.guard)
        for actual, expected in zip(datasets, self.datasets):
            self.assertIs(actual, expected)
            self.assertIs(actual.kpoints[0].occupied_projection_normalization,
                          expected.kpoints[0].occupied_projection_normalization)

    def test_bad_cached_q_k_frequency_or_mother_counts_reject_before_guard(self):
        original = self.datasets[-1]
        for changed in (replace(original, selected_iq=1), replace(original, q_count=8),
                        replace(original, q_weight=.5), replace(original, physics_hash="0" * 64),
                        replace(original, kpoints=original.kpoints[:1]),
                        replace(original, frequency_ha=original.frequency_ha[:1]),
                        replace(original, primitive_count=279),
                        replace(original, active_primitive_reduction=SimpleNamespace(
                            original_primitive_count=775,
                            mapping_sha256=original.active_primitive_reduction.mapping_sha256))):
            self.datasets[-1] = changed
            with self.subTest(changed=changed.selected_iq), \
                    mock.patch.object(runner, "read_periodic_galerkin_dataset_cache", side_effect=self.cache_read):
                with self.assertRaisesRegex(RuntimeError, "identity|contract"):
                    self.load()
            self.prepare_guard.assert_not_called()
        self.datasets[-1] = original

    def test_index_modified_during_array_loading_rejects_before_guard(self):
        def read(path, **kwargs):
            result = self.cache_read(path, **kwargs)
            if result is self.datasets[-1]:
                self.index.write_text(self.index.read_text() + " ")
            return result
        with mock.patch.object(runner, "read_periodic_galerkin_dataset_cache", side_effect=read):
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                self.load()
        self.prepare_guard.assert_not_called()

    def test_default_path_still_uses_verified_producer_reader_and_q1_guard(self):
        remaining = iter(zip(LABELS, self.datasets))

        def read_source(*args, **kwargs):
            label, dataset = next(remaining)
            self.events.append(label)
            return dataset

        with mock.patch.object(runner, "read_periodic_galerkin_dataset", side_effect=read_source) as read, \
                mock.patch.object(runner, "read_periodic_galerkin_dataset_cache", side_effect=AssertionError):
            datasets, _, _ = runner.load_frozen_c(self.freeze, self.freeze_hash, self.initial, self.output)
        self.assertEqual(len(datasets), 8)
        self.assertEqual(read.call_count, 8)
        self.assertTrue(all(call[1] == dict(include_reference_projection=False,
            verify_omitted_chunks=True, active_coefficients=self.initial) for call in read.call_args_list))
        self.prepare_guard.assert_called_once()
        self.assertEqual(self.events, [1, "guard"] + list(LABELS[1:]))

    def test_bad_index_structure_and_bindings_reject_before_any_cache_or_physics(self):
        original = json.loads(self.index.read_text())
        changes = [lambda x: x.update(status="building"), lambda x: x.update(scope="different"),
                   lambda x: x["records"].pop(), lambda x: x["records"].reverse(),
                   lambda x: x["records"][7].update(label=1),
                   lambda x: x["records"][7].update(path=x["records"][0]["path"]),
                   lambda x: x["records"][7].update(complete_sha256="bad"),
                   lambda x: x["records"][7]["binding"].update(manifest_sha256="0" * 64),
                   lambda x: x["records"][7]["binding"].update(status_sha256="0" * 64),
                   lambda x: x["records"][7]["binding"]["identifiers"].update(freeze_sha256="0" * 64),
                   lambda x: x["records"][7]["binding"]["identifiers"].update(acceptance_sha256="0" * 64),
                   lambda x: x["records"][7]["binding"]["identifiers"].pop("mapping_sha256")]
        for change in changes:
            data = json.loads(json.dumps(original))
            change(data)
            self.index.write_text(json.dumps(data))
            with self.subTest(change=change), mock.patch.object(runner, "read_periodic_galerkin_dataset_cache") as read:
                with self.assertRaises((ValueError, RuntimeError)):
                    self.load()
                read.assert_not_called()
            self.prepare_guard.assert_not_called()

    def test_wrong_index_hash_or_changed_late_source_reject_before_cache_read(self):
        with mock.patch.object(runner, "read_periodic_galerkin_dataset_cache") as read:
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                self.load(active_cache_index_sha256="0" * 64)
            read.assert_not_called()
        frozen = json.loads(self.freeze.read_text())
        for name in ("manifest.dat", "status.dat", "acceptance.json"):
            path = Path(frozen["datasets"][-1]["dataset"]) / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"changed")
            with self.subTest(name=name), mock.patch.object(runner, "read_periodic_galerkin_dataset_cache") as read:
                with self.assertRaisesRegex(RuntimeError, "SHA256"):
                    self.load()
                read.assert_not_called()
            path.write_bytes(saved)
        self.prepare_guard.assert_not_called()

    def test_late_cache_corruption_or_mapping_mismatch_prevents_any_guard(self):
        original = self.datasets[-1]
        for failure in ("hash", "mapping", "profile"):
            def read(path, **kwargs):
                result = self.cache_read(path, **kwargs)
                if result is original:
                    if failure != "mapping":
                        raise ValueError("cache " + failure + " mismatch")
                    return replace(result, active_primitive_reduction=SimpleNamespace(
                        original_primitive_count=1550, mapping_sha256="0" * 64))
                return result
            with self.subTest(failure=failure), mock.patch.object(runner, "read_periodic_galerkin_dataset_cache", side_effect=read):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.load()
            self.prepare_guard.assert_not_called()

    def test_wrong_input_nu_rejected_before_cache_or_guard(self):
        self.initial["C"][1] = torch.eye(31, dtype=torch.float64)[:, :2]
        with mock.patch.object(runner, "read_periodic_galerkin_dataset_cache") as read:
            with self.assertRaisesRegex((ValueError, RuntimeError), "profile|nu|coefficient"):
                self.load()
            read.assert_not_called()
        self.prepare_guard.assert_not_called()

    def test_reuse_and_persistence_or_unpaired_options_are_rejected(self):
        for options in (dict(persist_active_cache=True), dict(active_cache_index_sha256=None),
                        dict(active_cache_index=None)):
            with self.subTest(options=options), mock.patch.object(runner, "read_periodic_galerkin_dataset_cache") as read:
                with self.assertRaises((ValueError, RuntimeError)):
                    self.load(**options)
                read.assert_not_called()
        self.prepare_guard.assert_not_called()

    def test_cli_rejects_invalid_cache_options_before_creating_output(self):
        base = ["runner", "--freeze", "unused", "--freeze-sha256", "0" * 64,
                "--coefficients", "unused", "--coefficients-sha256", "0" * 64,
                "--benchmark", "unused", "--benchmark-sha256", "0" * 64]
        cases = (["--active-cache-index", str(self.index)],
                 ["--active-cache-index-sha256", sha(self.index)],
                 ["--active-cache-index", str(self.index), "--active-cache-index-sha256", sha(self.index),
                  "--persist-active-cache"])
        for index, flags in enumerate(cases):
            output = self.root / ("invalid_cli_" + str(index))
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", base + ["--output", str(output)] + flags):
                with self.assertRaisesRegex(RuntimeError, "together|incompatible"):
                    runner.main()
            self.assertFalse(output.exists())

    def test_main_records_settings_and_reused_index_hash_with_real_checkpoint_artifacts(self):
        coefficient_path = self.root / "initial.txt"
        runner.write_periodic_optimizer_coefficients(coefficient_path, self.initial)
        benchmark = self.root / "benchmark.json"
        benchmark.write_text(json.dumps(dict(status="success", rows=[dict(equivalence_gate="pass")],
            coefficients_sha256=sha(coefficient_path), input=dict(parent_freeze_sha256=self.freeze_hash),
            fastest=dict(threads=1, frequency_batch_size=3))))
        output = self.root / "main_output"
        argv = ["runner", "--freeze", str(self.freeze), "--freeze-sha256", self.freeze_hash,
                "--coefficients", str(coefficient_path), "--coefficients-sha256", sha(coefficient_path),
                "--benchmark", str(benchmark), "--benchmark-sha256", sha(benchmark), "--output", str(output),
                "--max-steps", "20", "--active-cache-index", str(self.index),
                "--active-cache-index-sha256", sha(self.index)]

        def solve(datasets, initial, **kwargs):
            self.assertEqual({key: kwargs[key] for key in self.settings(20)}, self.settings(20))
            history = [dict(step=0, loss=.8), dict(step=1, loss=.4)]
            for row in history:
                kwargs["best_callback"](row["step"], row["loss"], initial)
            return SimpleNamespace(coefficients=initial, history=history, best_step=1,
                                   steps_completed=1, stop_reason="plateau", total_backtracks=0)

        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(runner, "load_frozen_c", return_value=((), [], self.guard)) as load, \
                mock.patch.object(runner, "optimize_periodic_galerkin_basis", side_effect=solve):
            runner.main()
        self.assertEqual(str(load.call_args[1]["active_cache_index"]), str(self.index))
        result = json.loads((output / "RESULT.json").read_text())
        self.assertEqual(result["iteration_settings"], self.settings(20))
        self.assertEqual(result["active_cache_index_sha256"], sha(self.index))
        self.assertEqual(result["active_cache_mode"], "reuse")
        self.assertEqual(result["actual_scf_pbe_gate"], "pending")


if __name__ == "__main__":
    unittest.main()
