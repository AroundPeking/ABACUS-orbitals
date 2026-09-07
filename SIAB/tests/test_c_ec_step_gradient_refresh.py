"""Ec-step refresh orchestration; no response kernels execute in these tests."""

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_c_combined_step_gradient_refresh import WORKFLOW, record, save, sha
import test_c_combined_step_gradient_refresh as legacy_tests

sys.path.insert(0, str(WORKFLOW))


class EcStepRefreshTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("refresh_c_ec_step_gradient"),
                             "distinct accepted Ec-step gradient runner required")
        self.runner = importlib.import_module("refresh_c_ec_step_gradient")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage, self.parent = self.root / "new", self.root / "accepted"
        self.stage.mkdir()
        self.parent.mkdir()
        self.expected = record()
        self.coefficients = {"C": [[1.], [], [], [], []]}
        self.original = {"C": [[2.], [], [], [], []]}
        paths = {}
        for key in ("coefficient_path", "original_coefficient_path", "orbital_path",
                    "freeze_path", "active_cache_index_path"):
            paths[key] = self.parent / key
            paths[key].write_text(key)
        self.step_path = self.parent / "GRADIENT_STEP.json"
        save(self.step_path, {"direction": {"C": [[[3.]], [], [], [], []]}})
        self.loaded = dict(paths, quarter=dict(configuration=dict(frequency_batch_size=3),
            training_weights=dict(pi_weight=1., trace_log_weight=1., energy_weight=1.)),
            occupied_capture_floor=.9998999,
            candidate=dict(gradient_step_sha256=sha(self.step_path)),
            result=dict(scope="one_accepted_ec_gradient_step", source_commit="b"*40,
                candidate=self.expected, coefficient_sha256="c"*64, orbital_sha256="d"*64,
                freeze_sha256="e"*64, active_cache_index_sha256="f"*64,
                gradient_result_sha256="7"*64, gradient_acceptance_sha256="8"*64))
        self.options = dict(accepted_stage=self.parent, result_sha256="1"*64,
            acceptance_sha256="2"*64, stage=self.stage, deployment_sha256="3"*64, source_commit="a"*40)
        self.events = []
        self.patch("verify_source", return_value={})
        self.admit = self.patch("load_accepted_ec_step", side_effect=self.admission)
        self.diagnostic = {k: copy.deepcopy(self.expected[k]) for k in
                           ("loss", "rpa", "minimum_occupied_capture", "maximum_overlap_condition")}
        self.diagnostic.update(scope="radial_gradients_at_fixed_candidate_no_optimization",
            gradient_mode="energy_only", backward_passes=1, energy_gradient=dict(channels=[]),
            energy_gradient_units="Ha_per_cell_per_unit_coefficient", physical_release_gate="hold")
        self.datasets = tuple(SimpleNamespace(selected_iq=row["selected_iq"], q_weight=row["q_weight"],
            q_count=64, kpoints=tuple(range(64)), frequency_ha=SimpleNamespace(
                numel=lambda: 12, tolist=lambda: list(range(1, 13)))) for row in self.expected["rpa"]["per_q"])
        self.transport = dict(scope="transported_old_direction_diagnostic", descent_cosine=.8)
        self.guard = mock.Mock(return_value=self.expected["coefficient_guard"])
        def load(*args, **kwargs):
            self.events.append("cache")
            return self.datasets, [], self.guard
        def evaluate(*args, **kwargs):
            self.events.append("gradient")
            return copy.deepcopy(self.diagnostic)
        self.runtime = SimpleNamespace(torch=SimpleNamespace(set_num_threads=mock.Mock(),
            no_grad=mock.MagicMock(), float64="float64", tensor=lambda v, **kw: v),
            read_coefficients=mock.Mock(side_effect=lambda p, **kw:
                self.coefficients if p == paths["coefficient_path"] else self.original),
            load_frozen_c=mock.Mock(side_effect=load),
            evaluate_radial_gradients=mock.Mock(side_effect=evaluate),
            transported_descent_report=mock.Mock(return_value=self.transport))
        self.runtime_call = self.patch("_runtime", return_value=self.runtime)

    def patch(self, name, **kwargs):
        patcher = mock.patch.object(self.runner, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def admission(self, *args):
        self.events.append("admit")
        return self.loaded

    def run_refresh(self, **kwargs):
        return self.runner.refresh_gradient(**dict(self.options, **kwargs))

    def test_distinct_center_one_gradient_without_new_candidate_or_scf(self):
        before = {p: p.read_bytes() for p in self.parent.iterdir()}
        result = self.run_refresh()
        self.assertEqual(result["scope"], "accepted_ec_step_ec_gradient_refresh")
        self.assertEqual(self.events, ["admit", "cache", "gradient", "admit"])
        self.assertEqual(result["accepted_result_sha256"], "1"*64)
        self.assertEqual(result["preceding_gradient_result_sha256"], "7"*64)
        self.assertEqual(result["gradient_step_sha256"], sha(self.step_path))
        self.assertEqual(result["old_direction_reconstruction"], "validated_accepted_ec_step")
        self.assertEqual(result["transported_descent"], self.transport)
        for key, value in dict(cache_loads=1, rpa_evaluations=1, backward_passes=1,
            actual_scf_count=0, candidate_count=0, optimizer_steps=0, physical_release_gate="hold",
            actual_pbe_direction_derivatives="unmeasured", coefficient_update="none").items():
            self.assertEqual(result[key], value)
        self.runtime.evaluate_radial_gradients.assert_called_once_with(self.datasets, self.coefficients,
            occupied_capture_tolerance=1-self.loaded["occupied_capture_floor"], frequency_batch_size=3,
            weights=self.loaded["quarter"]["training_weights"], energy_only=True)
        self.runtime.torch.set_num_threads.assert_called_once_with(28)
        self.assertEqual(before, {p: p.read_bytes() for p in self.parent.iterdir()})
        self.assertEqual(json.loads((self.stage / "result/EC_GRADIENT_REFRESH.json").read_text()), result)
        provenance = json.loads((self.stage / "PROVENANCE.json").read_text())
        self.assertEqual(provenance["diagnostic_sha256"], sha(self.stage / "result/EC_GRADIENT_REFRESH.json"))

    def test_parent_rejection_prevents_runtime_and_cache(self):
        self.admit.side_effect = ValueError("rejected parent")
        with self.assertRaisesRegex(ValueError, "rejected parent"):
            self.run_refresh()
        self.runtime_call.assert_not_called()

    def test_changed_step_direction_is_rejected_before_cache(self):
        save(self.step_path, dict(direction={"C": [[[99.]], [], [], [], []]}))
        with self.assertRaises(ValueError):
            self.run_refresh()
        self.runtime.load_frozen_c.assert_not_called()

    def test_old_scope_and_commit_are_not_relabelled(self):
        for key, value in (("scope", "actual_pbe_tangent_combined_step"), ("source_commit", "a"*40)):
            old = self.loaded["result"][key]
            self.loaded["result"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_refresh()
            self.loaded["result"][key] = old
        self.runtime.load_frozen_c.assert_not_called()

    def test_exclusive_new_stage_prevents_parent_write_or_reentry(self):
        for stage in (self.parent, self.parent / "nested", self.root):
            stage.mkdir(exist_ok=True)
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                self.run_refresh(stage=stage)
        for name in ("result", "PROVENANCE.json"):
            path = self.stage / name
            path.mkdir() if name == "result" else path.write_text("old")
            with self.assertRaises(FileExistsError):
                self.run_refresh()
            path.rmdir() if path.is_dir() else path.unlink()

    def test_original_guard_precedes_new_gradient(self):
        self.guard.return_value = dict(self.expected["coefficient_guard"], gate=False)
        with self.assertRaises(ValueError):
            self.run_refresh()
        self.runtime.evaluate_radial_gradients.assert_not_called()

    def test_all_q_and_each_frequency_must_reproduce_and_second_backward_forbidden(self):
        for mutate in (lambda d: d.update(backward_passes=2),
                       lambda d: d.update(loss=d["loss"]+.001),
                       lambda d: d["rpa"]["per_q"][7]["candidate_contributions_ha"].__setitem__(11, -.1)):
            before = copy.deepcopy(self.diagnostic)
            mutate(self.diagnostic)
            with self.assertRaises(ValueError):
                self.run_refresh()
            self.assertFalse((self.stage / "PROVENANCE.json").exists())
            (self.stage / "result").rmdir()
            self.diagnostic = before

    def test_source_and_parent_postflight_failure_never_publish_success(self):
        self.admit.side_effect = [self.loaded, ValueError("changed parent")]
        with self.assertRaisesRegex(ValueError, "changed parent"):
            self.run_refresh()
        self.assertFalse((self.stage / "PROVENANCE.json").exists())
        self.assertFalse((self.stage / "result/EC_GRADIENT_REFRESH.json").exists())


class EcStepRefreshBatchTest(unittest.TestCase):
    def script(self):
        return (WORKFLOW / "run_c_ec_step_gradient.slurm").read_text()

    def test_batch_failure_and_reentry_preserve_failed_evidence(self):
        legacy_tests.GradientRefreshBatchTest.test_batch_failure_and_reentry_cannot_record_success(self)

    def test_batch_parent_alias_and_nested_paths_never_write_to_parent(self):
        legacy_tests.GradientRefreshBatchTest.test_batch_rejects_accepted_stage_before_any_lock_or_status_write(self)

    def test_batch_is_distinct_serial_diagnostic_and_shell_valid(self):
        path = WORKFLOW / "run_c_ec_step_gradient.slurm"
        self.assertTrue(path.is_file(), "distinct Ec-step refresh batch required")
        text = path.read_text()
        for value in ("#SBATCH --partition=long", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
                      "#SBATCH --cpus-per-task=28", "#SBATCH --mem=102400M", "#SBATCH --time=01:00:00",
                      "#SBATCH --no-requeue", "refresh_c_ec_step_gradient.py", "EXECUTION_LOCK"):
            self.assertIn(value, text)
        for value in ("srun ", "sbatch ", "scancel ", "mpirun ", "--nodelist"):
            self.assertNotIn(value, text)
        self.assertEqual(subprocess.run(["bash", "-n", str(path)]).returncode, 0)

    def test_source_verification_requires_own_reader_runner_and_batch(self):
        self.assertIsNotNone(importlib.util.find_spec("refresh_c_ec_step_gradient"))
        runner = importlib.import_module("refresh_c_ec_step_gradient")
        files = {name:"1"*64 for name in runner.REQUIRED_SOURCE_FILES}
        with mock.patch.object(runner.common, "verify_source", return_value=dict(files=files)) as common:
            runner.verify_source("stage", "2"*64, "3"*40)
            common.assert_called_once_with("stage", "2"*64, "3"*40)
            for name in tuple(files):
                value = files.pop(name)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.verify_source("stage", "2"*64, "3"*40)
                files[name] = value


if __name__ == "__main__":
    unittest.main()
