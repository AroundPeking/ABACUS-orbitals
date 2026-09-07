"""Stdlib orchestration tests; tensor kernels are the parent's remote test scope."""

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


WORKFLOW = (Path(__file__).resolve().parents[1] / "example_C_sternheimer" /
            "periodic_basis_optimization/galerkin_binding_workflow")
sys.path.insert(0, str(WORKFLOW))
SCRIPT = WORKFLOW / "run_c_combined_step_gradient.slurm"
HA_EV = 27.211386245988


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def record():
    energy_error = ((-.43+.51)/-.51)**2
    rpa = dict(candidate_energy_ha=-.43, reference_energy_ha=-.51,
        pi_relative_squared_error=.02, trace_log_relative_squared_error=.06-energy_error,
        energy_relative_squared_error=energy_error,
        complete_q_weight=True, q_weight_coverage=1., per_q=[dict(selected_iq=iq, q_weight=m/64,
            frequency_ha=[float(i+1) for i in range(12)],
            candidate_contributions_ha=[-.43*m/64/12]*12,
            reference_contributions_ha=[-.51*m/64/12]*12)
        for iq, m in zip((1, 22, 43, 6, 27, 23, 11, 55), (1, 8, 4, 6, 24, 12, 3, 6))])
    guard = dict(gate=True, scope="frozen_h_band_screen_not_scf_energy", scf_pbe_gate="pending",
        k_weight_convention="ABACUS_spin_included_no_renormalization", k_weight_sum=2.,
        occupied_band_sum_limit_ev_per_atom=.01, target_band_change_limit_ev=.05,
        occupied_band_sum_change_ev_per_atom=.001, maximum_target_band_change_ev=.02, minimum_gap_ev=4.)
    return dict(loss=.08, rpa=rpa, coefficient_guard=guard,
        energy_quantity="frozen_body_RPA_correlation_not_PBE_total",
        minimum_occupied_capture=.999997, maximum_overlap_condition=1.5e6,
        rpa_correlation_energy_ev_per_cell=-.43*HA_EV,
        rpa_correlation_energy_ev_per_c=-.43*HA_EV/2,
        reference_rpa_correlation_energy_ev_per_cell=-.51*HA_EV,
        reference_rpa_correlation_energy_ev_per_c=-.51*HA_EV/2)


class GradientRefreshTest(unittest.TestCase):
    def patch(self, target, name, **options):
        patcher = mock.patch.object(target, name, **options)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("refresh_c_combined_step_gradient"),
                             "independent Ec gradient refresh runner missing")
        self.runner = importlib.import_module("refresh_c_combined_step_gradient")
        temporary = tempfile.TemporaryDirectory(prefix="c-ec-refresh-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stage, self.source, self.accepted = (self.root / n for n in ("stage", "source", "accepted"))
        self.stage.mkdir()
        self.source.mkdir()
        self.accepted.mkdir()
        self.patch(self.runner, "SOURCE", new=self.source)
        files = {}
        for name in self.runner.REQUIRED_SOURCE_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# pinned source " + name + "\n")
            files[name] = sha(path)
        archive = self.stage / "source.tar.gz"
        archive.write_bytes(b"pinned source archive")
        self.deployment = dict(source_directory=str(self.source), source_commit="a"*40,
                               files=files, archive_path=str(archive), archive_sha256=sha(archive))
        save(self.stage / "DEPLOYMENT.json", self.deployment)
        self.options = dict(stage=self.stage, accepted_stage=self.accepted, source_commit="a"*40,
            deployment_sha256=sha(self.stage / "DEPLOYMENT.json"), result_sha256="1"*64,
            acceptance_sha256="2"*64)
        self.coefficients, self.original, self.previous = ({"C": [[value], [], [], [], []]}
                                                        for value in ("accepted", "original", "quarter"))
        self.old_direction = {"C": [["old D"], [], [], [], []]}
        candidate_dir, original_dir = self.accepted / "candidate", self.accepted / "quarter"
        candidate_dir.mkdir()
        original_dir.mkdir()
        paths = {}
        for key, path in (
            ("coefficient_path", candidate_dir / "COEFFICIENTS.txt"),
            ("orbital_path", candidate_dir / "C.orb"),
            ("original_coefficient_path", original_dir / "backoff_ORIGINAL_COEFFICIENTS.txt"),
            ("freeze_path", original_dir / "backoff_INPUT_FREEZE.json"),
            ("active_cache_index_path", original_dir / "backoff_ACTIVE_DATA_CACHE.json")):
            path.write_text("{}" if path.suffix == ".json" else key)
            paths[key] = path
        previous_path = original_dir / "INTERPOLATED_COEFFICIENTS.txt"
        previous_path.write_text("quarter coefficients")
        for name in ("DIRECTIONS.json", "CALIBRATION.json"):
            save(candidate_dir / name, dict(identity=name))
        self.expected = record()
        self.loaded = dict(paths, occupied_capture_floor=.9998999,
            quarter=dict(initial=dict(minimum_occupied_capture=.9999999),
                coefficient_sha256=sha(previous_path), configuration=dict(frequency_batch_size=3),
                training_weights=dict(pi_weight=1., trace_log_weight=1., energy_weight=1.)),
            candidate=dict(radius=.02, mixing_ratio=.1),
            result=dict(status="success", source_commit="b"*40, candidate=self.expected,
                directions_sha256=sha(candidate_dir / "DIRECTIONS.json"),
                calibration_sha256=sha(candidate_dir / "CALIBRATION.json"),
                center_result_sha256="3"*64,
                **{key: sha(paths[path]) for key, path in (
                    ("coefficient_sha256", "coefficient_path"), ("orbital_sha256", "orbital_path"),
                    ("freeze_sha256", "freeze_path"), ("active_cache_index_sha256", "active_cache_index_path"))}))
        self.events = []
        def accept(*args, **kwargs):
            self.events.append("admit")
            return self.loaded
        self.admit = self.patch(self.runner, "load_accepted_combined_step", side_effect=accept)
        self.diagnostic = {key: copy.deepcopy(self.expected[key]) for key in (
            "loss", "rpa", "minimum_occupied_capture", "maximum_overlap_condition")}
        self.diagnostic.update(scope="radial_gradients_at_fixed_candidate_no_optimization",
            gradient_mode="energy_only", backward_passes=1, energy_gradient=dict(channels=[]),
            energy_gradient_units="Ha_per_cell_per_unit_coefficient", physical_release_gate="hold")
        self.transport = dict(scope="transported_old_direction_diagnostic",
            descent_cosine=.9, angle_degrees=25., energy_derivative_ha_per_cell=-.01, radials=[])
        self.datasets = tuple(SimpleNamespace(selected_iq=q["selected_iq"], q_count=64,
            q_weight=q["q_weight"], kpoints=tuple(range(64)),
            frequency_ha=SimpleNamespace(numel=lambda: 12, tolist=lambda: list(range(1, 13))))
            for q in self.expected["rpa"]["per_q"])
        self.guard = mock.Mock(return_value=self.expected["coefficient_guard"])
        def load(*args, **kwargs):
            self.events.append("cache")
            return self.datasets, [dict(label=1)], self.guard
        def evaluate(*args, **kwargs):
            self.events.append("evaluate")
            return copy.deepcopy(self.diagnostic)
        def read(path, **kwargs):
            return {paths["coefficient_path"]: self.coefficients,
                    paths["original_coefficient_path"]: self.original,
                    previous_path: self.previous}[Path(path)]
        torch = SimpleNamespace(set_num_threads=mock.Mock(), equal=lambda a, b: a == b,
                                no_grad=mock.MagicMock())
        self.runtime = SimpleNamespace(torch=torch, read_coefficients=mock.Mock(side_effect=read),
            load_frozen_c=mock.Mock(side_effect=load), evaluate_radial_gradients=mock.Mock(side_effect=evaluate),
            combine_pbe_tangent=mock.Mock(return_value=dict(coefficients=self.coefficients,
                direction=self.old_direction, radius=.02, mixing_ratio=.1,
                scope="actual_pbe_tangent_candidate", direction_name="actual_pbe_tangent")),
            transported_descent_report=mock.Mock(return_value=self.transport))
        self.runtime_call = self.patch(self.runner, "_runtime", return_value=self.runtime)

    def run_refresh(self, **changes):
        return self.runner.refresh_gradient(**dict(self.options, **changes))

    def assert_no_result(self):
        self.assertFalse((self.stage / "result/EC_GRADIENT_REFRESH.json").exists())
        self.assertFalse((self.stage / "PROVENANCE.json").exists())

    def test_one_cache_full_forward_one_ec_backward_and_no_new_physics(self):
        before = {p: p.read_bytes() for p in self.accepted.rglob("*") if p.is_file()}
        result = self.run_refresh()
        self.assertEqual(self.events, ["admit", "cache", "evaluate", "admit"])
        self.admit.assert_called_with(self.accepted, "1"*64, "2"*64)
        self.runtime.torch.set_num_threads.assert_called_once_with(28)
        self.runtime.load_frozen_c.assert_called_once_with(self.loaded["freeze_path"],
            self.loaded["result"]["freeze_sha256"], self.original, self.stage / "result",
            active_cache_index=self.loaded["active_cache_index_path"],
            active_cache_index_sha256=self.loaded["result"]["active_cache_index_sha256"])
        self.runtime.evaluate_radial_gradients.assert_called_once_with(self.datasets, self.coefficients,
            occupied_capture_tolerance=1-self.loaded["occupied_capture_floor"], frequency_batch_size=3,
            weights=self.loaded["quarter"]["training_weights"], energy_only=True)
        self.runtime.transported_descent_report.assert_called_once_with(
            self.coefficients, self.diagnostic["energy_gradient"], self.old_direction)
        self.assertEqual(result["transported_descent"], self.transport)
        for key, expected in dict(cache_loads=1, rpa_evaluations=1, backward_passes=1, actual_scf_count=0,
            optimizer_steps=0, candidate_count=0, exported_candidate="none", coefficient_update="none",
            gradient_mode="energy_only", physical_release_gate="hold", ordinary_sos_qavg="pending", gw="pending").items():
            self.assertEqual(result[key], expected)
        self.assertNotIn("loss_gradient", result)
        self.assertNotIn("guard_sensitivity", result)
        self.assertEqual(result["occupied_capture_floor"], self.loaded["occupied_capture_floor"])
        self.assertEqual(result["source_commit"], "a"*40)
        self.assertEqual(result["accepted_source_commit"], "b"*40)
        diagnostic = self.stage / "result/EC_GRADIENT_REFRESH.json"
        self.assertEqual(json.loads(diagnostic.read_text()), result)
        provenance = json.loads((self.stage / "PROVENANCE.json").read_text())
        self.assertEqual(provenance["diagnostic_sha256"], sha(diagnostic))
        self.assertEqual(provenance["deployment_sha256"], self.options["deployment_sha256"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.accepted.rglob("*") if p.is_file()})
        self.assertEqual({p.name for p in (self.stage / "result").iterdir()}, {"EC_GRADIENT_REFRESH.json"})

    def test_rejected_acceptance_stops_before_runtime_and_cache(self):
        self.admit.side_effect = ValueError("not strictly accepted")
        with self.assertRaisesRegex(ValueError, "not strictly accepted"):
            self.run_refresh()
        self.runtime_call.assert_not_called()
        self.assert_no_result()

    def test_deployment_hash_source_and_file_pins_precede_cache(self):
        path = self.stage / "DEPLOYMENT.json"
        original = path.read_bytes()
        for change in (dict(source_commit="c"*40), dict(source_directory=str(self.root)),
                       dict(archive_sha256="0"*64), dict(files={})):
            save(path, dict(self.deployment, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_refresh(deployment_sha256=sha(path))
        path.write_bytes(original)
        for change in (dict(deployment_sha256="0"*64), dict(source_commit="bad")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_refresh(**change)
        required = self.source / self.runner.REQUIRED_SOURCE_FILES[0]
        required.write_text("changed running code")
        with self.assertRaises(ValueError):
            self.run_refresh()
        self.runtime.load_frozen_c.assert_not_called()
        self.assert_no_result()

    def test_new_code_must_not_claim_the_accepted_old_source_commit(self):
        self.loaded["result"]["source_commit"] = self.options["source_commit"]
        with self.assertRaises(ValueError):
            self.run_refresh()
        self.runtime_call.assert_not_called()

    def test_existing_output_or_accepted_stage_is_never_overwritten(self):
        for name in ("result", "PROVENANCE.json"):
            path = self.stage / name
            path.mkdir() if name == "result" else path.write_text("old evidence")
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                self.run_refresh()
            path.rmdir() if path.is_dir() else path.unlink()
        for path in (self.accepted, self.accepted / "new"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run_refresh(stage=path)
        self.runtime.load_frozen_c.assert_not_called()

    def test_old_direction_reconstruction_must_recover_the_exact_accepted_coefficients(self):
        self.runtime.combine_pbe_tangent.return_value["coefficients"] = self.previous
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            self.run_refresh()
        self.runtime.load_frozen_c.assert_not_called()
        self.assert_no_result()

    def test_guard_failure_precedes_full_forward(self):
        self.guard.return_value = dict(self.expected["coefficient_guard"], gate=False)
        with self.assertRaises(ValueError):
            self.run_refresh()
        self.runtime.evaluate_radial_gradients.assert_not_called()
        self.assert_no_result()

    def test_incomplete_loaded_q_k_or_frequency_sets_stop_before_forward(self):
        for key, value in (("q_count", 63), ("kpoints", tuple(range(63))), ("q_weight", .1)):
            old = getattr(self.datasets[0], key)
            setattr(self.datasets[0], key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.runner.validate_dataset_extent(self.datasets, self.loaded)
            setattr(self.datasets[0], key, old)
        with self.assertRaises(ValueError):
            self.runner.validate_dataset_extent(self.datasets[:7], self.loaded)

    def test_reproduction_checks_loss_capture_condition_and_every_frequency_contribution(self):
        for key, value in (("loss", .09), ("minimum_occupied_capture", .999996),
                           ("maximum_overlap_condition", 1.6e6)):
            changed = dict(self.diagnostic, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.runner.validate_reproduction(self.loaded, changed, self.expected["coefficient_guard"])
        for q in range(8):
            for frequency in range(12):
                for key in ("candidate_contributions_ha", "reference_contributions_ha", "frequency_ha"):
                    changed = copy.deepcopy(self.diagnostic)
                    changed["rpa"]["per_q"][q][key][frequency] += 1e-5
                    with self.subTest(q=q, frequency=frequency, key=key), self.assertRaises(ValueError):
                        self.runner.validate_reproduction(self.loaded, changed, self.expected["coefficient_guard"])

    def test_compensating_pi_and_trace_errors_do_not_reproduce_the_center(self):
        changed = copy.deepcopy(self.diagnostic)
        changed["rpa"]["pi_relative_squared_error"] += .005
        changed["rpa"]["trace_log_relative_squared_error"] -= .005
        with self.assertRaises(ValueError):
            self.runner.validate_reproduction(self.loaded, changed, self.expected["coefficient_guard"])

    def test_second_backward_loss_gradient_or_missing_energy_gradient_is_rejected(self):
        for change in (dict(backward_passes=2), dict(backward_passes=True), dict(gradient_mode="both"),
                       dict(loss_gradient={}), dict(energy_gradient=None)):
            changed = dict(self.diagnostic, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.runner.validate_reproduction(self.loaded, changed, self.expected["coefficient_guard"])

    def test_bad_reproduction_or_postflight_source_change_never_publishes_success(self):
        self.diagnostic["loss"] = .1
        with self.assertRaises(ValueError):
            self.run_refresh()
        self.runtime.transported_descent_report.assert_not_called()
        self.assert_no_result()
        (self.stage / "result").rmdir()
        self.diagnostic["loss"] = .08
        def changed_source(*args, **kwargs):
            (self.source / self.runner.REQUIRED_SOURCE_FILES[0]).write_text("changed during evaluation")
            return self.transport
        self.runtime.transported_descent_report.side_effect = changed_source
        with self.assertRaises(ValueError):
            self.run_refresh()
        self.assert_no_result()

    def test_main_exposes_only_pinned_inputs_new_source_and_exclusive_stage(self):
        argv = ["--"+name.replace("_", "-") for name in self.options]
        argv = [item for pair in zip(argv, map(str, self.options.values())) for item in pair]
        with mock.patch.object(self.runner, "refresh_gradient", return_value={}) as run, mock.patch("builtins.print"):
            self.runner.main(argv)
        self.assertEqual(run.call_args[1], {k: str(v) for k, v in self.options.items()})


class GradientRefreshBatchTest(unittest.TestCase):
    def script(self):
        self.assertTrue(SCRIPT.is_file(), "Ec-only gradient refresh batch missing")
        return SCRIPT.read_text()

    def test_clean_process_bootstraps_source_paths_without_torch(self):
        self.assertTrue((WORKFLOW / "refresh_c_combined_step_gradient.py").is_file())
        code = ("import sys; sys.path.insert(0, {workflow!r}); "
                "import refresh_c_combined_step_gradient as runner; "
                "assert str(runner.SOURCE/'SIAB/opt_orb_pytorch_dpsi') in sys.path; "
                "assert str(runner.SOURCE/'SIAB/example_C_sternheimer/periodic_basis_optimization') in sys.path; "
                "assert 'torch' not in sys.modules").format(workflow=str(WORKFLOW))
        environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run([sys.executable, "-B", "-S", "-c", code],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_single_node_single_task_resources_and_no_physics_launcher(self):
        text = self.script()
        for line in ("#SBATCH --partition=long", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
                     "#SBATCH --cpus-per-task=28", "#SBATCH --mem=102400M", "#SBATCH --time=01:00:00",
                     "#SBATCH --no-requeue"):
            self.assertIn(line, text)
        for token in ("--nodelist", "srun ", "sbatch ", "scancel ", "mpirun ", "run_c_combined_step.py"):
            self.assertNotIn(token, text)
        self.assertIn("OMP_NUM_THREADS=28", text)
        self.assertIn("MKL_NUM_THREADS=28", text)
        self.assertIn("optimizer-env-torch1121cpu/bin/python", text)
        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_batch_failure_and_reentry_cannot_record_success(self):
        text = self.script()
        with tempfile.TemporaryDirectory(prefix="c-refresh-batch-") as directory:
            root = Path(directory)
            stage = root / "stage"
            stage.mkdir()
            (root / "accepted").mkdir()
            runtime = root / "optimizer-env-torch1121cpu/bin/python"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("#!/bin/sh\nexit 7\n")
            runtime.chmod(0o700)
            # Execute only shell control flow with a failing fake Python.
            text = text.replace("/work1/ghj/c-solid-fd8-q13-standard-20260903", str(root))
            text = "\n".join(":" if line.startswith("source ") else line for line in text.splitlines()
                             if not line.startswith("/usr/bin/time "))
            batch = root / "batch.sh"
            batch.write_text(text + "\n")
            environment = dict(PATH="/usr/bin:/bin", HOME=str(root))
            values = dict(STAGE=str(stage), SOURCE=str(root / "source"), SOURCE_COMMIT="a"*40,
                          DEPLOYMENT_SHA256="d"*64, ACCEPTED_STAGE=str(root / "accepted"),
                          RESULT_SHA256="1"*64, ACCEPTANCE_SHA256="2"*64)
            environment.update({"C_EC_REFRESH_"+key: value for key, value in values.items()})
            first = subprocess.run(["bash", str(batch)], env=environment, capture_output=True, text=True)
            self.assertEqual(first.returncode, 7, first.stderr)
            self.assertEqual((stage / "STATUS").read_text().strip(), "failed")
            before = (stage / "STATUS").read_bytes()
            runtime.write_text("#!/bin/sh\nexit 0\n")
            second = subprocess.run(["bash", str(batch)], env=environment, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((stage / "STATUS").read_bytes(), before)
            fresh = root / "fresh"
            fresh.mkdir()
            environment["C_EC_REFRESH_STAGE"] = str(fresh)
            success = subprocess.run(["bash", str(batch)], env=environment, capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual((fresh / "STATUS").read_text().strip(), "success")

    def test_batch_rejects_accepted_stage_before_any_lock_or_status_write(self):
        text = self.script()
        with tempfile.TemporaryDirectory(prefix="c-refresh-batch-boundary-") as directory:
            root = Path(directory)
            old = root / "accepted"
            old.mkdir()
            nested = old / "nested"
            nested.mkdir()
            alias = root / "alias"
            alias.symlink_to(old, target_is_directory=True)
            text = "\n".join(":" if line.startswith("source ") else line for line in text.splitlines())
            batch = root / "batch.sh"
            batch.write_text(text + "\n")
            for stage in (old, nested, alias):
                environment = dict(PATH="/usr/bin:/bin", HOME=str(root))
                values = dict(STAGE=str(stage), ACCEPTED_STAGE=str(old), SOURCE=str(root),
                    SOURCE_COMMIT="a"*40, DEPLOYMENT_SHA256="d"*64,
                    RESULT_SHA256="1"*64, ACCEPTANCE_SHA256="2"*64)
                environment.update({"C_EC_REFRESH_"+k: v for k, v in values.items()})
                run = subprocess.run(["bash", str(batch)], env=environment, capture_output=True, text=True)
                with self.subTest(stage=stage):
                    self.assertNotEqual(run.returncode, 0)
                    self.assertEqual(set(old.rglob("*")), {nested})


if __name__ == "__main__":
    unittest.main()
