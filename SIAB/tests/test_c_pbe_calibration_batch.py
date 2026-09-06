"""A calibration batch is sequential, duplicate-safe, and never invokes RPA."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

WORKFLOW = Path(__file__).resolve().parents[1]/"example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
sys.path.insert(0, str(WORKFLOW))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PbeCalibrationBatchTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("run_c_pbe_direction_calibration"), "batch runner missing")
        self.m = importlib.import_module("run_c_pbe_direction_calibration")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.rows = []
        for i, (name, r) in enumerate((n, r) for n in ("T", "N") for r in (-.001, .001, -.002, .002)):
            p = self.root/"probes"/("probe_%02d" % i)
            p.mkdir(parents=True)
            (p/"PROBE.json").write_text(json.dumps(dict(direction_name=name, signed_radius=r)))
            self.rows.append(dict(path=str((p/"PROBE.json").relative_to(self.root/"probes")),
                sha256=sha(p/"PROBE.json"), status="prepared", direction_name=name, signed_radius=r))
        (self.root/"CALIBRATION_PREPARED.json").write_text(json.dumps(dict(status="success", probes=self.rows)))
        self.center = self.root/"center"
        self.center.mkdir()
        (self.center/"PBE_ENDPOINT_COLLECTION.json").write_text(json.dumps(dict(candidate_energy_ev=-309.85)))
        self.options = dict(prepared_root=self.root,
            preparation_sha256=sha(self.root/"CALIBRATION_PREPARED.json"), center_dir=self.center,
            center_preparation_sha256="b"*64, center_collection_sha256=sha(self.center/"PBE_ENDPOINT_COLLECTION.json"),
            launcher=["srun", "--mpi=pmi2", "--cpu-bind=none", "-n", "4", "/frozen/abacus_3p"])
        self.bindings = dict(center_result_sha256="a"*64, center_coefficient_sha256="c"*64,
                             center_orbital_sha256="d"*64)
        (self.root/"probes/DIRECTIONS.json").write_text(json.dumps(self.bindings))
        directions_sha = sha(self.root/"probes/DIRECTIONS.json")
        for row in self.rows:
            p = self.root/"probes"/row["path"]
            p.write_text(json.dumps(dict(self.bindings, direction_name=row["direction_name"],
                signed_radius=row["signed_radius"], directions_sha256=directions_sha)))
            row["sha256"] = sha(p)
        self.prepared = dict(self.bindings, status="success", probes=self.rows, directions_sha256=directions_sha,
                             pbe_collection_sha256=self.options["center_collection_sha256"])
        (self.root/"CALIBRATION_PREPARED.json").write_text(json.dumps(self.prepared))
        self.options["preparation_sha256"] = sha(self.root/"CALIBRATION_PREPARED.json")

    def prepare(self, **kwargs):
        path = Path(kwargs["output"])
        path.mkdir(parents=True)
        (path/self.m.PREPARATION).write_text("{}")
        return {}

    def collect(self, **kwargs):
        index = int(Path(kwargs["prepared_dir"]).name.split("_")[-1])
        row = self.rows[index]
        result = dict(status="collected", direction_name=row["direction_name"], signed_radius=row["signed_radius"],
                      candidate_energy_ev=-309.85+row["signed_radius"]*.01, scf_log_gate="pass")
        (Path(kwargs["prepared_dir"])/self.m.COLLECTION).write_text(json.dumps(result))
        return result

    def test_exact_eight_sequential_pbe_launches_no_center_rerun(self):
        with mock.patch.object(self.m, "prepare_probe_pbe", side_effect=self.prepare) as stage, \
                mock.patch.object(self.m, "collect_probe_pbe", side_effect=self.collect) as collect, \
                mock.patch.object(self.m.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            result = self.m.run_prepared_probes(**self.options)
        self.assertEqual(stage.call_count, 8)
        self.assertEqual(collect.call_count, 8)
        self.assertEqual(run.call_count, 8)
        self.assertEqual({call[1]["output"].parent for call in stage.call_args_list}, {self.root/"pbe"})
        self.assertEqual(result["mixed_curvature"], "unmeasured")
        for call in run.call_args_list:
            self.assertEqual(call[0][0], self.options["launcher"])
            self.assertNotEqual(call[1]["cwd"], self.center)
            self.assertTrue(call[1]["check"])
        with self.assertRaises(FileExistsError):
            self.m.run_prepared_probes(**self.options)

    def test_failed_executable_stops_without_retry_or_derivatives(self):
        with mock.patch.object(self.m, "prepare_probe_pbe", side_effect=self.prepare), \
                mock.patch.object(self.m, "collect_probe_pbe") as collect, \
                mock.patch.object(self.m.subprocess, "run", side_effect=subprocess.CalledProcessError(1, [])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.m.run_prepared_probes(**self.options)
        self.assertEqual(run.call_count, 1)
        collect.assert_not_called()
        self.assertFalse((self.root/"PBE_DIRECTION_CALIBRATION.json").exists())

    def test_incomplete_duplicate_or_rejected_probes_do_not_start_scf(self):
        path = self.root/"CALIBRATION_PREPARED.json"
        for rows in (self.rows[:-1], self.rows+[self.rows[0]], [dict(self.rows[0], status="rejected")]+self.rows[1:]):
            path.write_text(json.dumps(dict(status="success", probes=rows)))
            with mock.patch.object(self.m, "prepare_probe_pbe") as stage, self.assertRaises(ValueError):
                self.m.run_prepared_probes(**dict(self.options, preparation_sha256=sha(path)))
            stage.assert_not_called()

    def test_last_probe_staging_failure_starts_no_scf(self):
        def stage(**kwargs):
            if kwargs["output"].name == "probe_07":
                raise ValueError("corrupt final orbital")
            return self.prepare(**kwargs)
        with mock.patch.object(self.m, "prepare_probe_pbe", side_effect=stage), \
                mock.patch.object(self.m, "collect_probe_pbe", side_effect=self.collect), \
                mock.patch.object(self.m.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            with self.assertRaises(ValueError):
                self.m.run_prepared_probes(**self.options)
        run.assert_not_called()

    def test_frozen_batch_fits_long_cpu_memory_contract(self):
        lines = (WORKFLOW/"run_c_pbe_direction_calibration.slurm").read_text().splitlines()
        directives = dict(line[len("#SBATCH --"):].split("=", 1)
                          for line in lines if line.startswith("#SBATCH --") and "=" in line)
        self.assertEqual(directives["partition"], "long")
        self.assertEqual(directives["ntasks"], "4")
        self.assertEqual(directives["cpus-per-task"], "7")
        # Live long partition: MaxMemPerCPU=3755 MiB, 30 usable CPUs/node.
        self.assertLessEqual(int(directives["mem"].rstrip("M")), 4*7*3755)

    def test_directions_hash_and_probe_identity_are_checked_before_staging(self):
        path = self.root/"probes/DIRECTIONS.json"
        original = path.read_bytes()
        path.write_bytes(original+b" ")
        with mock.patch.object(self.m, "prepare_probe_pbe") as stage, self.assertRaises(ValueError):
            self.m.run_prepared_probes(**self.options)
        stage.assert_not_called()
        path.write_bytes(original)
        p = self.root/"probes"/self.rows[0]["path"]
        changed = json.loads(p.read_text())
        changed["signed_radius"] *= -1
        p.write_text(json.dumps(changed))
        self.rows[0]["sha256"] = sha(p)
        manifest = self.root/"CALIBRATION_PREPARED.json"
        manifest.write_text(json.dumps(dict(self.prepared, probes=self.rows)))
        with mock.patch.object(self.m, "prepare_probe_pbe") as stage, self.assertRaises(ValueError):
            self.m.run_prepared_probes(**dict(self.options, preparation_sha256=sha(manifest)))
        stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
