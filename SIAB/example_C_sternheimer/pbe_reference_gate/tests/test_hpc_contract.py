import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_pbe_branch.slurm"
sys.path.insert(0, str(ROOT))

import audit_gate
import prepare_gate


class HpcStaticContractTests(unittest.TestCase):
    def test_normal_full_node_array_contract(self):
        text = RUNNER.read_text()
        for value in (
            "#SBATCH --partition=normal",
            "#SBATCH --array=0-3",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=126500M",
            "#SBATCH --time=24:00:00",
            "#SBATCH --no-requeue",
            "#SBATCH --exclusive",
            "set -euo pipefail",
            "export OMP_NUM_THREADS=32",
            "export MKL_NUM_THREADS=32",
            "export OPENBLAS_NUM_THREADS=32",
            'mpirun -np 1 -ppn 1 "$ABACUS_REAL"',
        ):
            self.assertIn(value, text)
        self.assertNotIn("debug", text.lower())

    def test_runner_checks_inputs_and_restart_semantics(self):
        text = RUNNER.read_text()
        for value in (
            "GATE_ROOT",
            "ABACUS_ARTIFACT",
            "PSEUDO_ASSET",
            "ORBITAL_ASSET",
            "SLURM_JOB_PARTITION",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_CPUS_PER_TASK",
            "SLURM_NTASKS",
            "SLURM_JOB_NUM_NODES",
            '"$PREPARE" prepare',
            '"$PREPARE" render --mode fixed --restart',
            '"$PREPARE" render --mode free',
            'grep -q "^ocp 0$"',
            'grep -q "^efield_flag 0$"',
            'grep -q "^init_wfc file$"',
            'grep -q "^init_chg file$"',
            "trap 'record_failure",
        ):
            self.assertIn(value, text)
        self.assertIn('"$GATE_ROOT"/runs/.*.prepare.lock', text)
        self.assertIn(".task4-prepare.guard", text)
        self.assertIn("SLURM_ARRAY_JOB_ID", text)


class RuntimeFileContractTests(unittest.TestCase):
    def test_restart_copy_retries_short_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            content = b"0123456789abcdef\n"
            source.write_bytes(content)
            real_write = os.write

            def short_write(descriptor, data):
                return real_write(descriptor, bytes(data[:3]))

            with mock.patch.object(audit_gate.os, "write", side_effect=short_write):
                record = audit_gate._copy_regular(source, destination, "restart")

            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(record["size"], len(content))


class FakeHpcEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name).resolve()
        cls.assets = cls.base / "assets"
        cls.assets.mkdir()
        cls.pseudo = cls.assets / "C_ONCV_PBE-1.0.upf"
        cls.orbital = cls.assets / "C_gga_10au_100Ry_3s3p2d.orb"
        cls.pseudo.write_text("fake pseudo\n")
        cls.orbital.write_text("fake orbital\n")
        cls.bin_dir = cls.base / "bin"
        cls.bin_dir.mkdir()
        cls.fake_mpirun = cls.bin_dir / "mpirun"
        cls.fake_mpirun.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "[[ $1 == -np && $2 == 1 && $3 == -ppn && $4 == 1 ]]\n"
            "shift 4\n"
            'exec "$@"\n'
        )
        cls.fake_abacus = cls.bin_dir / "abacus"
        cls.fake_abacus.write_text(cls._fake_abacus_text())
        for path in (cls.fake_mpirun, cls.fake_abacus):
            path.chmod(path.stat().st_mode | stat.S_IXUSR)

        cls.completed_gate = cls.base / "completed-gate"
        for task in range(4):
            completed = cls._run_task(cls.completed_gate, task)
            if completed.returncode != 0:
                raise AssertionError(
                    f"fake task {task} failed\nstdout={completed.stdout}\n"
                    f"stderr={completed.stderr}"
                )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def _fake_abacus_text():
        return textwrap.dedent(
            r"""#!/usr/bin/env python3
import os
from pathlib import Path

phase = Path.cwd()
if os.environ.get("FAKE_ABACUS_FAIL_PHASE") == phase.name:
    raise SystemExit(17)
input_text = (phase / "INPUT").read_text()
restart = "init_wfc file" in input_text
out = phase / "OUT.C_PBE_REFERENCE_GATE"
out.mkdir(exist_ok=True)
branch = phase.parent.name
direction = int(branch[-1]) if branch.startswith("dir") else 0
energy = -147.4676776027294 + direction * 1.0e-8
if phase.name in {"fixed_cold", "field_seed", "free_restart1"}:
    energy += 1.0e-9

log = []
if restart:
    for spin in (1, 2):
        print(f"Read NAO wave functions from {out / ('wfs%d_nao.txt' % spin)}")
        log.append(f"Read in electron density: {out / ('chgs%d.cube' % spin)}")
log.extend(("#SCF IS CONVERGED#", f"!FINAL_ETOT_IS {energy:.16f} eV"))
(out / "running_scf.log").write_text("\n".join(log) + "\n")

spin1 = [1.0, 1.0, 1.0] + [0.0] * 19
spin2 = [1.0] + [0.0] * 21
rows = ["1     # ionic step", "Electronic state energy (eV) and occupations", "Spin number 2"]
for spin, occupations in ((1, spin1), (2, spin2)):
    rows.append(f"spin={spin} k-point=1/1 Cartesian=0.0000000 0.0000000 0.0000000 (123 plane wave)")
    for index, occupation in enumerate(occupations, 1):
        rows.append(f"{index} {-51.0 + index:.14f} {occupation:.15f}")
    rows.append("")
(out / "eig_occ.txt").write_text("\n".join(rows) + "\n")

for spin in (1, 2):
    (out / f"wfs{spin}_nao.txt").write_text(
        f"wavefunction branch={branch} phase={phase.name} spin={spin}\n"
    )
    (out / f"chgs{spin}.cube").write_text(
        f"density branch={branch} phase={phase.name} spin={spin}\n"
    )
"""
        )

    @classmethod
    def _run_task(cls, root, task, *, fail_phase=None):
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{cls.bin_dir}{os.pathsep}{environment['PATH']}",
                "GATE_ROOT": str(root),
                "ABACUS_ARTIFACT": str(cls.fake_abacus),
                "PSEUDO_ASSET": str(cls.pseudo),
                "ORBITAL_ASSET": str(cls.orbital),
                "PYTHON_EXE": sys.executable,
                "SLURM_JOB_PARTITION": "normal",
                "SLURM_ARRAY_TASK_ID": str(task),
                "SLURM_ARRAY_TASK_COUNT": "4",
                "SLURM_CPUS_PER_TASK": "32",
                "SLURM_NTASKS": "1",
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_TASKS_PER_NODE": "1",
                "SLURM_JOB_ID": "9001",
                "SLURM_ARRAY_JOB_ID": "9001",
            }
        )
        if fail_phase is not None:
            environment["FAKE_ABACUS_FAIL_PHASE"] = fail_phase
        return subprocess.run(
            ["bash", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def copy_gate(self):
        destination = self.base / f"case-{next(tempfile._get_candidate_names())}"
        shutil.copytree(self.completed_gate, destination)
        return destination

    def assert_not_passed(self, root):
        try:
            summary = audit_gate.audit_gate(root)
        except (OSError, ValueError):
            return
        self.assertNotEqual(summary["status"], "PBE_GATE_PASSED")

    def test_fake_four_task_execution_passes_global_audit(self):
        summary = audit_gate.audit_gate(self.completed_gate)
        self.assertEqual(summary["status"], "PBE_GATE_PASSED")
        self.assertEqual(
            summary["restart_chain_evidence"]["status"],
            "RESTART_CHAIN_VERIFIED",
        )
        self.assertEqual(len(summary["phases"]), 11)

    def test_phase_and_branch_manifests_capture_required_evidence(self):
        phase = self.completed_gate / "runs/dir1/free_restart1"
        phase_manifest = json.loads((phase / "PHASE_COMPLETE.json").read_text())
        restart_manifest = json.loads((phase / "RESTART_PROVENANCE.json").read_text())
        branch_manifest = json.loads(
            (self.completed_gate / "runs/dir1/BRANCH_COMPLETE.json").read_text()
        )
        self.assertEqual(phase_manifest["status"], "PHASE_COMPLETE")
        self.assertTrue(phase_manifest["restart_loaded"])
        self.assertEqual(restart_manifest["status"], "VERIFIED")
        self.assertEqual(branch_manifest["status"], "BRANCH_COMPLETE")
        self.assertEqual(phase_manifest["scheduler"]["partition"], "normal")
        self.assertEqual(phase_manifest["scheduler"]["cpus_per_task"], 32)
        self.assertEqual(phase_manifest["scheduler"]["ntasks"], 1)
        self.assertEqual(
            set(restart_manifest["files"]),
            {"wfs1_nao.txt", "wfs2_nao.txt", "chgs1.cube", "chgs2.cube"},
        )
        for record in restart_manifest["files"].values():
            self.assertEqual(record["source_sha256"], record["destination_sha256"])
            self.assertEqual(record["source_sha256"], record["snapshot_sha256"])

    def test_rejects_missing_or_tampered_restart_evidence(self):
        mutations = {
            "missing snapshot": lambda root: (
                root / "runs/dir0/free_restart1/restart_input_snapshot/wfs1_nao.txt"
            ).unlink(),
            "source output": lambda root: (
                root / "runs/dir0/field_seed/OUT.C_PBE_REFERENCE_GATE/wfs1_nao.txt"
            ).write_text("tampered\n"),
            "load stdout": lambda root: (
                root / "runs/dir0/free_restart1/abacus.stdout"
            ).write_text("missing load evidence\n"),
            "restart manifest": lambda root: (
                root / "runs/dir0/free_restart1/RESTART_PROVENANCE.json"
            ).unlink(),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                root = self.copy_gate()
                mutate(root)
                self.assert_not_passed(root)

    def test_planned_restart_rejects_tampered_destination_before_launch(self):
        root = self.base / f"planned-{next(tempfile._get_candidate_names())}"
        completed = self._run_task(root, 0, fail_phase="fixed_restart")
        self.assertNotEqual(completed.returncode, 0)
        phase = root / "runs/fixed/fixed_restart"
        (phase / "OUT.C_PBE_REFERENCE_GATE/wfs1_nao.txt").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "destination.*wfs1_nao"):
            audit_gate._verify_restart_provenance(
                root, "fixed", "fixed_restart", require_verified=False
            )

    def test_restart_manifest_paths_are_verified_not_trusted(self):
        root = self.copy_gate()
        path = root / "runs/dir0/free_restart1/RESTART_PROVENANCE.json"
        value = json.loads(path.read_text())
        value["files"]["wfs1_nao.txt"][
            "snapshot_relative_path"
        ] = "runs/dir0/free_restart1/restart_input_snapshot/wfs2_nao.txt"
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "relative path"):
            audit_gate._verify_restart_provenance(
                root, "dir0", "free_restart1", require_verified=True
            )

    def test_restart_load_evidence_rejects_extra_spin_message(self):
        root = self.copy_gate()
        phase = root / "runs/dir0/free_restart1"
        stdout = phase / "abacus.stdout"
        stdout.write_text(
            stdout.read_text()
            + "Read NAO wave functions from /unexpected/wfs3_nao.txt\n"
        )
        with self.assertRaisesRegex(ValueError, "exactly two.*wave-function"):
            audit_gate._restart_load_lines(phase)

    def test_rejects_manifest_binary_and_resource_tampering(self):
        mutations = {
            "phase manifest": lambda root: (
                root / "runs/dir1/free_restart2/PHASE_COMPLETE.json"
            ).unlink(),
            "branch manifest": lambda root: (
                root / "runs/dir1/BRANCH_COMPLETE.json"
            ).unlink(),
            "binary hash": lambda root: self._mutate_json(
                root / "runs/dir1/free_restart2/PHASE_COMPLETE.json",
                lambda data: data["executable"].update({"sha256": "0" * 64}),
            ),
            "resource": lambda root: self._mutate_json(
                root / "runs/dir1/free_restart2/PHASE_COMPLETE.json",
                lambda data: data["scheduler"].update({"cpus_per_task": 31}),
            ),
            "preparation provenance": lambda root: self._mutate_json(
                root / "runs/dir1/BRANCH_PROVENANCE.json",
                lambda data: data.update({"tampered_after_prepare": True}),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                root = self.copy_gate()
                mutate(root)
                self.assert_not_passed(root)

    @staticmethod
    def _mutate_json(path, mutate):
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data, sort_keys=True) + "\n")

    def test_existing_branch_is_rejected_without_overwrite(self):
        marker = self.completed_gate / "runs/fixed/fixed_cold/INPUT"
        before = marker.read_bytes()
        completed = self._run_task(self.completed_gate, 0)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(marker.read_bytes(), before)

    def test_failed_abacus_does_not_publish_branch_complete(self):
        root = self.base / f"failed-{next(tempfile._get_candidate_names())}"
        completed = self._run_task(root, 0, fail_phase="fixed_restart")
        self.assertNotEqual(completed.returncode, 0)
        branch = root / "runs/fixed"
        self.assertFalse((branch / "BRANCH_COMPLETE.json").exists())
        self.assertTrue((branch / "RUN_FAILED.json").is_file())

    def test_wrong_live_resource_contract_stops_before_prepare(self):
        root = self.base / f"wrong-resource-{next(tempfile._get_candidate_names())}"
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin_dir}{os.pathsep}{environment['PATH']}",
                "GATE_ROOT": str(root),
                "ABACUS_ARTIFACT": str(self.fake_abacus),
                "PSEUDO_ASSET": str(self.pseudo),
                "ORBITAL_ASSET": str(self.orbital),
                "PYTHON_EXE": sys.executable,
                "SLURM_JOB_PARTITION": "normal",
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_ARRAY_TASK_COUNT": "4",
                "SLURM_CPUS_PER_TASK": "31",
                "SLURM_NTASKS": "1",
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_TASKS_PER_NODE": "1",
                "SLURM_JOB_ID": "9002",
                "SLURM_ARRAY_JOB_ID": "9002",
            }
        )
        completed = subprocess.run(
            ["bash", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((root / "runs/fixed").exists())

    def test_initial_control_files_remain_bound_to_prepare_provenance(self):
        root = self.base / f"initial-control-{next(tempfile._get_candidate_names())}"
        branch = prepare_gate.prepare_branch(
            root,
            branch="fixed",
            pseudo=self.pseudo,
            orbital=self.orbital,
        )
        environment = {
            "SLURM_JOB_PARTITION": "normal",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_TASK_COUNT": "4",
            "SLURM_CPUS_PER_TASK": "32",
            "SLURM_NTASKS": "1",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_TASKS_PER_NODE": "1",
            "SLURM_JOB_ID": "9003",
            "SLURM_ARRAY_JOB_ID": "9003",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            audit_gate.initialize_branch_run(root, "fixed", self.fake_abacus, RUNNER)
        stru = branch / "fixed_cold/STRU"
        stru.write_text(
            stru.read_text().replace("37.79452249150619", "18.89726124575309")
        )
        with self.assertRaisesRegex(ValueError, "preparation provenance"):
            audit_gate.preflight_phase(root, "fixed", "fixed_cold")


if __name__ == "__main__":
    unittest.main()
