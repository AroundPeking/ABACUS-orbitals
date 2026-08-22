import json
import os
import re
import shlex
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
COMMON_RUNNER = ROOT / "run_pbe_branch_common.sh"
SERVER66_RUNNER = ROOT / "run_pbe_branch_server66.slurm"
SERVER66_ENVIRONMENT = ROOT / "server66_runtime_env.sh"
SUBMITTER = ROOT / "submit_pbe_gate.sh"
README = ROOT / "README.md"
sys.path.insert(0, str(ROOT))

import audit_gate
import prepare_gate
from resource_profiles import get_resource_profile


RESOURCE_PROFILES = ROOT / "resource_profiles.py"
PREPARE_SOURCE = ROOT / "prepare_gate.py"
AUDIT_SOURCE = ROOT / "audit_gate.py"
GATE_CONTRACT_SOURCE = ROOT / "gate_contract.py"


class ResourceProfileTests(unittest.TestCase):
    def test_df_dcu_profile(self):
        self.assertEqual(
            get_resource_profile("df_dcu"),
            {
                "name": "df_dcu",
                "partition": "normal",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 30,
                "memory_mb": 110610,
                "time_limit": "1-00:00:00",
                "over_subscribe": "NO",
            },
        )

    def test_server66_profile(self):
        self.assertEqual(
            get_resource_profile("server66"),
            {
                "name": "server66",
                "partition": "640",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 48,
                "memory_mb": 180000,
                "time_limit": "1-00:00:00",
                "over_subscribe": "OK",
            },
        )

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown C PBE gate profile"):
            get_resource_profile("automatic")

    def test_profile_returns_a_copy(self):
        profile = get_resource_profile("df_dcu")
        profile["partition"] = "changed"
        self.assertEqual(get_resource_profile("df_dcu")["partition"], "normal")

    def test_shell_command_prints_server66_profile(self):
        completed = subprocess.run(
            [sys.executable, str(RESOURCE_PROFILES), "shell", "server66"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(
            completed.stdout,
            "server66|640|1|1|48|180000|1-00:00:00|OK\n",
        )
        self.assertEqual(completed.stderr, "")

    def test_shell_command_rejects_unknown_profile_cleanly(self):
        completed = subprocess.run(
            [sys.executable, str(RESOURCE_PROFILES), "shell", "automatic"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertIn("unknown C PBE gate profile", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        lines = completed.stderr.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("usage:"))
        self.assertIn("resource_profiles.py: error:", lines[1])


class HpcStaticContractTests(unittest.TestCase):
    def test_runtime_sources_remain_python37_compatible(self):
        for path in (
            ROOT / "audit_gate.py",
            ROOT / "resource_profiles.py",
            ROOT / "submit_pbe_gate.sh",
            COMMON_RUNNER,
        ):
            self.assertNotIn("missing_ok=True", path.read_text())

    def test_df_dcu_entrypoint_has_exact_resource_shape_and_is_thin(self):
        text = RUNNER.read_text()
        directives = [
            line
            for line in text.splitlines()
            if line.startswith("#SBATCH ")
        ]
        self.assertEqual(
            directives,
            [
                "#SBATCH --partition=normal",
                "#SBATCH --array=0-3",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "#SBATCH --ntasks-per-node=1",
                "#SBATCH --cpus-per-task=30",
                "#SBATCH --mem=110610M",
                "#SBATCH --time=24:00:00",
                "#SBATCH --no-requeue",
                "#SBATCH --exclusive",
            ],
        )
        for value in (
            "export C_PBE_GATE_PROFILE=df_dcu",
            ': "${C_PBE_GATE_ENTRYPOINT:?C_PBE_GATE_ENTRYPOINT is required}"',
            ': "${C_PBE_GATE_COMMON_RUNNER:?C_PBE_GATE_COMMON_RUNNER is required}"',
            'source "$C_PBE_GATE_COMMON_RUNNER"',
        ):
            self.assertIn(value, text)
        self.assertNotIn("BASH_SOURCE", text)
        self.assertNotIn("prepare_branch_once", text)
        self.assertNotIn("mpirun", text)

    def test_server66_entrypoint_has_exact_resource_shape_and_is_thin(self):
        text = SERVER66_RUNNER.read_text()
        directives = [
            line
            for line in text.splitlines()
            if line.startswith("#SBATCH ")
        ]
        self.assertEqual(
            directives,
            [
                "#SBATCH --partition=640",
                "#SBATCH --array=0-3",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "#SBATCH --ntasks-per-node=1",
                "#SBATCH --cpus-per-task=48",
                "#SBATCH --mem=180000M",
                "#SBATCH --time=24:00:00",
                "#SBATCH --no-requeue",
            ],
        )
        self.assertNotIn("#SBATCH --exclusive", text)
        for value in (
            "export C_PBE_GATE_PROFILE=server66",
            ': "${C_PBE_GATE_ENTRYPOINT:?C_PBE_GATE_ENTRYPOINT is required}"',
            ': "${C_PBE_GATE_COMMON_RUNNER:?C_PBE_GATE_COMMON_RUNNER is required}"',
            'source "$C_PBE_GATE_COMMON_RUNNER"',
        ):
            self.assertIn(value, text)
        self.assertNotIn("BASH_SOURCE", text)
        self.assertNotIn("prepare_branch_once", text)
        self.assertNotIn("mpirun", text)

    def test_common_runner_owns_the_runtime_contract(self):
        text = COMMON_RUNNER.read_text()
        for value in (
            "resource_profiles.py",
            "C_PBE_GATE_PROFILE",
            "C_PBE_GATE_ENTRYPOINT",
            "C_PBE_GATE_COMMON_RUNNER",
            '"$RESOURCE_PROFILES_REAL" shell "$C_PBE_GATE_PROFILE"',
            "GATE_ROOT",
            "ABACUS_ARTIFACT",
            "PSEUDO_ASSET",
            "ORBITAL_ASSET",
            "SLURM_JOB_PARTITION",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_CPUS_PER_TASK",
            "SLURM_NTASKS",
            "SLURM_JOB_NUM_NODES",
            '"$PREPARE_REAL" prepare',
            '"$PREPARE_REAL" render --mode fixed --restart',
            '"$PREPARE_REAL" render --mode free',
            'grep -q "^ocp 0$"',
            'grep -q "^efield_flag 0$"',
            'grep -q "^init_wfc file$"',
            'grep -q "^init_chg file$"',
            "trap 'record_failure",
            '"$MPIRUN_REAL" -np 1 -ppn 1 "$ABACUS_REAL"',
        ):
            self.assertIn(value, text)
        self.assertIn('"$GATE_ROOT"/runs/.*.prepare.lock', text)
        self.assertIn(".task4-prepare.guard", text)
        self.assertIn("SLURM_ARRAY_JOB_ID", text)

    def test_runtime_environment_is_sourced_before_mpirun_resolution(self):
        text = COMMON_RUNNER.read_text()
        source_index = text.index('source "$ABACUS_ENV_REAL"')
        mpirun_index = text.index('command -v -- mpirun')
        launch_index = text.index('"$MPIRUN_REAL" -np 1 -ppn 1 "$ABACUS_REAL"')
        self.assertLess(source_index, mpirun_index)
        self.assertLess(mpirun_index, launch_index)
        self.assertEqual(
            [
                line.strip()
                for line in text.splitlines()
                if line.strip().startswith("source ")
            ],
            ['source "$ABACUS_ENV_REAL"'],
        )
        for value in (
            "ABACUS_ENV_SCRIPT",
            "export OMP_NUM_THREADS=$PROFILE_CPUS_PER_TASK",
            "export MKL_NUM_THREADS=$PROFILE_CPUS_PER_TASK",
            "export OPENBLAS_NUM_THREADS=$PROFILE_CPUS_PER_TASK",
        ):
            self.assertIn(value, text[source_index:])

    def test_server66_environment_is_minimal_and_credential_free(self):
        text = SERVER66_ENVIRONMENT.read_text()
        self.assertEqual(
            text.splitlines(),
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "source /etc/profile.d/modules.sh",
                "module purge",
                "module load gcc10.2",
                "module load intel20u4",
                'export LD_LIBRARY_PATH="/home/apps/gcc10.2/lib64:/home/apps/gcc10.2/lib:/home/apps/intel20u4/lib/intel64_lin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
            ],
        )
        lowered = text.lower()
        for forbidden in (
            "/home/ghj/.bashrc",
            "api_key",
            "apikey",
            "token",
            "conda",
            "alias ",
            "/home/ghj/",
            "cuda",
            "pythonpath",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_common_runner_resolves_every_executed_or_hashed_file(self):
        text = COMMON_RUNNER.read_text()
        for value in (
            "RESOURCE_PROFILES_REAL=$(resolve_regular",
            "ENTRYPOINT_REAL=$(resolve_regular",
            "COMMON_RUNNER_REAL=$(resolve_regular",
            "PYTHON_REAL=$(resolve_regular",
            "PREPARE_REAL=$(resolve_regular",
            "AUDIT_REAL=$(resolve_regular",
            "GATE_CONTRACT_REAL=$(resolve_regular",
            "ABACUS_ENV_REAL=$(resolve_regular",
            "ABACUS_REAL=$(resolve_regular",
            "PSEUDO_REAL=$(resolve_regular",
            "ORBITAL_REAL=$(resolve_regular",
            "MPIRUN_REAL=$(resolve_regular",
        ):
            self.assertIn(value, text)

    def test_common_runner_passes_complete_runtime_chain_to_auditor(self):
        text = COMMON_RUNNER.read_text()
        for value in (
            '--gate-profile "$C_PBE_GATE_PROFILE"',
            '--python "$PYTHON_REAL"',
            '--prepare-gate "$PREPARE_REAL"',
            '--audit-gate "$AUDIT_REAL"',
            '--gate-contract "$GATE_CONTRACT_REAL"',
            '--resource-profiles "$RESOURCE_PROFILES_REAL"',
            '--entrypoint "$ENTRYPOINT_REAL"',
            '--common-runner "$COMMON_RUNNER_REAL"',
            '--abacus "$ABACUS_REAL"',
            '--environment-script "$ABACUS_ENV_REAL"',
            '--mpirun "$MPIRUN_REAL"',
        ):
            self.assertIn(value, text)

    def test_df_dcu_directives_remain_debug_free(self):
        text = RUNNER.read_text()
        for value in (
            "#SBATCH --partition=normal",
            "#SBATCH --array=0-3",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=30",
            "#SBATCH --mem=110610M",
            "#SBATCH --time=24:00:00",
            "#SBATCH --no-requeue",
            "#SBATCH --exclusive",
        ):
            self.assertIn(value, text)
        self.assertNotIn("debug", text.lower())


class RuntimeFileContractTests(unittest.TestCase):
    @staticmethod
    def _restart_phase(root):
        phase = Path(root).resolve() / "phase"
        out = phase / "OUT.C_PBE_REFERENCE_GATE"
        out.mkdir(parents=True)
        for name in audit_gate.RESTART_FILES:
            (out / name).write_text(f"restart {name}\n")
        return phase, out

    @staticmethod
    def _write_restart_load_logs(phase, wfc_paths, charge_paths):
        (phase / "abacus.stdout").write_text(
            "".join(
                f"Read NAO wave functions from {path}\n" for path in wfc_paths
            )
        )
        (phase / "OUT.C_PBE_REFERENCE_GATE/running_scf.log").write_text(
            "".join(
                f"Read in electron density: {path}\n" for path in charge_paths
            )
        )

    @staticmethod
    def _scheduler_fixture(profile_name="df_dcu"):
        profile = get_resource_profile(profile_name)
        raw = (
            "JobId=9100 ArrayJobId=9001 ArrayTaskId=0 "
            f"Partition={profile['partition']} NumNodes={profile['nodes']} "
            f"NumCPUs={profile['cpus_per_task']} NumTasks={profile['ntasks']} "
            f"CPUs/Task={profile['cpus_per_task']} "
            f"MinMemoryNode={profile['memory_mb']}M "
            f"TimeLimit={profile['time_limit']} "
            f"OverSubscribe={profile['over_subscribe']}\n"
        )
        environment = {
            "C_PBE_GATE_PROFILE": profile_name,
            "SLURM_JOB_PARTITION": profile["partition"],
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_TASK_COUNT": "4",
            "SLURM_CPUS_PER_TASK": str(profile["cpus_per_task"]),
            "SLURM_NTASKS": str(profile["ntasks"]),
            "SLURM_JOB_NUM_NODES": str(profile["nodes"]),
            "SLURM_TASKS_PER_NODE": "1",
            "SLURM_MEM_PER_NODE": str(profile["memory_mb"]),
            "SLURM_JOB_ID": "9100",
            "SLURM_ARRAY_JOB_ID": "9001",
        }
        return raw, environment

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

    def test_restart_load_accepts_real_relative_phase_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase, _ = self._restart_phase(tmp)
            self._write_restart_load_logs(
                phase,
                (
                    "OUT.C_PBE_REFERENCE_GATE/wfs1_nao.txt",
                    "OUT.C_PBE_REFERENCE_GATE/wfs2_nao.txt",
                ),
                (
                    "OUT.C_PBE_REFERENCE_GATE/chgs1.cube",
                    "OUT.C_PBE_REFERENCE_GATE/chgs2.cube",
                ),
            )
            evidence = audit_gate._restart_load_lines(phase)
            self.assertEqual(len(evidence["wfc_load_lines"]), 2)
            self.assertEqual(len(evidence["charge_load_lines"]), 2)

    def test_restart_load_accepts_server66_abacus_charge_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase, _ = self._restart_phase(tmp)
            (phase / "abacus.stdout").write_text(
                "Read NAO wave functions from "
                "OUT.C_PBE_REFERENCE_GATE/wfs1_nao.txt\n"
                "Read NAO wave functions from "
                "OUT.C_PBE_REFERENCE_GATE/wfs2_nao.txt\n"
            )
            (phase / "OUT.C_PBE_REFERENCE_GATE/running_scf.log").write_text(
                "Read electron density from file\n"
                "Find the file OUT.C_PBE_REFERENCE_GATE/chgs1.cube , "
                "try to read it.\n"
                "Read electron density from file: "
                "OUT.C_PBE_REFERENCE_GATE/chgs1.cube\n"
                "Find the file OUT.C_PBE_REFERENCE_GATE/chgs2.cube , "
                "try to read it.\n"
                "Read electron density from file: "
                "OUT.C_PBE_REFERENCE_GATE/chgs2.cube\n"
            )

            evidence = audit_gate._restart_load_lines(phase)

            self.assertEqual(len(evidence["wfc_load_lines"]), 2)
            self.assertEqual(len(evidence["charge_load_lines"]), 2)
            self.assertTrue(
                all(
                    "Read electron density from file:" in line
                    for line in evidence["charge_load_lines"]
                )
            )

    def test_restart_load_rejects_extra_mixed_format_charge_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase, _ = self._restart_phase(tmp)
            self._write_restart_load_logs(
                phase,
                (
                    "OUT.C_PBE_REFERENCE_GATE/wfs1_nao.txt",
                    "OUT.C_PBE_REFERENCE_GATE/wfs2_nao.txt",
                ),
                (
                    "OUT.C_PBE_REFERENCE_GATE/chgs1.cube",
                    "OUT.C_PBE_REFERENCE_GATE/chgs2.cube",
                ),
            )
            log = phase / "OUT.C_PBE_REFERENCE_GATE/running_scf.log"
            log.write_text(
                log.read_text()
                + "Read electron density from file: "
                "OUT.C_PBE_REFERENCE_GATE/chgs1.cube\n"
            )

            with self.assertRaisesRegex(ValueError, "exactly two charge-density"):
                audit_gate._restart_load_lines(phase)

    def test_restart_load_accepts_absolute_phase_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase, out = self._restart_phase(tmp)
            self._write_restart_load_logs(
                phase,
                (out / "wfs1_nao.txt", out / "wfs2_nao.txt"),
                (out / "chgs1.cube", out / "chgs2.cube"),
            )
            evidence = audit_gate._restart_load_lines(phase)
            self.assertEqual(len(evidence["wfc_load_lines"]), 2)
            self.assertEqual(len(evidence["charge_load_lines"]), 2)

    def test_restart_load_rejects_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            external = base / "external"
            external.mkdir()
            for name in audit_gate.RESTART_FILES:
                (external / name).write_text(f"external {name}\n")
            for label, paths in (
                (
                    "traversal",
                    (
                        "../../external/wfs1_nao.txt",
                        "../../external/wfs2_nao.txt",
                        "../../external/chgs1.cube",
                        "../../external/chgs2.cube",
                    ),
                ),
                (
                    "symlink",
                    (
                        "escape/wfs1_nao.txt",
                        "escape/wfs2_nao.txt",
                        "escape/chgs1.cube",
                        "escape/chgs2.cube",
                    ),
                ),
            ):
                with self.subTest(label=label):
                    phase, _ = self._restart_phase(base / label)
                    if label == "symlink":
                        (phase / "escape").symlink_to(external, target_is_directory=True)
                    self._write_restart_load_logs(phase, paths[:2], paths[2:])
                    with self.assertRaisesRegex(ValueError, "phase-local.*restart path"):
                        audit_gate._restart_load_lines(phase)

    def test_scheduler_record_preserves_raw_scontrol_evidence(self):
        raw, environment = self._scheduler_fixture()
        fields = audit_gate._parse_scontrol_fields(raw)
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
            audit_gate, "_query_scheduler", return_value=(fields, raw)
        ):
            scheduler = audit_gate._scheduler_record("fixed")
        self.assertEqual(scheduler["observed"]["raw_record"], raw)
        self.assertEqual(
            scheduler["observed"]["scontrol_sha256"],
            audit_gate._sha256_bytes(raw.encode("utf-8")),
        )
        self.assertEqual(scheduler["profile"], "df_dcu")
        self.assertEqual(scheduler["time_limit"], "1-00:00:00")
        self.assertEqual(scheduler["over_subscribe"], "NO")

    def test_scheduler_record_accepts_exact_server66_evidence(self):
        raw, environment = self._scheduler_fixture("server66")
        fields = audit_gate._parse_scontrol_fields(raw)
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
            audit_gate, "_query_scheduler", return_value=(fields, raw)
        ):
            scheduler = audit_gate._scheduler_record("fixed")
        self.assertEqual(scheduler["profile"], "server66")
        self.assertEqual(scheduler["partition"], "640")
        self.assertEqual(scheduler["cpus_per_task"], 48)
        self.assertEqual(scheduler["memory_mb"], 180000)
        self.assertEqual(scheduler["time_limit"], "1-00:00:00")
        self.assertEqual(scheduler["over_subscribe"], "OK")
        self.assertEqual(scheduler["observed"]["over_subscribe"], "OK")

    def test_scheduler_record_requires_an_explicit_known_profile(self):
        raw, environment = self._scheduler_fixture()
        fields = audit_gate._parse_scontrol_fields(raw)
        for profile_name in ("", "automatic"):
            with self.subTest(profile=profile_name), mock.patch.dict(
                os.environ,
                dict(environment, C_PBE_GATE_PROFILE=profile_name),
                clear=False,
            ), mock.patch.object(
                audit_gate, "_query_scheduler", return_value=(fields, raw)
            ):
                with self.assertRaisesRegex(ValueError, "C_PBE_GATE_PROFILE"):
                    audit_gate._scheduler_record("fixed")

    def test_scheduler_record_rejects_cross_profile_environment_and_scontrol(self):
        df_raw, df_environment = self._scheduler_fixture("df_dcu")
        server_raw, server_environment = self._scheduler_fixture("server66")
        cases = (
            ("df profile with server environment", df_raw, server_environment),
            ("server profile with df environment", server_raw, df_environment),
            (
                "df profile with server scontrol",
                server_raw,
                dict(df_environment, C_PBE_GATE_PROFILE="df_dcu"),
            ),
            (
                "server profile with df scontrol",
                df_raw,
                dict(server_environment, C_PBE_GATE_PROFILE="server66"),
            ),
        )
        for label, raw, environment in cases:
            with self.subTest(label=label), mock.patch.dict(
                os.environ, environment, clear=False
            ), mock.patch.object(
                audit_gate,
                "_query_scheduler",
                return_value=(audit_gate._parse_scontrol_fields(raw), raw),
            ):
                with self.assertRaisesRegex(ValueError, "Slurm|scontrol|observed"):
                    audit_gate._scheduler_record("fixed")

    def test_scheduler_validator_rejects_cross_profile_record(self):
        raw, environment = self._scheduler_fixture("server66")
        fields = audit_gate._parse_scontrol_fields(raw)
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
            audit_gate, "_query_scheduler", return_value=(fields, raw)
        ):
            scheduler = audit_gate._scheduler_record("fixed")
        scheduler["profile"] = "df_dcu"
        with self.assertRaisesRegex(ValueError, "profile|partition"):
            audit_gate._validate_scheduler_record(scheduler, "fixed")

    def test_scheduler_validator_rejects_tampered_raw_scontrol_evidence(self):
        raw, environment = self._scheduler_fixture()
        fields = audit_gate._parse_scontrol_fields(raw)
        with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
            audit_gate, "_query_scheduler", return_value=(fields, raw)
        ):
            scheduler = audit_gate._scheduler_record("fixed")
        self.assertIsInstance(scheduler["observed"].get("raw_record"), str)
        scheduler["observed"]["raw_record"] += "tampered\n"
        with self.assertRaisesRegex(ValueError, "raw scontrol.*hash"):
            audit_gate._validate_scheduler_record(scheduler, "fixed")


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
        cls.fake_python = cls.bin_dir / "python-c-pbe-gate"
        cls.fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"exec {shlex.quote(sys.executable)} \"$@\"\n"
        )
        cls.fake_mpirun = cls.bin_dir / "mpirun"
        cls.fake_mpirun.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "[[ ${OMP_NUM_THREADS:-} == ${SLURM_CPUS_PER_TASK:-} ]]\n"
            "[[ ${MKL_NUM_THREADS:-} == ${SLURM_CPUS_PER_TASK:-} ]]\n"
            "[[ ${OPENBLAS_NUM_THREADS:-} == ${SLURM_CPUS_PER_TASK:-} ]]\n"
            "[[ $1 == -np && $2 == 1 && $3 == -ppn && $4 == 1 ]]\n"
            "shift 4\n"
            'exec "$@"\n'
        )
        cls.fake_scontrol = cls.bin_dir / "scontrol"
        cls.fake_scontrol.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "[[ $1 == show && $2 == job ]]\n"
            "printf '%s\\n' \"JobId=${SLURM_JOB_ID} "
            "ArrayJobId=${SLURM_ARRAY_JOB_ID} ArrayTaskId=${SLURM_ARRAY_TASK_ID} "
            "Partition=${FAKE_SCONTROL_PARTITION:-${SLURM_JOB_PARTITION}} "
            "NumNodes=${SLURM_JOB_NUM_NODES} NumCPUs=${SLURM_CPUS_PER_TASK} "
            "NumTasks=${SLURM_NTASKS} CPUs/Task=${SLURM_CPUS_PER_TASK} "
            "NtasksPerN:B:S:C=1:0:*:* "
            "MinMemoryNode=${FAKE_SCONTROL_MEMORY:-${SLURM_MEM_PER_NODE}M} "
            "TimeLimit=${FAKE_SCONTROL_TIME_LIMIT:-1-00:00:00} "
            "OverSubscribe=${FAKE_SCONTROL_OVER_SUBSCRIBE}\"\n"
        )
        cls.fake_abacus = cls.bin_dir / "abacus"
        cls.fake_abacus.write_text(cls._fake_abacus_text())
        for path in (
            cls.fake_python,
            cls.fake_mpirun,
            cls.fake_scontrol,
            cls.fake_abacus,
        ):
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        cls.fake_environment = cls.assets / "env_intel.sh"
        cls.environment_marker = cls.base / "environment-sourced.log"
        cls.fake_environment.write_text(
            "#!/usr/bin/env bash\n"
            f"printf 'sourced\\n' >>'{cls.environment_marker}'\n"
            "export OMP_NUM_THREADS=1\n"
            "export MKL_NUM_THREADS=1\n"
            "export OPENBLAS_NUM_THREADS=1\n"
        )

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
        print(
            "Read NAO wave functions from "
            f"OUT.C_PBE_REFERENCE_GATE/wfs{spin}_nao.txt"
        )
        log.append(
            "Read in electron density: "
            f"OUT.C_PBE_REFERENCE_GATE/chgs{spin}.cube"
        )
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
    def _task_environment(
        cls,
        root,
        task,
        *,
        fail_phase=None,
        pseudo=None,
        orbital=None,
        profile_name="df_dcu",
        entrypoint=RUNNER,
        common_runner=COMMON_RUNNER,
        overrides=None,
    ):
        profile = get_resource_profile(profile_name)
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{cls.bin_dir}{os.pathsep}{environment['PATH']}",
                "GATE_ROOT": str(root),
                "ABACUS_ARTIFACT": str(cls.fake_abacus),
                "ABACUS_ENV_SCRIPT": str(cls.fake_environment),
                "PSEUDO_ASSET": str(pseudo or cls.pseudo),
                "ORBITAL_ASSET": str(orbital or cls.orbital),
                "PYTHON_EXE": str(cls.fake_python),
                "C_PBE_GATE_ENTRYPOINT": str(entrypoint),
                "C_PBE_GATE_COMMON_RUNNER": str(common_runner),
                "SLURM_JOB_PARTITION": profile["partition"],
                "SLURM_ARRAY_TASK_ID": str(task),
                "SLURM_ARRAY_TASK_COUNT": "4",
                "SLURM_CPUS_PER_TASK": str(profile["cpus_per_task"]),
                "SLURM_NTASKS": str(profile["ntasks"]),
                "SLURM_JOB_NUM_NODES": str(profile["nodes"]),
                "SLURM_TASKS_PER_NODE": "1",
                "SLURM_MEM_PER_NODE": str(profile["memory_mb"]),
                "SLURM_JOB_ID": str(9100 + task),
                "SLURM_ARRAY_JOB_ID": "9001",
                "FAKE_SCONTROL_OVER_SUBSCRIBE": profile["over_subscribe"],
            }
        )
        if fail_phase is not None:
            environment["FAKE_ABACUS_FAIL_PHASE"] = fail_phase
        if overrides:
            environment.update(overrides)
        return environment

    @classmethod
    def _run_task(
        cls,
        root,
        task,
        *,
        fail_phase=None,
        pseudo=None,
        orbital=None,
        runner=RUNNER,
        profile_name="df_dcu",
        entrypoint=None,
        common_runner=COMMON_RUNNER,
        overrides=None,
    ):
        environment = cls._task_environment(
            root,
            task,
            fail_phase=fail_phase,
            pseudo=pseudo,
            orbital=orbital,
            profile_name=profile_name,
            entrypoint=entrypoint or runner,
            common_runner=common_runner,
            overrides=overrides,
        )
        return subprocess.run(
            ["bash", str(runner)],
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
        self.assertEqual(
            summary["restart_chain_evidence"]["environment_script"][
                "absolute_path"
            ],
            str(self.fake_environment),
        )
        self.assertEqual(
            summary["restart_chain_evidence"]["mpirun"]["absolute_path"],
            str(self.fake_mpirun),
        )
        self.assertEqual(summary["restart_chain_evidence"]["gate_profile"], "df_dcu")
        self.assertEqual(
            summary["restart_chain_evidence"]["resource_profiles"]["absolute_path"],
            str(RESOURCE_PROFILES),
        )
        self.assertEqual(
            summary["restart_chain_evidence"]["entrypoint"]["absolute_path"],
            str(RUNNER),
        )
        self.assertEqual(
            summary["restart_chain_evidence"]["common_runner"]["absolute_path"],
            str(COMMON_RUNNER),
        )
        for name, path in (
            ("python", self.fake_python),
            ("prepare_gate", PREPARE_SOURCE),
            ("audit_gate", AUDIT_SOURCE),
            ("gate_contract", GATE_CONTRACT_SOURCE),
        ):
            self.assertEqual(
                summary["restart_chain_evidence"][name]["absolute_path"],
                str(path),
            )

    def test_spooled_wrappers_use_explicit_immutable_runtime_sources(self):
        for profile_name, entrypoint in (
            ("df_dcu", RUNNER),
            ("server66", SERVER66_RUNNER),
        ):
            with self.subTest(profile=profile_name):
                spool = self.base / (
                    f"spool-{profile_name}-{next(tempfile._get_candidate_names())}"
                )
                spool.mkdir()
                spooled_wrapper = spool / "slurm_script"
                shutil.copy2(entrypoint, spooled_wrapper)
                self.assertFalse((spool / COMMON_RUNNER.name).exists())
                root = self.base / (
                    f"spooled-gate-{profile_name}-"
                    f"{next(tempfile._get_candidate_names())}"
                )
                incoming_profile = (
                    "server66" if profile_name == "df_dcu" else "df_dcu"
                )
                completed = self._run_task(
                    root,
                    0,
                    runner=spooled_wrapper,
                    entrypoint=entrypoint,
                    common_runner=COMMON_RUNNER,
                    profile_name=profile_name,
                    overrides={"C_PBE_GATE_PROFILE": incoming_profile},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                run = json.loads(
                    (root / "runs/fixed/BRANCH_RUN_PROVENANCE.json").read_text()
                )
                self.assertEqual(run["gate_profile"], profile_name)
                self.assertEqual(
                    run["entrypoint"]["absolute_path"], str(entrypoint)
                )
                self.assertEqual(
                    run["common_runner"]["absolute_path"], str(COMMON_RUNNER)
                )

    def test_runtime_environment_is_actually_sourced(self):
        self.assertTrue(self.environment_marker.is_file())
        self.assertGreaterEqual(len(self.environment_marker.read_text().splitlines()), 4)

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
        for manifest in (phase_manifest, branch_manifest):
            self.assertEqual(manifest["gate_profile"], "df_dcu")
            self.assertEqual(
                manifest["resource_profiles"]["absolute_path"],
                str(RESOURCE_PROFILES),
            )
            self.assertEqual(
                manifest["entrypoint"]["absolute_path"], str(RUNNER)
            )
            self.assertEqual(
                manifest["common_runner"]["absolute_path"], str(COMMON_RUNNER)
            )
            for name, path in (
                ("python", self.fake_python),
                ("prepare_gate", PREPARE_SOURCE),
                ("audit_gate", AUDIT_SOURCE),
                ("gate_contract", GATE_CONTRACT_SOURCE),
            ):
                self.assertEqual(manifest[name]["absolute_path"], str(path))
            self.assertEqual(
                manifest["environment_script"]["absolute_path"],
                str(self.fake_environment),
            )
            self.assertEqual(
                manifest["mpirun"]["absolute_path"], str(self.fake_mpirun)
            )
        self.assertEqual(phase_manifest["scheduler"]["partition"], "normal")
        self.assertEqual(phase_manifest["scheduler"]["profile"], "df_dcu")
        self.assertEqual(
            phase_manifest["scheduler"]["time_limit"], "1-00:00:00"
        )
        self.assertEqual(phase_manifest["scheduler"]["over_subscribe"], "NO")
        self.assertEqual(phase_manifest["scheduler"]["cpus_per_task"], 30)
        self.assertEqual(phase_manifest["scheduler"]["ntasks"], 1)
        self.assertEqual(
            phase_manifest["scheduler"]["observed"]["memory_raw"], "110610M"
        )
        self.assertEqual(
            phase_manifest["scheduler"]["observed"]["time_limit_raw"],
            "1-00:00:00",
        )
        self.assertEqual(
            phase_manifest["scheduler"]["observed"]["over_subscribe"],
            "NO",
        )
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

    def test_restart_load_evidence_rejects_external_paths(self):
        root = self.copy_gate()
        phase = root / "runs/dir0/free_restart1"
        stdout = phase / "abacus.stdout"
        stdout.write_text(
            "Read NAO wave functions from /external/unrelated/wfs1_nao.txt\n"
            "Read NAO wave functions from /external/unrelated/wfs2_nao.txt\n"
        )
        log = phase / "OUT.C_PBE_REFERENCE_GATE/running_scf.log"
        lines = [
            line
            for line in log.read_text().splitlines()
            if not line.startswith("Read in electron density:")
        ]
        log.write_text(
            "Read in electron density: /external/unrelated/chgs1.cube\n"
            "Read in electron density: /external/unrelated/chgs2.cube\n"
            + "\n".join(lines)
            + "\n"
        )
        with self.assertRaisesRegex(ValueError, "phase-local.*restart path"):
            audit_gate._restart_load_lines(phase)

    def test_restart_snapshot_directory_must_not_be_a_symlink(self):
        root = self.copy_gate()
        phase = root / "runs/dir0/free_restart1"
        snapshot = phase / "restart_input_snapshot"
        shutil.rmtree(snapshot)
        snapshot.symlink_to(
            phase.parent / "field_seed/OUT.C_PBE_REFERENCE_GATE",
            target_is_directory=True,
        )
        with self.assertRaisesRegex(ValueError, "snapshot.*non-symlink.*directory"):
            audit_gate._verify_restart_provenance(
                root, "dir0", "free_restart1", require_verified=True
            )

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
            "observed resource": lambda root: self._mutate_json(
                root / "runs/dir1/free_restart2/PHASE_COMPLETE.json",
                lambda data: data["scheduler"]["observed"].update(
                    {"memory_raw": None}
                ),
            ),
            "runtime profile": lambda root: self._mutate_json(
                root / "runs/dir1/free_restart2/PHASE_COMPLETE.json",
                lambda data: data.update({"gate_profile": "server66"}),
            ),
            "entrypoint hash": lambda root: self._mutate_json(
                root / "runs/dir1/free_restart2/PHASE_COMPLETE.json",
                lambda data: data["entrypoint"].update({"sha256": "0" * 64}),
            ),
            "common runner hash": lambda root: self._mutate_json(
                root / "runs/dir1/BRANCH_COMPLETE.json",
                lambda data: data["common_runner"].update({"sha256": "0" * 64}),
            ),
            "python hash": lambda root: self._mutate_json(
                root / "runs/dir1/free_restart2/PHASE_COMPLETE.json",
                lambda data: data["python"].update({"sha256": "0" * 64}),
            ),
            "prepare source hash": lambda root: self._mutate_json(
                root / "runs/dir1/BRANCH_COMPLETE.json",
                lambda data: data["prepare_gate"].update({"sha256": "0" * 64}),
            ),
            "audit source hash": lambda root: self._mutate_json(
                root / "runs/dir1/free_restart2/PHASE_COMPLETE.json",
                lambda data: data["audit_gate"].update({"sha256": "0" * 64}),
            ),
            "contract source hash": lambda root: self._mutate_json(
                root / "runs/dir1/BRANCH_COMPLETE.json",
                lambda data: data["gate_contract"].update({"sha256": "0" * 64}),
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

    def test_missing_runtime_environment_stops_before_prepare(self):
        root = self.base / f"missing-env-{next(tempfile._get_candidate_names())}"
        completed = self._run_task(
            root, 0, overrides={"ABACUS_ENV_SCRIPT": ""}
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse((root / "runs/fixed").exists())

    def test_audit_rejects_tampered_runtime_environment_script(self):
        original = self.fake_environment.read_bytes()
        try:
            self.fake_environment.write_bytes(original + b"# tampered\n")
            self.assert_not_passed(self.completed_gate)
        finally:
            self.fake_environment.write_bytes(original)

    def test_audit_rejects_tampered_mpirun(self):
        original = self.fake_mpirun.read_bytes()
        mode = self.fake_mpirun.stat().st_mode
        try:
            self.fake_mpirun.write_bytes(original + b"# tampered\n")
            self.fake_mpirun.chmod(mode)
            self.assert_not_passed(self.completed_gate)
        finally:
            self.fake_mpirun.write_bytes(original)
            self.fake_mpirun.chmod(mode)

    def test_audit_rehashes_every_recorded_runtime_source(self):
        for path in (
            self.fake_python,
            PREPARE_SOURCE,
            AUDIT_SOURCE,
            GATE_CONTRACT_SOURCE,
            RESOURCE_PROFILES,
            RUNNER,
            COMMON_RUNNER,
        ):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"# tampered\n")
                    self.assert_not_passed(self.completed_gate)
                finally:
                    path.write_bytes(original)

    def test_global_audit_rejects_mpirun_mismatch_across_branches(self):
        alternate_bin = self.base / f"mpi-{next(tempfile._get_candidate_names())}"
        alternate_bin.mkdir()
        alternate_mpirun = alternate_bin / "mpirun"
        alternate_mpirun.write_text(self.fake_mpirun.read_text() + "# alternate\n")
        alternate_mpirun.chmod(
            alternate_mpirun.stat().st_mode | stat.S_IXUSR
        )
        alternate_root = self.base / (
            f"alternate-mpi-{next(tempfile._get_candidate_names())}"
        )
        completed = self._run_task(
            alternate_root,
            2,
            overrides={
                "PATH": (
                    f"{alternate_bin}{os.pathsep}{self.bin_dir}"
                    f"{os.pathsep}{os.environ['PATH']}"
                )
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        baseline_run = json.loads(
            (
                self.completed_gate
                / "runs/dir1/BRANCH_RUN_PROVENANCE.json"
            ).read_text()
        )
        alternate_run = json.loads(
            (
                alternate_root / "runs/dir1/BRANCH_RUN_PROVENANCE.json"
            ).read_text()
        )
        self.assertEqual(
            baseline_run["environment_script"],
            alternate_run["environment_script"],
        )
        self.assertNotEqual(baseline_run["mpirun"], alternate_run["mpirun"])
        root = self.copy_gate()
        shutil.rmtree(root / "runs/dir1")
        shutil.copytree(alternate_root / "runs/dir1", root / "runs/dir1")
        self.assert_not_passed(root)

    def test_global_audit_rejects_profile_and_entrypoint_mismatch_across_branches(self):
        alternate_root = self.base / (
            f"server66-profile-{next(tempfile._get_candidate_names())}"
        )
        completed = self._run_task(
            alternate_root,
            2,
            runner=SERVER66_RUNNER,
            profile_name="server66",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        alternate_run = json.loads(
            (
                alternate_root / "runs/dir1/BRANCH_RUN_PROVENANCE.json"
            ).read_text()
        )
        self.assertEqual(alternate_run["gate_profile"], "server66")
        self.assertEqual(
            alternate_run["entrypoint"]["absolute_path"], str(SERVER66_RUNNER)
        )
        self.assertEqual(
            alternate_run["common_runner"]["absolute_path"], str(COMMON_RUNNER)
        )
        root = self.copy_gate()
        shutil.rmtree(root / "runs/dir1")
        shutil.copytree(alternate_root / "runs/dir1", root / "runs/dir1")
        with self.assertRaisesRegex(ValueError, "profile|entrypoint"):
            audit_gate.audit_gate(root)

    def test_global_audit_rejects_new_runtime_source_mismatch_across_branches(self):
        source_paths = {
            "python": self.fake_python,
            "prepare_gate": PREPARE_SOURCE,
            "audit_gate": AUDIT_SOURCE,
            "gate_contract": GATE_CONTRACT_SOURCE,
        }
        for field, source in source_paths.items():
            with self.subTest(field=field):
                root = self.copy_gate()
                alternate = self.base / (
                    f"alternate-{field}-{next(tempfile._get_candidate_names())}"
                )
                shutil.copy2(source, alternate)
                _, record = audit_gate._canonical_file(
                    alternate,
                    f"alternate {field}",
                    executable=field == "python",
                )
                branch_root = root / "runs/dir1"
                run_path = branch_root / "BRANCH_RUN_PROVENANCE.json"
                self._mutate_json(
                    run_path, lambda data: data.update({field: record})
                )
                phases = audit_gate.BRANCH_PHASES["dir1"]
                phase_hashes = {}
                restart_hashes = {}
                for index, phase in enumerate(phases):
                    phase_path = branch_root / phase / "PHASE_COMPLETE.json"
                    phase_update = {field: record}
                    if index:
                        restart_path = (
                            branch_root / phase / "RESTART_PROVENANCE.json"
                        )
                        self._mutate_json(
                            restart_path,
                            lambda data, source=phases[index - 1]: data.update(
                                {
                                    "source_phase_complete_sha256": phase_hashes[
                                        source
                                    ]
                                }
                            ),
                        )
                        restart_hashes[phase] = audit_gate._sha256_bytes(
                            restart_path.read_bytes()
                        )
                        phase_update["restart_provenance_sha256"] = (
                            restart_hashes[phase]
                        )
                    self._mutate_json(
                        phase_path,
                        lambda data, update=phase_update: data.update(update),
                    )
                    phase_hashes[phase] = audit_gate._sha256_bytes(
                        phase_path.read_bytes()
                    )
                self._mutate_json(
                    branch_root / "BRANCH_COMPLETE.json",
                    lambda data: data.update(
                        {
                            field: record,
                            "phase_complete_sha256": phase_hashes,
                            "restart_provenance_sha256": restart_hashes,
                            "branch_run_provenance_sha256": (
                                audit_gate._sha256_bytes(run_path.read_bytes())
                            ),
                        }
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError, f"{field} provenance differs across branches"
                ):
                    audit_gate.audit_gate(root)

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

    def test_run_failed_alone_blocks_global_audit(self):
        root = self.copy_gate()
        for branch, phases in audit_gate.BRANCH_PHASES.items():
            branch_root = root / "runs" / branch
            for name in ("BRANCH_RUN_PROVENANCE.json", "BRANCH_COMPLETE.json"):
                (branch_root / name).unlink()
            for phase in phases:
                (branch_root / phase / "PHASE_COMPLETE.json").unlink()
                restart = branch_root / phase / "RESTART_PROVENANCE.json"
                try:
                    restart.unlink()
                except FileNotFoundError:
                    pass
        (root / "runs/fixed/RUN_FAILED.json").write_text(
            '{"status":"RUN_FAILED"}\n'
        )
        with self.assertRaisesRegex(ValueError, "RUN_FAILED"):
            audit_gate.audit_gate(root)

    def test_four_array_tasks_prepare_concurrently_without_guard_race(self):
        root = self.base / f"concurrent-{next(tempfile._get_candidate_names())}"
        processes = [
            subprocess.Popen(
                ["bash", str(RUNNER)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._task_environment(root, task),
            )
            for task in range(4)
        ]
        results = [process.communicate(timeout=60) for process in processes]
        failures = [
            (process.returncode, stdout, stderr)
            for process, (stdout, stderr) in zip(processes, results)
            if process.returncode != 0
        ]
        self.assertEqual(failures, [])
        self.assertEqual(audit_gate.audit_gate(root)["status"], "PBE_GATE_PASSED")

    def test_global_audit_rejects_different_asset_content_across_branches(self):
        alternate_assets = self.base / f"assets-{next(tempfile._get_candidate_names())}"
        alternate_assets.mkdir()
        alternate_pseudo = alternate_assets / self.pseudo.name
        alternate_pseudo.write_text("different fake pseudo\n")
        alternate_root = self.base / f"alternate-{next(tempfile._get_candidate_names())}"
        completed = self._run_task(
            alternate_root,
            2,
            pseudo=alternate_pseudo,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        root = self.copy_gate()
        shutil.rmtree(root / "runs/dir1")
        shutil.copytree(alternate_root / "runs/dir1", root / "runs/dir1")
        self.assert_not_passed(root)

    def test_global_audit_rejects_frozen_preparation_protocol_drift(self):
        root = self.copy_gate()
        branch_root = root / "runs/dir1"
        provenance_path = branch_root / "BRANCH_PROVENANCE.json"
        self._mutate_json(
            provenance_path,
            lambda data: data["frozen_protocol"].update({"ecutwfc": 31.0}),
        )
        run_path = branch_root / "BRANCH_RUN_PROVENANCE.json"
        self._mutate_json(
            run_path,
            lambda data: data.update(
                {
                    "preparation_provenance_sha256": audit_gate._sha256_bytes(
                        provenance_path.read_bytes()
                    )
                }
            ),
        )
        self._mutate_json(
            branch_root / "BRANCH_COMPLETE.json",
            lambda data: data.update(
                {
                    "branch_run_provenance_sha256": audit_gate._sha256_bytes(
                        run_path.read_bytes()
                    )
                }
            ),
        )
        self.assert_not_passed(root)

    def test_asset_names_must_be_safe_basenames(self):
        provenance = {
            "sources": {
                "pseudo": {"basename": "../outside.upf"},
                "orbital": {"basename": "C.orb"},
            }
        }
        with self.assertRaisesRegex(ValueError, "safe.*basename"):
            audit_gate._asset_names(provenance)

    def test_scheduler_validator_rejects_malformed_observed_values(self):
        path = self.completed_gate / "runs/fixed/BRANCH_RUN_PROVENANCE.json"
        scheduler = json.loads(path.read_text())["scheduler"]
        scheduler["observed"]["memory_raw"] = None
        with self.assertRaisesRegex(ValueError, "scheduler observed evidence"):
            audit_gate._validate_scheduler_record(scheduler, "fixed")

    def test_wrong_live_resource_contract_stops_before_prepare(self):
        cases = (
            (
                "cpu",
                {"SLURM_CPUS_PER_TASK": "31"},
                "cpus per task must be 30",
            ),
            (
                "memory",
                {"SLURM_MEM_PER_NODE": "126500"},
                "memory per node must be 110610 MB",
            ),
        )
        for label, overrides, expected_error in cases:
            with self.subTest(label=label):
                root = self.base / (
                    f"wrong-{label}-{next(tempfile._get_candidate_names())}"
                )
                environment = self._task_environment(root, 0, overrides=overrides)
                completed = subprocess.run(
                    ["bash", str(RUNNER)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stderr)
                self.assertFalse((root / "runs/fixed").exists())

    def test_missing_or_nonnumeric_slurm_job_identity_stops_before_prepare(self):
        for label, overrides in (
            ("missing job", {"SLURM_JOB_ID": ""}),
            ("nonnumeric job", {"SLURM_JOB_ID": "UNKNOWN"}),
            ("missing array", {"SLURM_ARRAY_JOB_ID": ""}),
            ("nonnumeric array", {"SLURM_ARRAY_JOB_ID": "UNKNOWN"}),
        ):
            with self.subTest(label=label):
                root = self.base / f"job-id-{next(tempfile._get_candidate_names())}"
                completed = self._run_task(root, 0, overrides=overrides)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse((root / "runs/fixed").exists())

    def test_observed_scheduler_resource_mismatch_stops_before_prepare(self):
        cases = (
            ("memory", {"FAKE_SCONTROL_MEMORY": "120000M"}),
            ("time", {"FAKE_SCONTROL_TIME_LIMIT": "12:00:00"}),
            ("exclusive", {"FAKE_SCONTROL_OVER_SUBSCRIBE": "OK"}),
        )
        for label, overrides in cases:
            with self.subTest(label=label):
                root = self.base / f"scontrol-{next(tempfile._get_candidate_names())}"
                completed = self._run_task(root, 0, overrides=overrides)
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
            "PATH": f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}",
            "C_PBE_GATE_PROFILE": "df_dcu",
            "SLURM_JOB_PARTITION": "normal",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_TASK_COUNT": "4",
            "SLURM_CPUS_PER_TASK": "30",
            "SLURM_NTASKS": "1",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_TASKS_PER_NODE": "1",
            "SLURM_MEM_PER_NODE": "110610",
            "SLURM_JOB_ID": "9003",
            "SLURM_ARRAY_JOB_ID": "9003",
            "FAKE_SCONTROL_OVER_SUBSCRIBE": "NO",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            audit_gate.initialize_branch_run(
                root,
                "fixed",
                "df_dcu",
                self.fake_python,
                PREPARE_SOURCE,
                AUDIT_SOURCE,
                GATE_CONTRACT_SOURCE,
                RESOURCE_PROFILES,
                RUNNER,
                COMMON_RUNNER,
                self.fake_abacus,
                self.fake_environment,
                self.fake_mpirun,
            )
        stru = branch / "fixed_cold/STRU"
        stru.write_text(
            stru.read_text().replace("37.79452249150619", "18.89726124575309")
        )
        with self.assertRaisesRegex(ValueError, "preparation provenance"):
            audit_gate.preflight_phase(root, "fixed", "fixed_cold")


class SubmissionContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.bin_dir = self.base / "bin"
        self.bin_dir.mkdir()
        self.scheduler_log = self.base / "scheduler.log"
        self.sbatch_count = self.base / "sbatch.count"
        self.batch_environment = self.base / "batch-environment.txt"
        self.tamper_marker = self.base / "source-tampered"
        self._write_fake_command(
            "squeue",
            r'''#!/usr/bin/env bash
set -euo pipefail
printf 'squeue %s\n' "$*" >>"$FAKE_SCHEDULER_LOG"
[[ ${FAKE_SQUEUE_FAIL:-0} == 0 ]] || exit 31
printf '%s' "${FAKE_SQUEUE_OUTPUT:-}"
''',
        )
        self._write_fake_command(
            "sacct",
            r'''#!/usr/bin/env bash
set -euo pipefail
printf 'sacct %s\n' "$*" >>"$FAKE_SCHEDULER_LOG"
[[ ${FAKE_SACCT_FAIL:-0} == 0 ]] || exit 32
if [[ ${FAKE_CREATE_BRANCH_AFTER_CLAIM:-0} == 1 \
      && -d $GATE_ROOT/.submission-claim ]]; then
    mkdir -p "$GATE_ROOT/runs/fixed"
    printf 'injected after claim\n' >"$GATE_ROOT/runs/fixed/BRANCH_PROVENANCE.json"
fi
if [[ -n ${FAKE_TAMPER_SOURCE:-} && ! -e $FAKE_TAMPER_MARKER ]]; then
    replacement="${FAKE_TAMPER_SOURCE}.replacement.$$"
    cp -p "$FAKE_TAMPER_SOURCE" "$replacement"
    printf '\n# replaced after submitter resolution\n' >>"$replacement"
    mv -f "$replacement" "$FAKE_TAMPER_SOURCE"
    : >"$FAKE_TAMPER_MARKER"
fi
printf '%s' "${FAKE_SACCT_OUTPUT:-}"
''',
        )
        self._write_fake_command(
            "sbatch",
            r'''#!/usr/bin/env bash
set -euo pipefail
printf 'sbatch %s\n' "$*" >>"$FAKE_SCHEDULER_LOG"
export_map=
for argument in "$@"; do
    case "$argument" in
        --export=*) export_map=${argument#--export=} ;;
    esac
done
[[ -n $export_map ]] || exit 42
IFS=',' read -r -a exported_environment <<<"$export_map"
env -i "${exported_environment[@]}" \
    SLURM_JOB_ID=4242 \
    SLURM_ARRAY_JOB_ID=4242 \
    SLURM_ARRAY_TASK_ID=0 \
    /usr/bin/env >"$FAKE_BATCH_ENVIRONMENT"
if [[ ${FAKE_REQUIRE_DURABLE_RECEIPT:-0} == 1 ]]; then
    claim="$GATE_ROOT/.submission-claim"
    if [[ ! -f $claim/SBATCH_RECEIPT.txt || -L $claim/SBATCH_RECEIPT.txt \
          || -s $claim/SBATCH_RECEIPT.txt \
          || ! -f $claim/SBATCH_STDERR.txt || -L $claim/SBATCH_STDERR.txt \
          || -s $claim/SBATCH_STDERR.txt \
          || ! -f $claim/RECEIPT_FILES_PREPARED.json \
          || -L $claim/RECEIPT_FILES_PREPARED.json ]]; then
        exit 41
    fi
fi
count=0
[[ ! -f $FAKE_SBATCH_COUNT ]] || count=$(<"$FAKE_SBATCH_COUNT")
printf '%s\n' "$((count + 1))" >"$FAKE_SBATCH_COUNT"
printf '%s\n' "${FAKE_SBATCH_OUTPUT:-4242}"
exit "${FAKE_SBATCH_EXIT:-0}"
''',
        )
        self.assets = self.base / "assets"
        self.assets.mkdir()
        self.abacus = self.assets / "abacus"
        self.pseudo = self.assets / "C_ONCV_PBE-1.0.upf"
        self.orbital = self.assets / "C_gga_10au_100Ry_3s3p2d.orb"
        self.environment_script = self.assets / "env_intel.sh"
        self.python_real = self.assets / "python-c-pbe-gate"
        self.python_exe = self.assets / "python3"
        self.abacus.write_text("#!/usr/bin/env bash\nexit 0\n")
        self.abacus.chmod(self.abacus.stat().st_mode | stat.S_IXUSR)
        self.pseudo.write_text("pseudo\n")
        self.orbital.write_text("orbital\n")
        self.environment_script.write_text("#!/usr/bin/env bash\n")
        self.python_real.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"exec {shlex.quote(sys.executable)} \"$@\"\n"
        )
        self.python_real.chmod(self.python_real.stat().st_mode | stat.S_IXUSR)
        self.python_exe.symlink_to(self.python_real.name)
        self.gate_root = self.base / "gate"
        self.source_commit = "0123456789abcdef0123456789abcdef01234567"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_fake_command(self, name, content):
        path = self.bin_dir / name
        path.write_text(textwrap.dedent(content))
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def _environment(self, **overrides):
        environment = os.environ.copy()
        unrelated_bin = self.base / "unrelated-application/bin"
        environment.update(
            {
                "PATH": (
                    f"{self.bin_dir}{os.pathsep}{unrelated_bin}"
                    f"{os.pathsep}{environment['PATH']}"
                ),
                "GATE_PROFILE": "df_dcu",
                "GATE_ROOT": str(self.gate_root),
                "ABACUS_ARTIFACT": str(self.abacus),
                "ABACUS_ENV_SCRIPT": str(self.environment_script),
                "PSEUDO_SOURCE": str(self.pseudo),
                "ORBITAL_SOURCE": str(self.orbital),
                "PYTHON_EXE": str(self.python_exe),
                "SOURCE_COMMIT": self.source_commit,
                "FAKE_SCHEDULER_LOG": str(self.scheduler_log),
                "FAKE_SBATCH_COUNT": str(self.sbatch_count),
                "FAKE_BATCH_ENVIRONMENT": str(self.batch_environment),
                "FAKE_TAMPER_MARKER": str(self.tamper_marker),
                "OPENAI_API_KEY": "fake-api-key-must-not-enter-batch",
                "GITHUB_TOKEN": "fake-token-must-not-enter-batch",
                "SSH_AUTH_SOCK": "/tmp/fake-agent-socket-must-not-enter-batch",
                "SSH_AGENT_PID": "987654",
                "CONDA_PREFIX": "/opt/fake-conda-must-not-enter-batch",
                "CONDA_DEFAULT_ENV": "fake-conda-environment",
                "UNRELATED_APPLICATION_HOME": (
                    "/opt/unrelated-application-must-not-enter-batch"
                ),
                "C_PBE_GATE_PROFILE": "ambient-profile-must-not-win",
                "HOSTNAME": "server66-host-hint-must-not-select-profile",
                "SLURM_CLUSTER_NAME": "server66-cluster-hint-must-not-select-profile",
            }
        )
        environment.update({key: str(value) for key, value in overrides.items()})
        return environment

    def _run_submitter(self, script=SUBMITTER, **overrides):
        return subprocess.run(
            ["bash", str(script)],
            check=False,
            capture_output=True,
            text=True,
            env=self._environment(**overrides),
        )

    def _submission_count(self):
        if not self.sbatch_count.exists():
            return 0
        return int(self.sbatch_count.read_text().strip())

    def _recorded_batch_environment(self):
        return dict(
            line.split("=", 1)
            for line in self.batch_environment.read_text().splitlines()
        )

    def test_submitter_static_contract(self):
        text = SUBMITTER.read_text()
        for value in (
            "GATE_ROOT",
            "ABACUS_ARTIFACT",
            "ABACUS_ENV_SCRIPT",
            "PSEUDO_SOURCE",
            "ORBITAL_SOURCE",
            "PYTHON_EXE",
            "SOURCE_COMMIT",
            "GATE_PROFILE",
            "SUBMITTED_JOB_ID.txt",
            "SUBMISSION_PROVENANCE.json",
            "squeue",
            "sacct",
            "sbatch",
            "--array=0-3",
            "PSEUDO_ASSET",
            "ORBITAL_ASSET",
            "run_pbe_branch.slurm",
            "run_pbe_branch_server66.slurm",
            "run_pbe_branch_common.sh",
            "resource_profiles.py",
            "C_PBE_GATE_PROFILE",
            "C_PBE_GATE_ENTRYPOINT",
            "C_PBE_GATE_COMMON_RUNNER",
        ):
            self.assertIn(value, text)
        self.assertNotIn("--dependency", text)
        self.assertNotIn("git -C", text)
        self.assertNotIn("hostname", text.lower())
        self.assertNotIn('EXPORT_MAP="ALL,', text)

    def test_readme_documents_physical_and_operational_gate(self):
        text = README.read_text()
        for value in (
            "20 Angstrom",
            "neutral carbon triplet",
            "fixed integer occupation",
            "weak-field",
            "zero-field free restart",
            "normal",
            "30 OpenMP threads",
            "110610 MB",
            "24 hours",
            "canonical absolute paths",
            "SOURCE_COMMIT",
            "standalone source archive",
            "does not require `.git`",
            "login node",
            "DIAGNOSTIC_ONLY",
            "PBE_GATE_PASSED",
            "Delta-ST",
            "ABACUS_ENV_SCRIPT",
            "Intel MPI/MKL",
        ):
            self.assertIn(value, text)

    def test_readme_documents_cluster_profiles_and_migration_contract(self):
        text = README.read_text()
        for value in (
            "df_dcu",
            "server66",
            "normal",
            "1 node",
            "1 MPI rank",
            "30 OpenMP threads",
            "110610 MB",
            "OverSubscribe=NO",
            "640",
            "48 OpenMP threads",
            "180000 MB",
            "OverSubscribe=OK",
            "27722d5e3e5cf2c94d00ac9489152b7ea00adcf51a8b8bb3a8eed3d8d094c279",
            "e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99",
            "7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d",
            "server66_runtime_env.sh",
            "/etc/profile.d/modules.sh",
            "module purge",
            "module load gcc10.2",
            "module load intel20u4",
            "full 48-CPU and 180000-MB allocation",
            "exact allowlist",
            "must not use `ALL`",
            "21709225",
            "server66 preflight",
            "git archive",
            "SOURCE_COMMIT.txt",
            "SOURCE_ARCHIVE.sha256",
            "Ran 171 tests",
            "BASH_SYNTAX.txt",
            "PY_COMPILE.txt",
            "SBATCH_TEST_ONLY.txt",
            "PREFLIGHT_EVIDENCE.sha256",
            "PREFLIGHT_PASSED",
            "sbatch --test-only",
            "GATE_PROFILE=df_dcu \"$GATE_DIR/submit_pbe_gate.sh\"",
            "GATE_PROFILE=server66 \"$GATE_DIR/submit_pbe_gate.sh\"",
            "scheduler completion is not a physical pass",
            "global audit",
            "PBE_GATE_PASSED",
        ):
            self.assertIn(value, text)

        self.assertIn("24 hours", text)
        self.assertRegex(
            text,
            r"21709225[^\n]*only after[^\n]*server66 preflight[^\n]*pass",
        )
        self.assertIn(
            "If any preflight command fails, leave df_dcu job `21709225` untouched.",
            text,
        )
        self.assertRegex(
            text,
            r"(?s)`PREFLIGHT_PASSED`.*before running `scancel 21709225`",
        )
        self.assertRegex(
            text,
            r"only the login-node global audit\s+can create `PBE_GATE_PASSED`",
        )

    def test_readme_profile_examples_are_executable_and_bound_to_artifacts(self):
        text = README.read_text()
        blocks = {}
        for profile in ("df_dcu", "server66"):
            marker = f"For {profile},"
            self.assertIn(marker, text)
            section = text.split(marker, 1)[1]
            match = re.search(
                r"```bash\n(?P<block>.*?)\n```",
                section,
                re.DOTALL,
            )
            self.assertIsNotNone(match, f"missing {profile} submission code block")
            block = match.group("block")
            blocks[profile] = block
            syntax = subprocess.run(
                ["bash", "-n"],
                input=block,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            self.assertNotRegex(block, r"<[^>\n]+>")
            self.assertNotRegex(block, r"(?m)^export SOURCE_COMMIT=")
            self.assertIn(
                ': "${SOURCE_COMMIT:?SOURCE_COMMIT must be exported by the staging step}"',
                block,
            )
            self.assertIn(
                '[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {',
                block,
            )
            self.assertIn("export SOURCE_COMMIT", block)
            self.assertIn(
                f'GATE_PROFILE={profile} "$GATE_DIR/submit_pbe_gate.sh"',
                block,
            )

        self.assertIn(
            'export GATE_ROOT="/work1/ghj/c-atom-pbe-equivalence-${SOURCE_COMMIT:0:12}"',
            blocks["df_dcu"],
        )
        self.assertIn(
            'export GATE_ROOT="/home/ghj/abacus/260822/'
            'c-atom-pbe-equivalence-server66-${SOURCE_COMMIT:0:12}"',
            blocks["server66"],
        )
        for line in (
            "export ABACUS_ARTIFACT=/home/ghj/abacus/260809/"
            "sternheimer-solid-delta/artifacts/abacus-407979/abacus",
            'export ABACUS_ENV_SCRIPT="$GATE_DIR/server66_runtime_env.sh"',
            'export PSEUDO_SOURCE="$GATE_ROOT/assets/C_ONCV_PBE-1.0.upf"',
            'export ORBITAL_SOURCE="$GATE_ROOT/assets/'
            'C_gga_10au_100Ry_3s3p2d.orb"',
            "export PYTHON_EXE=/home/ghj/app/miniconda3/bin/python3",
        ):
            self.assertIn(line, blocks["server66"])

        self.assertRegex(
            text,
            r"\| `/home/ghj/abacus/260809/sternheimer-solid-delta/artifacts/"
            r"abacus-407979/abacus` \| "
            r"`27722d5e3e5cf2c94d00ac9489152b7ea00adcf51a8b8bb3a8eed3d8d094c279` \|",
        )
        for source in (
            "gate_contract.py",
            "prepare_gate.py",
            "audit_gate.py",
            "resource_profiles.py",
            "selected profile-specific entrypoint",
            "run_pbe_branch_common.sh",
            "submit_pbe_gate.sh",
        ):
            self.assertIn(source, text)
        self.assertIn(
            "The submitter records the declared `SOURCE_COMMIT`; the staging and "
            "preflight evidence proves its association with the extracted archive.",
            text,
        )

    def _assert_profile_submission(self, profile_name, entrypoint):
        completed = self._run_submitter(GATE_PROFILE=profile_name)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self._submission_count(), 1)
        self.assertEqual(
            (self.gate_root / "SUBMITTED_JOB_ID.txt").read_text(), "4242\n"
        )
        provenance = json.loads(
            (self.gate_root / "SUBMISSION_PROVENANCE.json").read_text()
        )
        self.assertEqual(provenance["status"], "SUBMITTED")
        self.assertEqual(provenance["job_id"], "4242")
        self.assertEqual(provenance["source_commit"], self.source_commit)
        self.assertEqual(provenance["gate_profile"], profile_name)

        python = str(self.python_real)
        receipt = self.gate_root / ".submission-claim/SBATCH_RECEIPT.txt"
        resolved_paths = {
            "gate_root": self.gate_root,
            "resource_profiles": RESOURCE_PROFILES,
            "entrypoint": entrypoint,
            "common_runner": COMMON_RUNNER,
            "python_exe": python,
            "gate_contract": GATE_CONTRACT_SOURCE,
            "prepare_gate": PREPARE_SOURCE,
            "audit_gate": AUDIT_SOURCE,
            "submitter": SUBMITTER,
            "abacus_artifact": self.abacus,
            "abacus_env_script": self.environment_script,
            "pseudo_source": self.pseudo,
            "orbital_source": self.orbital,
            "receipt": receipt,
        }
        self.assertEqual(
            provenance["resolved_paths"],
            {name: str(path) for name, path in resolved_paths.items()},
        )

        file_paths = {
            "resource_profiles": RESOURCE_PROFILES,
            "entrypoint": entrypoint,
            "common_runner": COMMON_RUNNER,
            "python": python,
            "gate_contract": GATE_CONTRACT_SOURCE,
            "prepare_gate": PREPARE_SOURCE,
            "audit_gate": AUDIT_SOURCE,
            "submitter": SUBMITTER,
            "abacus": self.abacus,
            "abacus_env_script": self.environment_script,
            "pseudo": self.pseudo,
            "orbital": self.orbital,
            "receipt": receipt,
        }
        self.assertEqual(set(provenance["files"]), set(file_paths))
        for name, path in file_paths.items():
            record = provenance["files"][name]
            self.assertEqual(record["path"], str(path))
            self.assertGreater(record["size"], 0)
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")

        exported_environment = {
            "GATE_ROOT": str(self.gate_root),
            "ABACUS_ARTIFACT": str(self.abacus),
            "ABACUS_ENV_SCRIPT": str(self.environment_script),
            "PSEUDO_ASSET": str(self.pseudo),
            "ORBITAL_ASSET": str(self.orbital),
            "PYTHON_EXE": python,
            "C_PBE_GATE_PROFILE": profile_name,
            "C_PBE_GATE_ENTRYPOINT": str(entrypoint),
            "C_PBE_GATE_COMMON_RUNNER": str(COMMON_RUNNER),
        }
        self.assertEqual(provenance["runner_environment"], exported_environment)
        export_map = ",".join(
            f"{name}={value}" for name, value in exported_environment.items()
        )
        self.assertEqual(
            provenance["command"],
            [
                "sbatch",
                "--parsable",
                f"--job-name={provenance['job_name']}",
                "--array=0-3",
                f"--export={export_map}",
                str(entrypoint),
            ],
        )

        log = self.scheduler_log.read_text()
        self.assertEqual(log.count("sbatch "), 1)
        self.assertIn(f"--export={export_map}", log)
        self.assertIn(f" {entrypoint}", log)
        self.assertNotIn("--dependency", log)
        self.assertNotIn("ALL,", log)

        batch_environment = self._recorded_batch_environment()
        scheduler_environment = {
            "SLURM_JOB_ID": "4242",
            "SLURM_ARRAY_JOB_ID": "4242",
            "SLURM_ARRAY_TASK_ID": "0",
        }
        self.assertEqual(
            batch_environment,
            dict(exported_environment, **scheduler_environment),
        )
        sensitive_values = (
            "fake-api-key-must-not-enter-batch",
            "fake-token-must-not-enter-batch",
            "/tmp/fake-agent-socket-must-not-enter-batch",
            "/opt/fake-conda-must-not-enter-batch",
            "fake-conda-environment",
            "/opt/unrelated-application-must-not-enter-batch",
            "ambient-profile-must-not-win",
            "server66-host-hint-must-not-select-profile",
            "server66-cluster-hint-must-not-select-profile",
            str(self.base / "unrelated-application/bin"),
        )
        command_text = "\n".join(provenance["command"])
        batch_text = self.batch_environment.read_text()
        for value in sensitive_values:
            self.assertNotIn(value, command_text)
            self.assertNotIn(value, batch_text)

    def test_df_dcu_submission_records_exact_profile_and_environment(self):
        self._assert_profile_submission("df_dcu", RUNNER)

    def test_server66_submission_records_exact_profile_and_environment(self):
        self._assert_profile_submission("server66", SERVER66_RUNNER)

    def test_missing_or_unknown_profile_stops_before_claim_and_sbatch(self):
        for index, (label, profile_name) in enumerate(
            (("missing", ""), ("unknown", "automatic"))
        ):
            with self.subTest(label=label):
                gate_root = self.base / f"profile-{index}"
                completed = self._run_submitter(
                    GATE_ROOT=gate_root,
                    GATE_PROFILE=profile_name,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("GATE_PROFILE", completed.stderr)
                self.assertFalse((gate_root / ".submission-claim").exists())
        self.assertEqual(self._submission_count(), 0)

    def test_replaced_runtime_source_stops_before_claim_and_sbatch(self):
        for index, (label, profile_name) in enumerate(
            (
                ("python", "df_dcu"),
                ("abacus", "df_dcu"),
                ("abacus_env_script", "df_dcu"),
                ("pseudo", "df_dcu"),
                ("orbital", "df_dcu"),
                ("resource_profiles", "df_dcu"),
                ("gate_contract", "df_dcu"),
                ("prepare_gate", "df_dcu"),
                ("audit_gate", "df_dcu"),
                ("df_dcu_entrypoint", "df_dcu"),
                ("server66_entrypoint", "server66"),
                ("common_runner", "df_dcu"),
                ("submitter", "df_dcu"),
            )
        ):
            with self.subTest(profile=profile_name, source=label):
                archive = self.base / f"tamper-{index}/pbe_reference_gate"
                shutil.copytree(ROOT, archive)
                assets = self.base / f"tamper-{index}/assets"
                shutil.copytree(self.assets, assets, symlinks=True)
                gate_root = self.base / f"tamper-gate-{index}"
                tamper_marker = self.base / f"tamper-marker-{index}"
                scheduler_log = self.base / f"tamper-scheduler-{index}.log"
                sbatch_count = self.base / f"tamper-sbatch-{index}.count"
                batch_environment = self.base / f"tamper-batch-{index}.txt"
                source_paths = {
                    "python": assets / self.python_real.name,
                    "abacus": assets / self.abacus.name,
                    "abacus_env_script": assets / self.environment_script.name,
                    "pseudo": assets / self.pseudo.name,
                    "orbital": assets / self.orbital.name,
                    "resource_profiles": archive / RESOURCE_PROFILES.name,
                    "gate_contract": archive / GATE_CONTRACT_SOURCE.name,
                    "prepare_gate": archive / PREPARE_SOURCE.name,
                    "audit_gate": archive / AUDIT_SOURCE.name,
                    "df_dcu_entrypoint": archive / RUNNER.name,
                    "server66_entrypoint": archive / SERVER66_RUNNER.name,
                    "common_runner": archive / COMMON_RUNNER.name,
                    "submitter": archive / SUBMITTER.name,
                }
                completed = self._run_submitter(
                    script=archive / SUBMITTER.name,
                    GATE_PROFILE=profile_name,
                    GATE_ROOT=gate_root,
                    ABACUS_ARTIFACT=assets / self.abacus.name,
                    ABACUS_ENV_SCRIPT=assets / self.environment_script.name,
                    PSEUDO_SOURCE=assets / self.pseudo.name,
                    ORBITAL_SOURCE=assets / self.orbital.name,
                    PYTHON_EXE=assets / self.python_exe.name,
                    FAKE_SCHEDULER_LOG=scheduler_log,
                    FAKE_SBATCH_COUNT=sbatch_count,
                    FAKE_BATCH_ENVIRONMENT=batch_environment,
                    FAKE_TAMPER_SOURCE=source_paths[label],
                    FAKE_TAMPER_MARKER=tamper_marker,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("changed after resolution", completed.stderr)
                self.assertFalse((gate_root / ".submission-claim").exists())
                self.assertFalse(sbatch_count.exists())

    def test_all_exported_paths_reject_commas_or_newlines_before_claim(self):
        comma_abacus = self.assets / "abacus,invalid"
        shutil.copy2(self.abacus, comma_abacus)
        comma_environment = self.assets / "environment,invalid.sh"
        shutil.copy2(self.environment_script, comma_environment)
        comma_pseudo = self.assets / "pseudo,invalid.upf"
        shutil.copy2(self.pseudo, comma_pseudo)
        newline_orbital = self.assets / "orbital\ninvalid.orb"
        shutil.copy2(self.orbital, newline_orbital)
        comma_python = self.assets / "python,invalid"
        comma_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"exec {shlex.quote(sys.executable)} \"$@\"\n"
        )
        comma_python.chmod(comma_python.stat().st_mode | stat.S_IXUSR)
        comma_archive = self.base / "archive,invalid/pbe_reference_gate"
        shutil.copytree(ROOT, comma_archive)

        cases = (
            ("gate root", SUBMITTER, {"GATE_ROOT": self.base / "gate,invalid"}),
            (
                "executable",
                SUBMITTER,
                {"ABACUS_ARTIFACT": comma_abacus},
            ),
            (
                "environment",
                SUBMITTER,
                {"ABACUS_ENV_SCRIPT": comma_environment},
            ),
            ("pseudo", SUBMITTER, {"PSEUDO_SOURCE": comma_pseudo}),
            ("orbital", SUBMITTER, {"ORBITAL_SOURCE": newline_orbital}),
            ("python", SUBMITTER, {"PYTHON_EXE": comma_python}),
            (
                "entrypoint and common runner",
                comma_archive / SUBMITTER.name,
                {},
            ),
        )
        for index, (label, script, overrides) in enumerate(cases):
            with self.subTest(label=label):
                gate_root = overrides.get(
                    "GATE_ROOT", self.base / f"invalid-export-{index}"
                )
                completed = self._run_submitter(
                    script=script,
                    GATE_ROOT=gate_root,
                    **{
                        name: value
                        for name, value in overrides.items()
                        if name != "GATE_ROOT"
                    },
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("commas or newlines", completed.stderr)
                self.assertFalse((gate_root / ".submission-claim").exists())
        self.assertEqual(self._submission_count(), 0)

    def test_python_and_source_commit_are_required_and_validated(self):
        cases = (
            ("missing Python", {"PYTHON_EXE": ""}),
            ("missing environment", {"ABACUS_ENV_SCRIPT": ""}),
            ("missing commit", {"SOURCE_COMMIT": ""}),
            ("short commit", {"SOURCE_COMMIT": "a" * 39}),
            ("uppercase commit", {"SOURCE_COMMIT": "A" * 40}),
        )
        for index, (label, overrides) in enumerate(cases):
            with self.subTest(label=label):
                completed = self._run_submitter(
                    GATE_ROOT=self.base / f"required-{index}", **overrides
                )
                self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._submission_count(), 0)

    def test_standalone_source_archive_submits_without_parent_git(self):
        archive_parent = self.base / "standalone"
        archive = archive_parent / "pbe_reference_gate"
        shutil.copytree(ROOT, archive)
        self.assertFalse(any((path / ".git").exists() for path in archive.parents))
        completed = self._run_submitter(script=archive / SUBMITTER.name)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self._submission_count(), 1)

    def test_second_invocation_never_submits_again(self):
        first = self._run_submitter()
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self._run_submitter(
            FAKE_SQUEUE_OUTPUT="4242|c_pbe_gate|RUNNING\n",
            FAKE_SACCT_OUTPUT="4242|c_pbe_gate|RUNNING\n",
        )
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(self._submission_count(), 1)
        log = self.scheduler_log.read_text()
        self.assertGreaterEqual(log.count("squeue "), 2)
        self.assertGreaterEqual(log.count("sacct "), 2)

    def test_existing_scheduler_states_block_submission(self):
        for state in ("PENDING", "RUNNING", "COMPLETING", "COMPLETED"):
            with self.subTest(state=state):
                gate = self.base / f"gate-{state.lower()}"
                completed = self._run_submitter(
                    GATE_ROOT=gate,
                    FAKE_SQUEUE_OUTPUT=f"777|c_pbe_gate|{state}\n",
                    FAKE_SACCT_OUTPUT=f"777|c_pbe_gate|{state}\n",
                )
                self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._submission_count(), 0)

    def test_scheduler_query_failure_is_unobservable_and_blocks(self):
        for command in ("FAKE_SQUEUE_FAIL", "FAKE_SACCT_FAIL"):
            with self.subTest(command=command):
                gate = self.base / f"gate-{command.lower()}"
                completed = self._run_submitter(GATE_ROOT=gate, **{command: "1"})
                self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._submission_count(), 0)

    def test_existing_formal_evidence_or_markers_block_submission(self):
        cases = (
            "RESULT_SUMMARY.json",
            "PBE_GATE_PASSED",
            "RUN_FAILED.json",
            "runs/fixed/BRANCH_PROVENANCE.json",
        )
        for index, relative in enumerate(cases):
            with self.subTest(relative=relative):
                gate = self.base / f"evidence-{index}"
                marker = gate / relative
                marker.parent.mkdir(parents=True)
                marker.write_text("evidence\n")
                completed = self._run_submitter(GATE_ROOT=gate)
                self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._submission_count(), 0)

    def test_assets_must_be_canonical_nonempty_regular_files(self):
        empty = self.assets / "empty.upf"
        empty.touch()
        symlink = self.assets / "linked.orb"
        symlink.symlink_to(self.orbital)
        directory = self.assets / "directory"
        directory.mkdir()
        empty_environment = self.assets / "empty-env.sh"
        empty_environment.touch()
        linked_environment = self.assets / "linked-env.sh"
        linked_environment.symlink_to(self.environment_script)
        cases = (
            {"PSEUDO_SOURCE": empty},
            {"ORBITAL_SOURCE": symlink},
            {"ABACUS_ARTIFACT": directory},
            {"ABACUS_ENV_SCRIPT": empty_environment},
            {"ABACUS_ENV_SCRIPT": linked_environment},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(index=index):
                completed = self._run_submitter(
                    GATE_ROOT=self.base / f"bad-asset-{index}", **overrides
                )
                self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._submission_count(), 0)

    def test_concurrent_claim_allows_exactly_one_sbatch(self):
        processes = [
            subprocess.Popen(
                ["bash", str(SUBMITTER)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._environment(),
            )
            for _ in range(4)
        ]
        results = [process.communicate(timeout=30) for process in processes]
        self.assertEqual(sum(process.returncode == 0 for process in processes), 1)
        self.assertEqual(self._submission_count(), 1, results)

    def test_branch_evidence_created_after_claim_blocks_sbatch(self):
        completed = self._run_submitter(FAKE_CREATE_BRANCH_AFTER_CLAIM="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(
            (self.gate_root / "runs/fixed/BRANCH_PROVENANCE.json").is_file()
        )
        self.assertEqual(self._submission_count(), 0)

    def test_receipt_files_are_durable_before_sbatch_begins(self):
        completed = self._run_submitter(FAKE_REQUIRE_DURABLE_RECEIPT="1")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self._submission_count(), 1)

    def test_sbatch_failure_after_job_id_retains_receipt_and_blocks_retry(self):
        first = self._run_submitter(FAKE_SBATCH_EXIT="99")
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual(self._submission_count(), 1)
        receipt = self.gate_root / ".submission-claim/SBATCH_RECEIPT.txt"
        self.assertEqual(receipt.read_text(), "4242\n")
        second = self._run_submitter()
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(self._submission_count(), 1)
        log = self.scheduler_log.read_text()
        self.assertGreaterEqual(log.count("squeue "), 3)
        self.assertGreaterEqual(log.count("sacct "), 3)

    def test_malformed_success_receipt_is_ambiguous_and_blocks_retry(self):
        first = self._run_submitter(FAKE_SBATCH_OUTPUT="submitted maybe")
        self.assertNotEqual(first.returncode, 0)
        self.assertTrue(
            (self.gate_root / ".submission-claim/SUBMISSION_AMBIGUOUS.json").is_file()
        )
        second = self._run_submitter()
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(self._submission_count(), 1)

    def test_submitted_job_id_marker_must_be_local_regular_file(self):
        self.gate_root.mkdir()
        external = self.base / "external-job-id.txt"
        external.write_text("4242\n")
        (self.gate_root / "SUBMITTED_JOB_ID.txt").symlink_to(external)
        completed = self._run_submitter()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("non-symlink regular file", completed.stderr)
        self.assertEqual(self._submission_count(), 0)

    def test_runtime_source_files_must_be_nonempty_local_regular_files(self):
        for index, mutation in enumerate(("empty", "symlink")):
            with self.subTest(mutation=mutation):
                archive = self.base / f"runtime-{index}/pbe_reference_gate"
                shutil.copytree(ROOT, archive)
                git_environment = os.environ.copy()
                git_environment.update(
                    {
                        "GIT_AUTHOR_NAME": "Test",
                        "GIT_AUTHOR_EMAIL": "test@example.invalid",
                        "GIT_COMMITTER_NAME": "Test",
                        "GIT_COMMITTER_EMAIL": "test@example.invalid",
                    }
                )
                subprocess.run(
                    ["git", "init", "-q"], cwd=archive.parent, check=True
                )
                subprocess.run(
                    ["git", "commit", "--allow-empty", "-q", "-m", "fixture"],
                    cwd=archive.parent,
                    env=git_environment,
                    check=True,
                )
                runtime = archive / "audit_gate.py"
                runtime.unlink()
                if mutation == "empty":
                    runtime.touch()
                else:
                    runtime.symlink_to(ROOT / "audit_gate.py")
                completed = self._run_submitter(
                    script=archive / SUBMITTER.name,
                    GATE_ROOT=self.base / f"runtime-gate-{index}",
                )
                self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(self._submission_count(), 0)


if __name__ == "__main__":
    unittest.main()
