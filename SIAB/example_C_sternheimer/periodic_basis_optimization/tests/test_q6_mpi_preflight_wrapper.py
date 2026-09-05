import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


WRAPPER = Path(__file__).resolve().parents[1] / 'galerkin_binding_workflow/run_c_q6_node_failure_preflight_dfdcu.slurm'


class Q6PreflightWrapperTest(unittest.TestCase):
    def fixture(self, root, probe_ok=True, parent_state='NODE_FAIL'):
        source = root / 'source'
        source.mkdir()
        binaries = root / 'bin'
        binaries.mkdir()
        probe = source / 'c_mpi_communication_preflight.c'
        probe.write_text('frozen communication probe fixture\n')
        runner = source / 'run_c_solid_fd8_q13_standard_dfdcu.slurm'
        runner.write_text('echo physics >> "$CALLS"\n')
        environment = root / 'environment.sh'
        environment.write_text('true\n')
        scripts = {
            'sha256sum': '#!/bin/bash\nshasum -a 256 "$@"\n',
            'sacct': '#!/bin/bash\nprintf "%s|\\n" "$TEST_PARENT_STATE"\n',
            'mpiicc': '#!/bin/bash\necho compile >> "$CALLS"\n',
            'timeout': '#!/bin/bash\nshift\nexec "$@"\n',
            'mpirun': '#!/bin/bash\necho probe >> "$CALLS"\n'
                       'if [[ "$TEST_PROBE_OK" = yes ]]; then\n'
                       'echo "MPI_COMM_PREFLIGHT_OK ranks=48 rounds=50 seconds=31"\n'
                       'else exit 9; fi\n',
        }
        for name, body in scripts.items():
            path = binaries / name
            path.write_text(body)
            path.chmod(0o755)
        env = dict(os.environ, PATH=str(binaries) + ':' + os.environ['PATH'],
                   C_SOLID_DFDCU_STAGE_ROOT=str(root), OUTPUT_ROOT=str(root / 'result'),
                   DFDCU_ENV=str(environment), SLURM_ARRAY_TASK_ID='2',
                   SLURM_NTASKS='48', SLURM_NTASKS_PER_NODE='1', SLURM_CPUS_PER_TASK='30',
                   RECOVERY_PARENT_JOB_ID='21870198_2',
                   MPI_PREFLIGHT_SOURCE_SHA256=hashlib.sha256(probe.read_bytes()).hexdigest(),
                   PREFLIGHT_WRAPPER_SHA256=hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
                   CALLS=str(root / 'calls'), TEST_PARENT_STATE=parent_state,
                   TEST_PROBE_OK='yes' if probe_ok else 'no')
        env.pop('C_SOLID_CHARGE_RESTART_CONTRACT', None)
        return env

    def test_success_runs_probe_before_physics_and_prevents_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root)
            first = subprocess.run(['bash', str(WRAPPER)], env=env, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual((root / 'calls').read_text().splitlines(), ['compile', 'probe', 'physics'])
            second = subprocess.run(['bash', str(WRAPPER)], env=env, capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((root / 'calls').read_text().splitlines(), ['compile', 'probe', 'physics'])

    def test_probe_failure_never_enters_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = self.fixture(root, probe_ok=False)
            result = subprocess.run(['bash', str(WRAPPER)], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('physics', (root / 'calls').read_text())

    def test_running_parent_and_hash_mismatch_stop_before_probe(self):
        for invalid in ('running', 'hash'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env = self.fixture(root, parent_state='RUNNING' if invalid == 'running' else 'NODE_FAIL')
                if invalid == 'hash':
                    env['MPI_PREFLIGHT_SOURCE_SHA256'] = '0' * 64
                result = subprocess.run(['bash', str(WRAPPER)], env=env, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / 'calls').exists())


if __name__ == '__main__':
    unittest.main()
