import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).parents[1] / 'galerkin_binding_workflow' / 'c_solid_charge_restart.py'


class ChargeRestartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('charge_restart', MODULE)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def make_parent(self, root):
        parent = root / 'parent'
        parent.mkdir()
        suffix = 'C_DIAMOND_FD8_Q8_NFREQ12_PCA1E6_BASIS_OPT_DFDCU'
        out = parent / ('OUT.' + suffix)
        out.mkdir()
        (parent / 'INPUT').write_text('INPUT_PARAMETERS\nsuffix ' + suffix + '\nsternheimer_q_index 23\nnspin 1\nrpa 1\nout_sternheimer_basis_opt 1\nout_sternheimer_librpa 0\n')
        for name in self.mod.INPUT_FILES[1:]:
            (parent / name).write_text(name + '\n')
        (out / (suffix + '-CHARGE-DENSITY.restart')).write_bytes(b'charge')
        (out / 'running_scf.log').write_text('#SCF IS CONVERGED#\n#TOTAL ENERGY# -309.85908204 eV\n')
        (parent / 'abacus.out').write_text('Etot_without_rpa(Ha): -11.030309685369982\nTransport retry count exceeded\nPMPI_Barrier\n')
        return parent

    def test_prepare_keeps_physics_and_parent_unchanged(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            parent = self.make_parent(root)
            contract = self.mod.audit_parent(parent, '21862974_4', 'FAILED', '127:0')
            work = root / 'recovery'
            work.mkdir()
            for filename in self.mod.INPUT_FILES:
                (work / filename).write_bytes((parent / filename).read_bytes())
            original = (parent / 'INPUT').read_bytes()
            self.mod.prepare(contract, work)
            actual = self.mod.read_input(work / 'INPUT')
            expected = self.mod.read_input(parent / 'INPUT')
            self.assertEqual(actual.pop('init_chg'), 'file')
            self.assertEqual(actual.pop('read_file_dir'), './restart-charge/')
            self.assertEqual(actual, expected)
            gate = self.mod.read_input(work / 'restart-pbe/INPUT')
            self.assertEqual(gate['rpa'], '0')
            self.assertEqual(gate['out_sternheimer_basis_opt'], '0')
            self.assertEqual((parent / 'INPUT').read_bytes(), original)
            with self.assertRaises(FileExistsError):
                self.mod.prepare(contract, work)

    def test_rejects_completed_parent_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            parent = self.make_parent(root)
            with self.assertRaises(ValueError):
                self.mod.audit_parent(parent, '21862974_4', 'RUNNING', '0:0')
            contract = self.mod.audit_parent(parent, '21862974_4', 'FAILED', '127:0')
            (parent / 'KPT').write_text('different')
            with self.assertRaises(ValueError):
                self.mod.verify_parent(contract)

    def test_rejects_existing_response_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            parent = self.make_parent(Path(name))
            out = next(parent.glob('OUT.*')) / 'STERNHEIMER_BASIS_OPT_V1'
            out.mkdir()
            (out / 'manifest.dat').write_text('format_version 1')
            with self.assertRaises(ValueError):
                self.mod.audit_parent(parent, '21862974_4', 'FAILED', '127:0')

    def test_scf_requires_restart_load_convergence_and_same_energy(self):
        log = '#SCF IS CONVERGED#\n#TOTAL ENERGY# -309.85908204 eV\nRead electron density from file: ../restart-charge/X.restart\n'
        self.mod.validate_scf_text(log, -309.85908204, 'X.restart')
        for bad in [log.replace('IS CONVERGED', 'IS NOT CONVERGED'), log.replace('Read electron density', 'Unused electron density'), log.replace('-309.85908204', '-309.85908')]:
            with self.assertRaises(ValueError):
                self.mod.validate_scf_text(bad, -309.85908204, 'X.restart')

    def test_response_energy_is_separate_gate(self):
        good = 'Etot_without_rpa(Ha): -11.030309685369982\n'
        self.mod.validate_response_energy(good, -11.030309685369982)
        for bad in ['Etot_without_rpa(Ha): nan', good.replace('-11.030309685369982', '-11.03030'), good + good]:
            with self.assertRaises(ValueError):
                self.mod.validate_response_energy(bad, -11.030309685369982)

    def test_runner_gates_before_response_and_default_is_opt_in(self):
        runner = MODULE.with_name('run_c_solid_fd8_q13_standard_dfdcu.slurm').read_text()
        self.assertIn('if [[ -n "${C_SOLID_CHARGE_RESTART_CONTRACT:-}" ]]', runner)
        self.assertLess(runner.index('MPI_COMM_PREFLIGHT_OK'), runner.index('validate-pbe'))
        self.assertLess(runner.index('validate-pbe'), runner.index('-o abacus.time'))
        self.assertIn('validate-response', runner)
        self.assertNotIn('UCX_TLS=tcp', runner)


if __name__ == '__main__':
    unittest.main()
