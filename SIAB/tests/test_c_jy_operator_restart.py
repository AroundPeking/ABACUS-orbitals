import importlib.util
from pathlib import Path
import unittest
import struct
import tempfile
import json

PATH = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/prepare_c_jy_operator_restart.py'
SPEC = importlib.util.spec_from_file_location('operator_restart', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OperatorRestartTest(unittest.TestCase):
    def test_finite_q_requires_accepted_gamma_compatibility(self):
        self.assertEqual(MODULE.export_q_contract(1, None)['selected_iq'], 1)
        with self.assertRaisesRegex(ValueError, 'Gamma'):
            MODULE.export_q_contract(22, None)
        with self.assertRaisesRegex(ValueError, 'representative'):
            MODULE.export_q_contract(2, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'RESULT.json'
            result = dict(status='success', gamma_operator_compatibility_gate='hold',
                failure_reasons=['k1'], per_k=[dict(pass_gate=True)]*64)
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'Gamma'):
                MODULE.export_q_contract(22, path)
            result.update(gamma_operator_compatibility_gate='pass', failure_reasons=[])
            path.write_text(json.dumps(result))
            contract = MODULE.export_q_contract(22, path)
            self.assertEqual(contract['selected_iq'], 22)
            self.assertEqual(contract['gamma_acceptance_sha256'], MODULE.sha(path))

    def test_gamma_coulomb_reuse_requires_complete_unique_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for rank, pair in zip((0, 12, 36), range(3)):
                path = root / ('v1_coulomb_full_iq_1_rank%d.dat' % rank)
                path.write_bytes(struct.pack('<8iiq2d', -20129433, 1, 2, 1,
                    2, 1, 1, 1, pair, 44, 1.0, 0.0))
                files.append(path)
            self.assertEqual(MODULE.gamma_coulomb_files(root), files)
            files[-1].unlink()
            with self.assertRaisesRegex(ValueError, 'complete'):
                MODULE.gamma_coulomb_files(root)
            files[-1].write_bytes(files[0].read_bytes())
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                MODULE.gamma_coulomb_files(root)

    def test_input_retains_values_and_rejects_duplicates(self):
        self.assertEqual(MODULE.input_values('INPUT_PARAMETERS\nnfreq 12 # frozen\n'), {'nfreq': '12'})
        with self.assertRaises(ValueError):
            MODULE.input_values('nbands 44\nNBANDS 22\n')

    def test_frequency_grid_is_not_regenerated(self):
        text = 'ABACUS_STERNHEIMER_BASIS_OPT_MANIFEST_V1\n'
        text += ''.join('frequency %d %.17e %.17e\n' % (i, i+0.1, i+0.2) for i in range(12))
        rows = MODULE.frequency_rows(text).splitlines()
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0], '1.00000000000000006e-01 2.00000000000000011e-01')
        with self.assertRaises(ValueError):
            MODULE.frequency_rows(text.replace('frequency 11', 'frequency 10'))
        with self.assertRaises(ValueError):
            MODULE.frequency_rows(text.replace('BASIS_OPT', 'BASIS_OPERATORS'))


if __name__ == '__main__':
    unittest.main()
