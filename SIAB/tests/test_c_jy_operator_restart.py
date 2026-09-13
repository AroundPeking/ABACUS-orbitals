import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/prepare_c_jy_operator_restart.py'
SPEC = importlib.util.spec_from_file_location('operator_restart', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OperatorRestartTest(unittest.TestCase):
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
