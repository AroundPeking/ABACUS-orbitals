import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / (
    'example_C_sternheimer/periodic_basis_optimization/'
    'galerkin_binding_workflow/prepare_c_jy_operator_restart.py')
SPEC = importlib.util.spec_from_file_location('expanded_mother', PATH)
expanded = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(expanded)


class ExpandedMotherTest(unittest.TestCase):
    def test_bessel_row_and_primitive_count_contract(self):
        self.assertEqual(expanded.radial_rows_from_bessel(100.0, 10.0), 31)
        self.assertEqual(expanded.radial_rows_from_bessel(230.0, 10.0), 48)
        self.assertEqual(expanded.primitive_count(48, 3, 2), 1536)
        self.assertEqual(expanded.primitive_count(48, 4, 2), 2400)
        expanded.validate_bessel_contract(48, 230.0, 10.0)
        expanded.validate_bessel_contract(100, 1000.0, 10.0)
        with self.assertRaisesRegex(ValueError, 'does not produce'):
            expanded.validate_bessel_contract(48, 225.0, 10.0)

    def test_existing_coefficients_are_zero_padded_without_changing_rank(self):
        coefficients = {
            'C': [
                np.arange(31 * count, dtype=np.float64).reshape(31, count)
                for count in (4, 4, 3, 2, 0)
            ]
        }
        coefficients['C'][-1] = np.empty((31, 0), dtype=np.float64)
        result = expanded.zero_pad_radial_coefficients(
            coefficients, source_rows=31, target_rows=48)
        self.assertEqual([tuple(value.shape) for value in result['C']],
                         [(48, 4), (48, 4), (48, 3), (48, 2), (48, 0)])
        for old, new in zip(coefficients['C'], result['C']):
            np.testing.assert_array_equal(new[:31], old)
            self.assertEqual(np.count_nonzero(new[31:]), 0)

    def test_zero_padding_to_hundred_rows_preserves_the_source_prefix(self):
        coefficients = {
            'C': [
                np.arange(31 * count, dtype=np.float64).reshape(31, count)
                for count in (4, 4, 3, 2, 0)
            ]
        }
        coefficients['C'][-1] = np.empty((31, 0), dtype=np.float64)
        result = expanded.zero_pad_radial_coefficients(
            coefficients, source_rows=31, target_rows=100)
        self.assertEqual([tuple(value.shape) for value in result['C']],
                         [(100, 4), (100, 4), (100, 3), (100, 2), (100, 0)])
        for old, new in zip(coefficients['C'], result['C']):
            np.testing.assert_array_equal(new[:31], old)
            self.assertEqual(np.count_nonzero(new[31:]), 0)

    def test_source_runner_checks_the_expanded_native_dimension(self):
        runner = PATH.with_name('run_c_jy_operator_restart.slurm').read_text()
        self.assertIn('expected_primitive_count', runner)
        self.assertIn('grep -Fx "primitive_count $expected_primitive_count"', runner)


if __name__ == '__main__':
    unittest.main()
