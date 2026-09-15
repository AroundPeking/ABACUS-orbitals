import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch


REPO = Path(__file__).resolve().parents[2]
MODULES = REPO/'SIAB/opt_orb_pytorch_dpsi'
sys.path.insert(0, str(MODULES))
from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)

SCRIPT = (REPO/'SIAB/example_C_sternheimer/periodic_basis_optimization'
          /'galerkin_binding_workflow/export_c_jy_expanded_coefficients.py')
spec = importlib.util.spec_from_file_location('export_c_jy_expanded', SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ExportCjYExpandedCoefficientsTest(unittest.TestCase):
    def test_zero_pads_old_coefficients_without_changing_compact_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root/'old.txt'
            output = root/'expanded.txt'
            manifest = root/'EXPANSION.json'
            profile = (4, 4, 3, 2, 0)
            coefficients = {'C': [
                torch.arange(31*count, dtype=torch.float64).reshape(31, count)
                for count in profile
            ]}
            write_periodic_optimizer_coefficients(old, coefficients)

            result = export.export_expanded_coefficients(
                old, output, manifest, profile=profile,
                source_rows=31, target_rows=48)

            expanded = read_periodic_optimizer_coefficients(
                output, element='C', radial_rows=48, expected_nu=profile)
            for before, after in zip(coefficients['C'], expanded['C']):
                self.assertTrue(torch.equal(after[:31], before))
                self.assertTrue(torch.count_nonzero(after[31:]) == 0)
            self.assertEqual(result['ao_per_C'], 45)
            self.assertEqual(result['source_radial_rows'], 31)
            self.assertEqual(result['expanded_radial_rows'], 48)
            self.assertEqual(result['new_parameter_count'], 17*sum(profile))
            self.assertEqual(export.sha(output), result['expanded_coefficients_sha256'])


if __name__ == '__main__':
    unittest.main()
