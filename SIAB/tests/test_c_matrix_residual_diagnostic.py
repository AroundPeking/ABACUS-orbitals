import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import common  # noqa: F401
from periodic_galerkin_data import read_periodic_galerkin_dataset
import test_periodic_galerkin_streaming_reduction as fixtures

WORKFLOW = Path(__file__).resolve().parents[1] / 'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(WORKFLOW))


class CMatrixResidualDiagnosticTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluate_c_matrix_residuals'))
        return importlib.import_module('evaluate_c_matrix_residuals')

    def test_no_step_matrix_diagnostic_matches_initial_objective(self):
        module = self.module()
        from evaluate_streaming_active_initial import evaluate_initial
        with tempfile.TemporaryDirectory() as directory:
            coefficients = fixtures.write_multik_fixture(directory)
            q = read_periodic_galerkin_dataset(directory, include_reference_projection=False,
                                               active_coefficients=coefficients)
            initial = evaluate_initial((q,), coefficients, {'C': (1, 0, 0)})
            item = module.evaluate_matrix_q(q, coefficients, {'C': (1, 0, 0)})
            report = module.merge_and_compare([item], initial)
            self.assertEqual(report['baseline_equivalence_gate'], 'pass')
            self.assertEqual(report['optimizer_steps'], 0)
            self.assertAlmostEqual(report['pi_relative_squared_error'], initial['rpa']['pi_relative_squared_error'])
            self.assertEqual(len(report['per_q'][0]['residual_squared_norm']), 2)
            self.assertIn('little-endian', report['whitening_sha256_definition'])
            self.assertEqual(report['per_q'][0]['whitened_auxiliary_rank'], q.whitened_auxiliary_rank)
            tiny_nonzero_q = copy.deepcopy(item)
            tiny_nonzero_q['qpoint'] = [1.e-15, 0., 0.]
            self.assertEqual(module.merge_and_compare([tiny_nonzero_q], initial)['largest_weighted_residual_finite_q'],
                             item['selected_iq'])
            bad = copy.deepcopy(item)
            bad['candidate_contributions_ha'][0] += 0.01
            with self.assertRaisesRegex(RuntimeError, 'equivalence'):
                module.merge_and_compare([bad], initial)
            with self.assertRaisesRegex(RuntimeError, 'q identity'):
                module.merge_and_compare([item, item], initial)

    def test_cli_hash_failure_never_loads_physics(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'freeze.json').write_text('{}')
            argv = ['diag', '--freeze', str(p/'freeze.json'), '--freeze-sha256', '0'*64,
                    '--accepted-initial', 'unused', '--accepted-initial-sha256', '0'*64,
                    '--coefficients', 'unused', '--coefficients-sha256', '0'*64,
                    '--output', str(p/'out')]
            with patch.object(sys, 'argv', argv), patch.object(module, 'read_periodic_galerkin_dataset') as reader:
                with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                    module.main()
                reader.assert_not_called()
            self.assertEqual((p/'out/STATUS').read_text(), 'failed\n')
            self.assertFalse((p/'out/RESULT.json').exists())


if __name__ == '__main__':
    unittest.main()
