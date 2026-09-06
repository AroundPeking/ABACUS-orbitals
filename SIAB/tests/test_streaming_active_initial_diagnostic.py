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

WORKFLOW = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(WORKFLOW))


class StreamingInitialDiagnosticTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluate_streaming_active_initial'))
        return importlib.import_module('evaluate_streaming_active_initial')

    def test_complete_q_initial_evaluation_matches_baseline_without_steps(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            c = fixtures.write_multik_fixture(directory)
            full = read_periodic_galerkin_dataset(directory, include_reference_projection=False)
            compact = read_periodic_galerkin_dataset(directory, include_reference_projection=False, active_coefficients=c)
            baseline = module.evaluate_initial((full,), c, {'C': (1, 0, 0)})
            actual = module.evaluate_initial((compact,), c, {'C': (1, 0, 0)})
            history = [baseline, {'previous_step_gradient_norm': baseline['gradient_norm']}]
            result = module.compare_history(actual, history)
            self.assertEqual(result['equivalence_gate'], 'pass')
            self.assertEqual(actual['optimizer_steps'], 0)
            self.assertEqual(actual['record_count'], 2)
            self.assertGreater(actual['gradient_norm'], 0)
            for field, delta in (('loss', .001), ('gradient_norm', .1), ('minimum_occupied_capture', .01)):
                invalid = copy.deepcopy(actual)
                invalid[field] += delta
                with self.assertRaisesRegex(RuntimeError, 'equivalence'):
                    module.compare_history(invalid, history)
            invalid = copy.deepcopy(actual)
            invalid['rpa']['per_q'][0]['q_weight'] = .5
            with self.assertRaisesRegex(RuntimeError, 'weight'):
                module.compare_history(invalid, history)

    def test_missing_or_partial_reference_is_rejected(self):
        module = self.module()
        with self.assertRaisesRegex(RuntimeError, 'history'):
            module.compare_history({}, [])

    def test_cli_hash_failure_stops_before_loading_physics(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = root/'freeze.json'
            freeze.write_text('{}')
            output = root/'out'
            argv = ['diagnostic', '--freeze', str(freeze), '--freeze-sha256', '0'*64,
                    '--history', 'unused', '--history-sha256', '0'*64,
                    '--coefficients', 'unused', '--coefficients-sha256', '0'*64,
                    '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(module, 'read_periodic_galerkin_dataset') as read:
                with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                    module.main()
                read.assert_not_called()
            self.assertEqual((output/'STATUS').read_text(), 'failed\n')
            self.assertFalse((output/'RESULT.json').exists())


if __name__ == '__main__':
    unittest.main()
