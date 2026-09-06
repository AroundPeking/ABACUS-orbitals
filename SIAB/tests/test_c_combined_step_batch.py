"""Actual PBE failure must stop before any full response evaluation."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import common  # noqa: F401

WORKFLOW = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(WORKFLOW))


class CombinedBatchTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('run_c_combined_step'), 'combined runner missing')
        self.m = importlib.import_module('run_c_combined_step')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = dict(candidate_path=self.root/'CANDIDATE.json', candidate_sha256='a'*64,
            output=self.root/'pbe', center_dir=self.root/'center',
            center_preparation_sha256='b'*64, center_collection_sha256='c'*64,
            launcher=['never-run-ABACUS-in-unit-test'])
        self.first = dict(loss=.1, rpa=dict(candidate_energy_ha=-.42, reference_energy_ha=-.51))
        self.candidate = dict(loss=.09, rpa=dict(candidate_energy_ha=-.43, reference_energy_ha=-.51))

    def prepare(self, **options):
        p = Path(options['output'])
        p.mkdir()
        (p/self.m.PREPARATION).write_text('{}')

    def test_failed_actual_pbe_never_calls_forward(self):
        forward = mock.Mock(side_effect=AssertionError('must not evaluate response'))
        with mock.patch.object(self.m, 'prepare_combined_pbe', side_effect=self.prepare), \
             mock.patch.object(self.m, 'collect_combined_pbe', return_value=dict(pbe_gate='fail')), \
             mock.patch.object(self.m.subprocess, 'run') as launch:
            result = self.m.measure_candidate(**self.options, forward=forward)
        self.assertEqual(launch.call_count, 1)
        forward.assert_not_called()
        self.assertEqual(result['rpa_evaluations'], 0)
        self.assertEqual(result['candidate_gate'], 'rejected_actual_pbe')

    def test_scf_failure_and_preparation_failure_cannot_run_response(self):
        for prepare_failure in (False, True):
            options = dict(self.options, output=self.root/str(prepare_failure))
            with mock.patch.object(self.m, 'prepare_combined_pbe',
                    side_effect=ValueError('bad input') if prepare_failure else self.prepare), \
                 mock.patch.object(self.m.subprocess, 'run', side_effect=RuntimeError('failed SCF')) as launch, \
                 mock.patch.object(self.m, 'collect_combined_pbe') as collect:
                forward = mock.Mock()
                with self.assertRaises((RuntimeError, ValueError)):
                    self.m.measure_candidate(**options, forward=forward)
                forward.assert_not_called()
                collect.assert_not_called()
                self.assertEqual(launch.call_count, 0 if prepare_failure else 1)

    def test_only_after_pbe_pass_two_forwards_record_measured_improvement(self):
        for index, candidate in enumerate((self.candidate, self.first)):
            output = self.root/str(index)
            events = []
            def forward():
                events.append('forward')
                return self.first, candidate
            def collect(**kwargs):
                events.append('pbe')
                return dict(pbe_gate='pass')
            with mock.patch.object(self.m, 'prepare_combined_pbe', side_effect=self.prepare), \
                 mock.patch.object(self.m, 'collect_combined_pbe', side_effect=collect), \
                 mock.patch.object(self.m.subprocess, 'run'):
                result = self.m.measure_candidate(**dict(self.options, output=output), forward=forward)
            self.assertEqual(events, ['pbe', 'forward'])
            self.assertEqual(result['rpa_evaluations'], 2)
            self.assertEqual(result['candidate_gate'], 'improved_frozen_body' if index == 0 else 'rejected_no_improvement')
            self.assertEqual(result['physical_release_gate'], 'hold')
            self.assertEqual(result['ordinary_sos_qavg'], 'pending')

    def test_slurm_keeps_single_node_frozen_layout_and_resource_cap(self):
        text = (WORKFLOW/'run_c_combined_step.slurm').read_text()
        for line in ('#SBATCH --partition=long', '#SBATCH --nodes=1', '#SBATCH --ntasks=4',
                     '#SBATCH --cpus-per-task=7', '#SBATCH --mem=102400M', '#SBATCH --no-requeue'):
            self.assertIn(line, text)
        self.assertNotIn('--nodelist', text)
        self.assertNotIn('mpirun', text)

    def test_cheap_capture_and_overlap_rejections_use_guard_outcome(self):
        self.assertTrue(hasattr(self.m, 'screen_candidate'))
        errors = (
            'fixed radial prefix overlap has no positive direction',
            'fixed radial prefix overlap is rank deficient',
            'fixed radial prefix overlap condition number exceeds limit',
            'fixed radial prefix does not span the occupied manifold',
            'fixed radial prefix occupied capture is non-finite',
        )
        for message in errors:
            with self.subTest(message=message), \
                 mock.patch.object(self.m, '_minimum_occupied_capture', side_effect=RuntimeError(message)), \
                 self.assertRaises(self.m.CandidateGuardError):
                self.m.screen_candidate((), {}, lambda _: dict(gate=True), .999)
        for capture in (.998, float('nan')):
            with mock.patch.object(self.m, '_minimum_occupied_capture', return_value=capture), \
                 self.assertRaises(self.m.CandidateGuardError):
                self.m.screen_candidate((), {}, lambda _: dict(gate=True), .999)

    def test_cheap_screen_preserves_unexpected_failures(self):
        self.assertTrue(hasattr(self.m, 'screen_candidate'))
        with mock.patch.object(self.m, '_minimum_occupied_capture', side_effect=RuntimeError('unrelated failure')), \
             self.assertRaisesRegex(RuntimeError, '^unrelated failure$'):
            self.m.screen_candidate((), {}, lambda _: dict(gate=True), .999)

    def test_cheap_screen_records_original_floor_and_rejects_band_gate_first(self):
        self.assertTrue(hasattr(self.m, 'screen_candidate'))
        with mock.patch.object(self.m, '_minimum_occupied_capture', return_value=.9995) as capture:
            bands, value = self.m.screen_candidate((), {}, lambda _: dict(gate=True), .999)
            self.assertEqual((bands, value), (dict(gate=True), .9995))
            self.assertEqual(capture.call_args[1], dict(relative_rank_tolerance=1e-12, condition_limit=1e12))
        with mock.patch.object(self.m, '_minimum_occupied_capture') as capture, \
             self.assertRaises(self.m.CandidateGuardError):
            self.m.screen_candidate((), {}, lambda _: dict(gate=False), .999)
        capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
