"""One real SCF is a prerequisite, never a proxy for the full-q step."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

WF = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(WF))


class EcStepBatchTest(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('run_c_ec_gradient_step')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.slot = Path(self.temp.name)/'pbe'
        self.events = []

    def prepare(self):
        self.events.append('prepare')
        self.slot.mkdir()
        return 'a'*64

    def collect(self, digest):
        self.assertEqual(digest, 'a'*64)
        self.events.append('collect')
        return dict(pbe_gate='pass')

    def test_real_pbe_precedes_full_forward_and_gate(self):
        center = dict(loss=2., rpa=dict(candidate_energy_ha=-.4, reference_energy_ha=-.5))
        candidate = dict(loss=1., rpa=dict(candidate_energy_ha=-.42, reference_energy_ha=-.5))
        def forward():
            self.events.append('forward')
            return center, candidate
        with patch.object(self.m.subprocess, 'run', side_effect=lambda *a, **kw:self.events.append('scf')) as run:
            result = self.m.measure_candidate(self.slot, self.prepare, self.collect, ['frozen'], forward)
        self.assertEqual(self.events, ['prepare', 'scf', 'collect', 'forward'])
        self.assertEqual(run.call_args[1]['check'], True)
        self.assertEqual(result['candidate_gate'], 'improved_frozen_body')
        self.assertEqual(result['actual_scf_count'], 1)
        self.assertEqual(result['rpa_evaluations'], 2)
        self.assertEqual(result['backward_passes'], 0)
        self.assertEqual(result['physical_release_gate'], 'hold')

    def test_pbe_failure_never_calls_forward(self):
        forward = Mock(side_effect=AssertionError('forbidden response'))
        with patch.object(self.m.subprocess, 'run'):
            result = self.m.measure_candidate(self.slot, self.prepare, lambda d:dict(pbe_gate='fail'), [], forward)
        forward.assert_not_called()
        self.assertEqual(result['candidate_gate'], 'rejected_actual_pbe')
        self.assertEqual(result['rpa_evaluations'], 0)

    def test_scf_runtime_failure_propagates_before_collect(self):
        collect, forward = Mock(), Mock()
        with patch.object(self.m.subprocess, 'run', side_effect=RuntimeError('abacus failed')):
            with self.assertRaisesRegex(RuntimeError, 'abacus failed'):
                self.m.measure_candidate(self.slot, self.prepare, collect, [], forward)
        collect.assert_not_called()
        forward.assert_not_called()

    def test_no_improvement_is_rejected(self):
        center = dict(loss=1., rpa=dict(candidate_energy_ha=-.4, reference_energy_ha=-.5))
        for loss, energy in ((2., -.42), (1., -.39)):
            slot = self.slot/str(loss)
            slot.mkdir(parents=True)
            candidate = dict(loss=loss, rpa=dict(candidate_energy_ha=energy, reference_energy_ha=-.5))
            with patch.object(self.m.subprocess, 'run'):
                result = self.m.measure_candidate(slot, lambda:'a'*64, self.collect, [], lambda:(center,candidate))
            self.assertEqual(result['candidate_gate'], 'rejected_no_improvement')

    def test_equal_loss_is_allowed_only_with_smaller_energy_error(self):
        center = dict(loss=1., rpa=dict(candidate_energy_ha=-.4, reference_energy_ha=-.5))
        candidate = dict(loss=1., rpa=dict(candidate_energy_ha=-.42, reference_energy_ha=-.5))
        with patch.object(self.m.subprocess, 'run'):
            result = self.m.measure_candidate(self.slot, self.prepare, self.collect, [], lambda:(center,candidate))
        self.assertEqual(result['candidate_gate'], 'improved_frozen_body')

    def test_nonfinite_or_changed_reference_cannot_be_accepted(self):
        center = dict(loss=1., rpa=dict(candidate_energy_ha=-.4, reference_energy_ha=-.5))
        for i, (energy, reference) in enumerate(((float('nan'), -.5), (-.42, -.6))):
            slot = self.slot/str(i)
            slot.mkdir(parents=True)
            candidate = dict(loss=.9, rpa=dict(candidate_energy_ha=energy, reference_energy_ha=reference))
            with patch.object(self.m.subprocess, 'run'), self.assertRaises(ValueError):
                self.m.measure_candidate(slot, lambda:'a'*64, self.collect, [], lambda:(center,candidate))


if __name__ == '__main__':
    unittest.main()
