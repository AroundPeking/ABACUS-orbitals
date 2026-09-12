import importlib.util
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[1] / 'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/c_portable_frozen_replay.py'


class PortableReplayTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('portable_replay', FILE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def fixture(self):
        labels = [1, 2, 3, 6, 7, 8, 11, 28]
        indices = [1, 22, 43, 6, 27, 23, 11, 55]
        mult = [1, 8, 4, 6, 24, 12, 3, 6]
        f = {'status': 'success', 'datasets': []}
        c = {'status': 'success', 'scope': 'exact_active_dataset_derivative_not_new_reference', 'records': []}
        for label, iq, n in zip(labels, indices, mult):
            item = dict(label=label, selected_iq=iq, q_weight=n/64., manifest_sha256='a'*64,
                        status_sha256='b'*64, acceptance_sha256='c'*64)
            f['datasets'].append(item)
            c['records'].append(dict(label=label, complete_sha256='d'*64, binding=dict(
                manifest_sha256='a'*64, status_sha256='b'*64,
                identifiers=dict(freeze_sha256=self.m.FREEZE_SHA,
                                 acceptance_sha256='c'*64, mapping_sha256='e'*64))))
        return f, c

    def test_canonical_mapping_preserves_original_bindings(self):
        f, c = self.fixture()
        self.assertEqual(len(self.m.validate_cache_contract(f, c)), 8)

    def test_missing_duplicate_and_reordered_q_rejected(self):
        for action in ('missing', 'duplicate', 'reorder'):
            f, c = self.fixture()
            if action == 'missing': c['records'].pop()
            elif action == 'duplicate': c['records'][-1] = c['records'][0]
            else: f['datasets'].reverse()
            with self.assertRaises(ValueError): self.m.validate_cache_contract(f, c)

    def test_changed_freeze_and_weights_rejected(self):
        f, c = self.fixture()
        c['records'][0]['binding']['identifiers']['freeze_sha256'] = 'f'*64
        with self.assertRaises(ValueError): self.m.validate_cache_contract(f, c)
        f, c = self.fixture(); f['datasets'][0]['q_weight'] = 1.
        with self.assertRaises(ValueError): self.m.validate_cache_contract(f, c)

    def test_relative_path_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ('../outside', '/absolute'):
                with self.assertRaises(ValueError): self.m.within(Path(root), name)
            self.assertEqual(self.m.within(Path(root), 'data/q1'), Path(root).resolve()/'data/q1')

    def test_replay_rejects_energy_gradient_or_reference_change(self):
        self.m.check_replay(-.43, -.43, [1., 2.], [1., 2.], -.514, -.514)
        for args in ((-.42, -.43, [1.], [1.], -.514, -.514),
                     (-.43, -.43, [2.], [1.], -.514, -.514),
                     (-.43, -.43, [1.], [1.], -.510, -.514),
                     (float('nan'), -.43, [1.], [1.], -.514, -.514)):
            with self.assertRaises(ValueError): self.m.check_replay(*args)

    def test_full_q_diagnostic_does_not_release_physics(self):
        self.assertEqual(self.m.PROBE_STEPS, (-1e-4, 1e-4, 5e-4, 1e-3))
        self.assertEqual(self.m.PHYSICAL_RELEASE, 'hold')

    def test_q_replay_rejects_compensating_frequency_changes(self):
        import copy
        rpa = dict(complete_q_weight=True, q_weight_coverage=1., per_q=[
            dict(selected_iq=iq, q_weight=n/64., frequency_ha=list(range(1,13)),
                 candidate_contributions_ha=[-.001]*12,
                 reference_contributions_ha=[-.002]*12)
            for iq,n in zip(self.m.INDICES, self.m.MULT)])
        self.m.check_q_replay(rpa, rpa)
        changed=copy.deepcopy(rpa)
        changed['per_q'][0]['candidate_contributions_ha'][0] += .0001
        changed['per_q'][0]['candidate_contributions_ha'][1] -= .0001
        with self.assertRaises(ValueError): self.m.check_q_replay(changed, rpa)

    def test_evaluation_settings_cannot_change_frozen_objective(self):
        settings=dict(frequency_batch_size=3, weights=dict(
            energy_weight=1., pi_weight=1., trace_log_weight=1.))
        self.m.check_settings(settings)
        settings['weights']['pi_weight']=0.
        with self.assertRaises(ValueError): self.m.check_settings(settings)


if __name__ == '__main__': unittest.main()
