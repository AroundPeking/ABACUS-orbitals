import unittest
from dataclasses import replace

import torch
import common  # noqa: F401
import test_periodic_galerkin_fit as fixtures
from periodic_galerkin_fit import CandidateGuardError
from periodic_galerkin_pbe_guard import prepare_frozen_band_guard


class ResponseBandPolicyTest(unittest.TestCase):
    def fixture(self):
        dataset = fixtures.PeriodicGalerkinFitTest().three_level_dataset()
        record = replace(dataset.kpoints[0], k_weight=2.,
            overlap=torch.eye(6, dtype=torch.complex128),
            hamiltonian_ha=torch.diag(torch.tensor([-.5,.2,.4,.8,1.2,1.8], dtype=torch.complex128)),
            source=torch.zeros((1,1,6), dtype=torch.complex128),
            reference_projection=torch.zeros((1,1,1,6), dtype=torch.complex128),
            occupied_projection=torch.tensor([[1.,0.,0.,0.,0.,0.]], dtype=torch.complex128))
        dataset = replace(dataset, primitive_count=6, kpoints=(record,),
            primitive_blocks=(replace(dataset.primitive_blocks[0], n_primitive=6),))
        return dataset, {'C': [torch.eye(6, dtype=torch.float64)[:, :5].clone()]}

    def test_high_virtual_motion_is_diagnostic_but_legacy_rejects(self):
        from c_response_band_policy import prepare_c_band_guard
        dataset, initial = self.fixture()
        trial = {'C': [initial['C'][0].clone()]}
        trial['C'][0][5,3] = .1
        with self.assertRaises(CandidateGuardError):
            prepare_c_band_guard(dataset, initial)(trial)
        guard = prepare_c_band_guard(dataset, initial, policy='occupied_plus_two_virtual_v1')
        result = guard(trial, diagnostics=True)
        self.assertTrue(result['gate'])
        self.assertEqual(result['protected_virtual_bands'], 2)
        self.assertEqual(result['first_two_virtual_accuracy'], 'provisional_not_reference_validated')
        self.assertEqual(result['scf_pbe_gate'], 'pending')
        self.assertEqual(result['target_band_details'][0]['band_indices'], [1,2,3])
        self.assertEqual(result['unprotected_band_details'][0]['band_indices'], [4,5])
        self.assertGreater(result['maximum_unprotected_band_change_ev'], .05)

    def test_occupied_and_first_two_virtual_bands_remain_protected(self):
        from c_response_band_policy import prepare_c_band_guard
        dataset, initial = self.fixture()
        guard = prepare_c_band_guard(dataset, initial, policy='occupied_plus_two_virtual_v1')
        for column in (0,1,2):
            trial = {'C': [initial['C'][0].clone()]}
            trial['C'][0][5,column] = .1
            with self.subTest(column=column), self.assertRaises(CandidateGuardError):
                guard(trial)

    def test_legacy_outputs_and_defaults_are_unchanged(self):
        from c_response_band_policy import prepare_c_band_guard
        dataset, initial = self.fixture()
        for diagnostics in (False, True):
            expected = prepare_frozen_band_guard(dataset, initial)(initial, diagnostics=diagnostics)
            actual = prepare_c_band_guard(dataset, initial)(initial, diagnostics=diagnostics)
            self.assertEqual(actual, expected)

    def test_response_policy_forwards_free_accuracy_to_real_screen(self):
        from c_response_band_policy import prepare_c_band_guard, RESPONSE
        from run_c_combined_step import screen_candidate
        dataset, initial = self.fixture()
        guard = prepare_c_band_guard(dataset, initial, policy=RESPONSE)
        trial = {'C': [initial['C'][0].clone()]}
        trial['C'][0][5,1] = .1
        with self.assertRaises(CandidateGuardError):
            screen_candidate((dataset,),trial,guard,1e-12)
        result,capture = screen_candidate((dataset,),trial,
            lambda c: guard(c,enforce_accuracy=False),1e-12,enforce_band_accuracy=False)
        self.assertFalse(result['gate'])
        self.assertGreater(result['maximum_target_band_change_ev'],.05)
        self.assertGreater(capture,1e-12)
        with self.assertRaises(ValueError):guard(trial,enforce_accuracy='false')

    def test_invalid_policy_fails_before_loading_cache(self):
        import optimize_c_all_radial_fast as runtime
        for policy in ('two', 2, None, True):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'band guard policy'):
                runtime.load_frozen_c('missing','0'*64,{},None,band_guard_policy=policy)

    def test_full_diagnostics_are_explicit_and_do_not_change_guard(self):
        dataset, initial = self.fixture()
        guard = prepare_frozen_band_guard(dataset, initial, extra_virtual_bands=2)
        basic = guard(initial, diagnostics=True)
        full = guard(initial, diagnostics=True, include_unprotected_bands=True)
        self.assertEqual({key:full[key] for key in basic}, basic)
        self.assertNotIn('unprotected_band_details', basic)
        with self.assertRaises(ValueError):
            guard(initial, include_unprotected_bands=True)
        with self.assertRaises(ValueError):
            guard(initial, diagnostics=True, include_unprotected_bands='yes')


if __name__ == '__main__':
    unittest.main()
