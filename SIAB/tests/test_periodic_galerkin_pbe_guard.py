import unittest
from dataclasses import replace

import torch

import common  # noqa: F401
import test_periodic_galerkin_fit as fixtures
from periodic_galerkin_pbe_guard import prepare_frozen_band_guard


class FrozenBandGuardTest(unittest.TestCase):
    def fixture(self):
        dataset = fixtures.PeriodicGalerkinFitTest().three_level_dataset()
        dataset = replace(dataset, kpoints=tuple(replace(r, k_weight=1./len(dataset.kpoints))
                                                for r in dataset.kpoints))
        initial = {'C': [torch.eye(3, dtype=torch.float64)[:, :2].clone()]}
        return dataset, initial

    def test_initial_passes_without_claiming_scf_acceptance(self):
        dataset, initial = self.fixture()
        guard = prepare_frozen_band_guard(dataset, initial, atoms_per_cell=2)
        result = guard(initial)
        self.assertTrue(result['gate'])
        self.assertEqual(result['scf_pbe_gate'], 'pending')
        self.assertEqual(result['occupied_band_sum_change_ev_per_atom'], 0.)
        self.assertEqual(result['maximum_target_band_change_ev'], 0.)

    def test_all_radial_rotation_is_allowed_when_span_is_preserved(self):
        dataset, initial = self.fixture()
        guard = prepare_frozen_band_guard(dataset, initial, atoms_per_cell=2)
        angle = torch.tensor(0.2, dtype=torch.float64)
        rotation = torch.stack((torch.stack((angle.cos(), -angle.sin())),
                                torch.stack((angle.sin(), angle.cos()))))
        coeff = initial['C'][0].matmul(rotation).requires_grad_(True)
        before = coeff.detach().clone()
        result = guard({'C': [coeff]})
        self.assertTrue(result['gate'])
        self.assertIsNone(coeff.grad)
        torch.testing.assert_close(coeff, before, rtol=0, atol=0)

    def test_virtual_spectrum_drift_is_rejected_not_relabelled_pbe(self):
        dataset, initial = self.fixture()
        guard = prepare_frozen_band_guard(dataset, initial, atoms_per_cell=2,
                                         maximum_target_band_change_ev=0.001)
        changed = initial['C'][0].clone()
        changed[2, 1] = 0.5
        from periodic_galerkin_fit import CandidateGuardError
        with self.assertRaises(CandidateGuardError):
            guard({'C': [changed]})

    def test_invalid_limits_and_missing_records_are_rejected(self):
        dataset, initial = self.fixture()
        for options in ({'atoms_per_cell': 0}, {'atoms_per_cell': True},
                        {'maximum_target_band_change_ev': float('nan')},
                        {'occupied_band_sum_limit_ev_per_atom': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_frozen_band_guard(dataset, initial, **options)

    def test_indirect_overlap_is_rejected_even_with_positive_direct_gaps(self):
        dataset, initial = self.fixture()
        first = replace(dataset.kpoints[0], k_weight=0.5)
        second = replace(first, source_ik=2, target_ik=2,
                         hamiltonian_ha=first.hamiltonian_ha+2*first.overlap)
        dataset = replace(dataset, kpoints=(first, second))
        guard = prepare_frozen_band_guard(dataset, initial)
        from periodic_galerkin_fit import CandidateGuardError
        with self.assertRaises(CandidateGuardError):
            guard(initial)


if __name__ == '__main__':
    unittest.main()
