import unittest
from dataclasses import replace

import numpy as np
import torch

import common  # noqa: F401
import periodic_available_mother as mother
from periodic_galerkin_data import PeriodicGalerkinActivePrimitiveReduction
from periodic_galerkin_sternheimer import evaluate_periodic_galerkin_mother_response
import test_periodic_galerkin_sternheimer as production_tests


class AvailableMotherTest(unittest.TestCase):
    def fixture(self):
        return production_tests.PeriodicGalerkinSternheimerTest().complete_two_level_dataset()[0]

    def test_matches_unchanged_production_without_reference_snapshots(self):
        dataset = self.fixture()
        expected = evaluate_periodic_galerkin_mother_response(dataset).response
        record = replace(dataset.kpoints[0], reference_projection=torch.empty(0))
        actual, report = mother.available_mother_response(replace(dataset, kpoints=(record,)))
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-13, atol=1e-14)
        self.assertEqual(report['space'], 'available_unreduced_mother')
        self.assertFalse(report['exact_reference_snapshot_fit'])

    def test_reduced_metadata_is_not_claimed_as_full_mother(self):
        dataset = self.fixture()
        reduction = PeriodicGalerkinActivePrimitiveReduction(3, 'a'*64, (0, 2), (), 'b'*64)
        dataset = replace(dataset, active_primitive_reduction=reduction)
        _, report = mother.available_mother_response(dataset)
        self.assertEqual(report['space'], 'available_active_reduced_mother')
        self.assertEqual(report['original_primitive_count'], 3)
        self.assertEqual(report['available_primitive_count'], 2)
        self.assertEqual(report['mapping_sha256'], 'b'*64)
        with self.assertRaisesRegex(ValueError, 'unreduced'):
            evaluate_periodic_galerkin_mother_response(dataset)

    def test_nonzero_q_mapping_and_weights_applied_once(self):
        dataset = self.fixture()
        record = replace(dataset.kpoints[0], target_ik=2, target_kpoint=(.25, 0., 0.),
                         k_weight=.3, occupation=torch.tensor([.4], dtype=torch.float64))
        dataset = replace(dataset, selected_iq=2, qpoint=(.25, 0., 0.), q_weight=.125,
                          kpoints=(record,))
        expected = evaluate_periodic_galerkin_mother_response(dataset).response
        actual, report = mother.available_mother_response(dataset)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-13, atol=1e-14)
        self.assertEqual(report['k_records'][0]['target_ik'], 2)
        self.assertAlmostEqual(report['k_weight_sum'], .3)

    def test_scale_invariance_and_null_direction(self):
        dataset = self.fixture()
        r = dataset.kpoints[0]
        t = torch.tensor([[.01, 0., 0.], [0., 20., 0.]], dtype=torch.complex128)
        r = replace(r, overlap=t.T.conj()@r.overlap@t,
                    hamiltonian_ha=t.T.conj()@r.hamiltonian_ha@t,
                    source=r.source@t, occupied_projection=r.occupied_projection@t)
        expected, _ = mother.available_mother_response(dataset)
        actual, report = mother.available_mother_response(
            replace(dataset, primitive_count=3, kpoints=(r,)))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        self.assertEqual(report['k_records'][0]['retained_rank'], 2)

    def test_occupied_normalization_preserves_span_not_source_weight(self):
        dataset = self.fixture()
        r = dataset.kpoints[0]
        r = replace(r, occupied_projection=.9*r.occupied_projection,
                    occupied_projection_normalization=torch.tensor([[1/.9]], dtype=torch.complex128))
        actual, report = mother.available_mother_response(replace(dataset, kpoints=(r,)))
        expected, _ = mother.available_mother_response(dataset)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        self.assertAlmostEqual(report['k_records'][0]['raw_minimum_capture'], .81)
        self.assertAlmostEqual(report['k_records'][0]['minimum_capture'], 1.)

    def test_rejects_indefinite_overlap_nonpositive_gap_and_capture_loss(self):
        dataset = self.fixture()
        r = dataset.kpoints[0]
        bad = replace(r, overlap=torch.tensor([[1., 2.], [2., 1.]], dtype=torch.complex128))
        with self.assertRaisesRegex(ValueError, 'indefinite'):
            mother.available_mother_response(replace(dataset, kpoints=(bad,)))
        bad = replace(r, hamiltonian_ha=-torch.eye(2, dtype=torch.complex128))
        with self.assertRaisesRegex(ValueError, 'transition gap'):
            mother.available_mother_response(replace(dataset, kpoints=(bad,)))
        bad = replace(r, occupied_projection=.9*r.occupied_projection)
        with self.assertRaisesRegex(ValueError, 'capture'):
            mother.available_mother_response(replace(dataset, kpoints=(bad,)))

    def test_complex_coupled_multiple_k_matches_production(self):
        dataset = self.fixture()
        r = dataset.kpoints[0]
        t = torch.tensor([[1., .1j, .2], [.05, 1., -.12j], [.1j, .15, 1.]],
                         dtype=torch.complex128)
        h = torch.tensor([[-.5, 0., 0.], [0., .7, .08j], [0., -.08j, 1.3]],
                         dtype=torch.complex128)
        r = replace(r, overlap=t.T.conj()@t, hamiltonian_ha=t.T.conj()@h@t,
                    source=torch.tensor([[[0., .3, .1j], [0., .02j, -.2]]],
                                        dtype=torch.complex128)@t,
                    occupied_projection=torch.tensor([[1., 0., 0.]], dtype=torch.complex128)@t,
                    reference_projection=torch.zeros((2, 1, 2, 3), dtype=torch.complex128),
                    k_weight=.25)
        r2 = replace(r, source_ik=2, target_ik=3, k_weight=.75)
        dataset = replace(dataset, selected_iq=2, qpoint=(.25, 0., 0.), primitive_count=3,
                          kpoints=(r, r2), whitened_auxiliary_rank=2, raw_auxiliary_dimension=2,
                          frequency_ha=torch.tensor([.05, 10.], dtype=torch.float64),
                          frequency_weights_ha=torch.tensor([.2, .8], dtype=torch.float64),
                          reference_response=torch.ones((2, 2, 2), dtype=torch.complex128))
        expected = evaluate_periodic_galerkin_mother_response(dataset).response
        actual, report = mother.available_mother_response(dataset)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-12, atol=1e-14)
        self.assertEqual(report['k_weight_sum'], 1.)

    def test_ill_conditioned_transform_uses_production_virtual_hermitization(self):
        dataset = self.fixture()
        r = dataset.kpoints[0]
        rng = np.random.default_rng(31)
        n = 12
        u, _ = np.linalg.qr(rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)))
        v, _ = np.linalg.qr(rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)))
        t = torch.from_numpy(u@np.diag(np.geomspace(1., 1e-5, n))@v.conj().T)
        h = torch.diag(torch.linspace(.6, 2., n, dtype=torch.float64)).to(torch.complex128)
        h[0, 0] = -.5
        source = torch.zeros((1, 1, n), dtype=torch.complex128)
        source[0, 0, 1:] = .1
        occupied = torch.zeros((1, n), dtype=torch.complex128)
        occupied[0, 0] = 1.
        r = replace(r, overlap=t.T.conj()@t, hamiltonian_ha=t.T.conj()@h@t,
                    source=source@t, occupied_projection=occupied@t,
                    reference_projection=torch.zeros((2, 1, 1, n), dtype=torch.complex128))
        dataset = replace(dataset, primitive_count=n, kpoints=(r,),
                          frequency_ha=torch.tensor([.05, 10.], dtype=torch.float64),
                          frequency_weights_ha=torch.tensor([.2, .8], dtype=torch.float64),
                          reference_response=torch.zeros((2, 1, 1), dtype=torch.complex128))
        expected = evaluate_periodic_galerkin_mother_response(dataset).response
        actual, report = mother.available_mother_response(dataset)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=1e-10)
        entry = report['k_records'][0]
        self.assertGreater(entry['virtual_h_antisymmetric_residual_before_symmetrization'], 1e-10)
        self.assertEqual(report['hamiltonian_projection'], 'production_virtual_then_hermitian_part')

    def test_nonhermitian_raw_hamiltonian_is_not_hidden_by_projection(self):
        dataset = self.fixture()
        r = dataset.kpoints[0]
        h = r.hamiltonian_ha.clone()
        h[0, 1] = .01j
        with self.assertRaisesRegex(ValueError, 'Hermitian'):
            mother.available_mother_response(replace(dataset, kpoints=(replace(r, hamiltonian_ha=h),)))


if __name__ == '__main__':
    unittest.main()
