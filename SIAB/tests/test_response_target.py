import unittest
from dataclasses import replace

import numpy as np
import torch

import common  # noqa: F401
import periodic_available_mother as mother
import response_target as target
from response_galerkin import galerkin_pi
from response_radial_fit import ResponseFitSector, shared_radial_fit_loss
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock
import test_periodic_available_mother as mother_tests


class ResponseTargetTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(43)
        a = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
        h = a@a.conj().T+np.eye(5)
        source = rng.normal(size=(2, 3, 5))+1j*rng.normal(size=(2, 3, 5))
        return h, source, np.array([-.5, -.3]), np.array([.8, .4]), np.array([.1, 2., 9.]), np.array([.2, .7, 1.3])

    def test_covariance_matches_both_explicit_resolvents_and_pi(self):
        h, source, eps, occ, freq, weights = self.fixture()
        covariance, pi, report = target.spectral_response_target(
            h, source, eps, occ, freq, weights, k_weight=.3, q_weight=.125)
        expected = np.zeros_like(h)
        expected_pi = np.zeros_like(pi)
        for iw, w in enumerate(freq):
            for n, occupation in enumerate(occ):
                for sign in (1, -1):
                    x = np.linalg.solve(h-eps[n]*np.eye(5)+sign*1j*w*np.eye(5),
                                        -source[n].conj().T)
                    expected += .5*.125*.3*occupation*weights[iw]*(x@x.conj().T)
                    expected_pi[iw] += .3*occupation*(source[n]@x)
        np.testing.assert_allclose(covariance, expected, rtol=2e-13, atol=1e-13)
        np.testing.assert_allclose(pi, expected_pi, rtol=2e-13, atol=1e-13)
        existing, _ = galerkin_pi(h, source, eps, occ, freq, np.eye(5), .3)
        np.testing.assert_allclose(pi, existing, rtol=2e-13, atol=1e-13)
        self.assertFalse(report['exact_reference_snapshot_fit'])
        self.assertEqual(report['branches'], 'same_record_plus_minus_resolvent_average')

    def test_weights_once_and_unitary_coordinates(self):
        h, s, eps, occ, freq, weights = self.fixture()
        cov, pi, _ = target.spectral_response_target(h, s, eps, occ, freq, weights, k_weight=.2, q_weight=.1)
        cov2, pi2, _ = target.spectral_response_target(h, s, eps, occ, freq, 3*weights, k_weight=.4, q_weight=.5)
        np.testing.assert_allclose(cov2, 30*cov, atol=2e-13)
        np.testing.assert_allclose(pi2, 2*pi, atol=2e-13)
        u, _ = np.linalg.qr(s[0].conj().T@ s[0]+np.eye(5))
        rotated, response, _ = target.spectral_response_target(u.conj().T@h@u, s@u, eps, occ, freq, weights, k_weight=.2, q_weight=.1)
        np.testing.assert_allclose(rotated, u.conj().T@cov@u, atol=2e-13)
        np.testing.assert_allclose(response, pi, atol=2e-13)

    def test_bad_weights_gaps_and_hamiltonian_rejected(self):
        h, s, eps, occ, freq, weights = self.fixture()
        for bad in (np.array([.2, -.1, 1.]), np.ones(2), np.array([np.nan, 1., 1.])):
            with self.assertRaisesRegex(ValueError, 'weight'):
                target.spectral_response_target(h, s, eps, occ, freq, bad, k_weight=.2, q_weight=.1)
        with self.assertRaisesRegex(ValueError, 'gap'):
            target.spectral_response_target(-h, s, eps, occ, freq, weights, k_weight=.2, q_weight=.1)
        h[0, 1] += .1j
        with self.assertRaisesRegex(ValueError, 'Hermitian'):
            target.spectral_response_target(h, s, eps, occ, freq, weights, k_weight=.2, q_weight=.1)

    def test_finite_q_target_embedding_and_full_mother_reconstruction(self):
        dataset = mother_tests.AvailableMotherTest().fixture()
        r = dataset.kpoints[0]
        change = torch.tensor([[1., .2j, .3], [.1j, 1., .1], [0., .1j, 1.]], dtype=torch.complex128)
        h = torch.diag(torch.tensor([-.5, .8, 1.5], dtype=torch.complex128))
        r = replace(r, target_ik=3, reciprocal_shift=(1, 0, 0), k_weight=.3,
            overlap=change.T.conj()@change, hamiltonian_ha=change.T.conj()@h@change,
            source=torch.tensor([[[0., .3j, .2], [0., .1, -.1j]]], dtype=torch.complex128)@change,
            occupied_projection=torch.tensor([[1., 0., 0.]], dtype=torch.complex128)@change,
            occupation=torch.tensor([.6], dtype=torch.float64), reference_projection=torch.empty(0))
        dataset = replace(dataset, selected_iq=22, q_weight=.125, primitive_count=3,
            kpoints=(r,), frequency_ha=torch.tensor([.2, 1.]), frequency_weights_ha=torch.tensor([.4, .8]),
            reference_response=torch.zeros((2, 2, 2), dtype=torch.complex128))
        pieces = []
        target.build_available_response_targets(dataset, lambda *args: pieces.append(args),
                                                 relative_rank_tolerance=1e-10)
        self.assertEqual(len(pieces), 1)
        cov, embedding, pi, metadata = pieces[0]
        expected, _ = mother.available_mother_response(dataset, relative_rank_tolerance=1e-10)
        np.testing.assert_allclose(pi, expected, rtol=2e-13, atol=1e-14)
        self.assertEqual(metadata['target_ik'], 3)
        self.assertEqual(metadata['reciprocal_shift'], [1, 0, 0])
        self.assertEqual(metadata['q_weight'], .125)
        self.assertEqual(metadata['coordinate_merge'], 'none_per_q_source_target_record')
        # An uncontracted primitive set spans the entire virtual target.
        block = PeriodicGalerkinPrimitiveBlock('C', 0, 0, 0, 3, 0)
        sector = ResponseFitSector('q22-k1-k3', (block,), embedding, cov, occupied_rank=1)
        coefficients = {'C': [torch.eye(3, dtype=torch.float64).requires_grad_()]}
        loss, report = shared_radial_fit_loss(coefficients, [sector])
        self.assertLess(abs(float(loss.detach())), 1e-12)
        self.assertEqual(report['augmented_total_rank_by_sector'], [3])


if __name__ == '__main__':
    unittest.main()
