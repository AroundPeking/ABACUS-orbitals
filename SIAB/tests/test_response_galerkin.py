import importlib.util
import unittest

import numpy as np


class ResponseGalerkinTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('response_galerkin'),
                             'response Galerkin diagnostic is missing')
        import response_galerkin
        self.m = response_galerkin

    def test_occupied_complement_and_pod_order(self):
        occupied = np.array([[1., 1j, 0., 0.]])/np.sqrt(2.)
        occ, virtual, info = self.m.occupied_frames(occupied)
        np.testing.assert_allclose(occupied @ virtual, 0., atol=1e-14)
        np.testing.assert_allclose(occ.conj().T @ occ, np.eye(1), atol=1e-14)
        cov = virtual @ np.diag([1., 3., 2.]) @ virtual.conj().T
        frame, values = self.m.virtual_pod(cov, virtual)
        np.testing.assert_allclose(values, [3., 2., 1.], atol=1e-14)
        np.testing.assert_allclose(frame.conj().T @ frame, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(info['minimum_capture'], 1.)

    def test_complex_sos_and_weights(self):
        h = np.diag([-.8, -.5, .7, 1.4]).astype(complex)
        frame = np.eye(4, dtype=complex)[:, 2:]
        source = np.array([[[.2, .3j, .8+.2j, -.4j], [.1, .2, .3, .7j]],
                           [[.4, .5j, -.2+.3j, .8], [.5, .6, .4j, -.2]]])
        eps, occ, freqs = np.array([-.8, -.5]), np.array([1., .5]), np.array([.1, .9])
        actual, info = self.m.galerkin_pi(h, source, eps, occ, freqs, frame, .03125)
        expected = np.zeros_like(actual)
        for iw, w in enumerate(freqs):
            for n in range(2):
                for a, energy in enumerate([.7, 1.4]):
                    v = source[n, :, a+2]
                    gap = energy-eps[n]
                    expected[iw] -= .03125*occ[n]*2*gap/(gap**2+w**2)*np.outer(v, v.conj())
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertLess(info['spectral_residual'], 1e-13)

    def test_unitary_coordinate_invariance(self):
        rng = np.random.default_rng(97)
        a = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
        u, _ = np.linalg.qr(a)
        h = np.diag(np.arange(1., 6.))
        source = a[:2].reshape(1, 2, 5)
        frame = np.eye(5)[:, 1:4]
        args = (np.array([-.5]), np.array([1.]), np.array([.1, 2.]))
        p, _ = self.m.galerkin_pi(h, source, *args, frame, .25)
        q, _ = self.m.galerkin_pi(u.conj().T@h@u, source@u, *args, u.conj().T@frame, .25)
        np.testing.assert_allclose(p, q, atol=1e-13)

    def test_pod_projection_is_not_galerkin_solution(self):
        h = np.diag([1., 4.])
        b = np.array([1., 1.])
        frame = np.ones((2, 1))/np.sqrt(2.)
        w = .2
        full = np.linalg.solve(h+1j*w*np.eye(2), -b)
        projected = frame@frame.T@full
        reduced = frame@np.linalg.solve(frame.T@h@frame+1j*w*np.eye(1), -frame.T@b)
        self.assertGreater(np.linalg.norm(projected-reduced), .1)
        p, _ = self.m.galerkin_pi(h, b.reshape(1, 1, 2), np.array([0.]),
                                  np.ones(1), np.array([w]), frame, 1.)
        self.assertAlmostEqual(p[0, 0, 0].real, 2*np.real(b@reduced))

    def test_trace_log_sensitivity_and_positivity(self):
        p = np.diag([-.3, -.1]).astype(complex)
        value, info = self.m.trace_log(p)
        self.assertAlmostEqual(value, np.log(1.3)+np.log(1.1)-.4)
        self.assertAlmostEqual(info['min_I_minus_Pi'], 1.1)
        with self.assertRaises(ValueError):
            self.m.trace_log(np.diag([1.1, -.1]))

    def test_invalid_domains(self):
        with self.assertRaises(ValueError):
            self.m.occupied_frames(np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            self.m.virtual_pod(np.diag([-1., 1.]), np.eye(2))
        args = (np.eye(2), np.ones((1, 1, 2)), np.array([2.]), np.ones(1),
                np.array([.2]), np.eye(2), .1)
        with self.assertRaises(ValueError):
            self.m.galerkin_pi(*args)
        with self.assertRaises(ValueError):
            self.m.trace_log(np.array([[0., 1.], [0., 0.]]))


if __name__ == '__main__':
    unittest.main()
