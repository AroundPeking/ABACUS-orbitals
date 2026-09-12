import importlib.util
import unittest
import numpy as np


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('response_metric'),
                             'metric prototype implementation is missing')
        import response_metric
        self.m = response_metric
        self.rng = np.random.default_rng(73)

    def complex_array(self, shape):
        return self.rng.normal(size=shape) + 1j*self.rng.normal(size=shape)

    def test_complex_reconstruction_in_physical_space(self):
        b = self.complex_array((7, 4))
        psi = b @ self.complex_array((4, 6))
        s, q = b.conj().T @ b, psi.conj().T @ b
        metric = self.m.Metric(s)
        x, c, info = metric.restore(q, 1e-12)
        np.testing.assert_allclose(b @ c, psi, atol=1e-12)
        np.testing.assert_allclose(x.conj().T @ x, psi.conj().T @ psi, atol=1e-11)
        self.assertLess(info['projection_relative_residual'], 1e-12)

    def test_rank_deficient_metric_reports_discarded_projection(self):
        metric = self.m.Metric(np.diag([4., 1., 1e-12, 0.]))
        q = np.array([[2., 1., 1e-6, 0.]], dtype=complex)
        x, c, info = metric.restore(q, 1e-8)
        self.assertEqual(info['retained_rank'], 2)
        self.assertGreater(info['projection_relative_residual'], 0.)
        np.testing.assert_allclose(c[:, 0], [.5, 1., 0., 0.])
        self.assertAlmostEqual(float(np.vdot(x, x).real), 2.)

    def test_reject_invalid_metric_and_inputs(self):
        for s in [np.diag([1., -.01]), np.array([[1., .1], [0., 1.]]),
                  np.diag([1., np.nan]), np.zeros((2, 2))]:
            with self.assertRaises(ValueError):
                self.m.Metric(s)
        metric = self.m.Metric(np.eye(2))
        for cutoff in [-1., 0., 1., np.nan]:
            with self.assertRaises(ValueError):
                metric.restore(np.ones((1, 2)), cutoff)
        with self.assertRaises(ValueError):
            metric.restore(np.array([[1., np.inf]]), 1e-8)

    def test_unitary_mother_basis_invariance(self):
        b = self.complex_array((8, 4))
        psi = self.complex_array((8, 6))
        t, _ = np.linalg.qr(self.complex_array((4, 4)))
        spectra = []
        for basis in [b, b @ t]:
            x, _, _ = self.m.Metric(basis.conj().T @ basis).restore(psi.conj().T @ basis, 1e-12)
            spectra.append(self.m.spectrum(self.m.covariance(x, np.arange(1., 7.)))['eigenvalues'])
        np.testing.assert_allclose(spectra[0], spectra[1], rtol=1e-12, atol=1e-12)

    def test_weighted_covariance_and_rank(self):
        x = np.diag([3., 2., 1.]).astype(complex)
        cov = self.m.covariance(x, np.array([1., 1., 0.]))
        result = self.m.spectrum(cov)
        np.testing.assert_allclose(result['eigenvalues'], [9., 4., 0.])
        self.assertEqual(result['rank_for_relative_rms']['0.1'], 2)
        self.assertAlmostEqual(result['relative_rms_by_rank'][1], np.sqrt(4./13.))
        for weights in [np.array([1., -1., 1.]), np.array([np.nan]*3)]:
            with self.assertRaises(ValueError):
                self.m.covariance(x, weights)

    def test_reject_indefinite_covariance(self):
        with self.assertRaises(ValueError):
            self.m.spectrum(np.diag([1., -.1]))


if __name__ == '__main__':
    unittest.main()
