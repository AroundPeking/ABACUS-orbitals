import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'))
from audit_c_jy_whitening_rank import compare_whitening


class WhiteningRankTest(unittest.TestCase):
    def test_missing_direction_never_admitted(self):
        v = np.diag([1e-5, .1, 1.]).astype(complex)
        w = np.diag(1/np.sqrt(np.diag(v)))
        result = compare_whitening(v, w, v, w[:, 1:], 1e-10)
        self.assertEqual(result['threshold_ranks'], [3, 3])
        self.assertEqual(result['stored_ranks'], [3, 2])
        self.assertEqual(result['post_threshold_discarded'], [0, 1])
        self.assertAlmostEqual(result['reference_space_projector_loss_fro'], 1.)
        self.assertEqual(result['physical_release_gate'], 'hold')

    def test_raw_sign_gauge_and_phase(self):
        v = np.array([[2., .2], [.2, 1.]], complex)
        e, u = np.linalg.eigh(v)
        w = u/np.sqrt(e)
        signs = np.array([1., -1.])
        vn = signs[:, None]*v*signs[None, :]
        wn = signs[:, None]*w*np.array([1j, -1.])
        result = compare_whitening(v, w, vn, wn, 1e-10)
        self.assertLess(result['metric']['relative'], 1e-14)
        self.assertLess(result['reference_space_projector_loss_fro'], 1e-13)

    def test_reject_nonfinite_and_bad_threshold(self):
        v = np.eye(2, dtype=complex)
        for threshold in (0., float('nan'), 1.):
            with self.assertRaises(ValueError):
                compare_whitening(v, v, v, v, threshold)
        vn = v.copy()
        vn[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            compare_whitening(v, v, vn, v, 1e-10)


if __name__ == '__main__':
    unittest.main()
