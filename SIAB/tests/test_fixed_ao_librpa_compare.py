import pathlib
import sys
import tempfile
import unittest

import numpy as np


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from fixed_ao_librpa_compare import compare_response, read_frequency_grid


class FixedAOLibRPACompareTest(unittest.TestCase):
    def test_equivalent_m_and_chi_responses_match_after_coulomb_transform(self):
        coulomb = np.array([[4.0, 1.0], [1.0, 2.0]], dtype=np.complex128)
        chi = np.array([[-0.30, 0.04j], [-0.04j, -0.15]], dtype=np.complex128)
        m_response = coulomb @ chi @ coulomb

        result = compare_response(coulomb, m_response, chi, threshold=0.0)

        self.assertEqual(result["n_coulomb_positive"], 2)
        self.assertEqual(result["n_coulomb_dropped"], 0)
        self.assertLess(result["m_relative_frobenius"], 1.0e-13)
        self.assertLess(result["pi_relative_frobenius"], 1.0e-13)
        self.assertLess(result["integrand_absolute_error"], 1.0e-13)

    def test_nonpositive_coulomb_mode_is_projected_from_both_responses(self):
        coulomb = np.diag([3.0, 0.0]).astype(np.complex128)
        chi = np.diag([-0.2, -9.0]).astype(np.complex128)
        m_response = coulomb @ chi @ coulomb

        result = compare_response(coulomb, m_response, chi, threshold=0.0)

        self.assertEqual(result["n_coulomb_positive"], 1)
        self.assertEqual(result["n_coulomb_dropped"], 1)
        self.assertLess(result["pi_relative_frobenius"], 1.0e-13)

    def test_reads_librpa_frequency_nodes_and_weights(self):
        stdout = """
Grid type: Minimax time-frequency grids
Frequency node & weight:
 0      0.125      0.25
 1      0.500      0.75
Time node & weight:
"""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "librpa.stdout"
            path.write_text(stdout, encoding="ascii")
            frequency, weight = read_frequency_grid(path)

        np.testing.assert_allclose(frequency, [0.125, 0.5], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(weight, [0.25, 0.75], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
