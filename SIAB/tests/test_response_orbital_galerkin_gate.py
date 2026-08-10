"""Tests for the held-out response-orbital Galerkin gate."""

import dataclasses
import pathlib
import sys
import unittest


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from response_orbital_galerkin_gate import (
    evaluate_response_orbital_galerkin_gate,
)
from SIAB.tests.test_frozen_occupied_delta_st import _inputs


class ResponseOrbitalGalerkinGateTest(unittest.TestCase):
    def test_identity_response_space_matches_direct_and_spectral_forms(self):
        primitive, fixed, _ = _inputs()
        primitive = dataclasses.replace(
            primitive,
            representation="response_orbital_uniform_grid_gamma",
        )

        result = evaluate_response_orbital_galerkin_gate(
            primitive,
            fixed,
            relative_rank_tolerance=1.0e-8,
        )

        self.assertLess(result.spectral_direct_relative_frobenius, 1.0e-12)
        self.assertLess(result.spectral_direct_max_abs_difference, 1.0e-12)
        self.assertEqual(result.response_dimension, 3)
        self.assertEqual(result.fixed_dimension, 2)
        self.assertEqual(result.direct.retained_parent_rank_by_spin, (2,))
        self.assertEqual(result.spectral_diagnostics[0]["retained_virtual_rank"], 2)


if __name__ == "__main__":
    unittest.main()
