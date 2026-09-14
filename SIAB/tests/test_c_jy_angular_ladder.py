import unittest
from dataclasses import replace
from pathlib import Path
import sys

import common  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
    'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'))
import run_c_jy_angular_ladder as ladder
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock
from periodic_available_mother import available_mother_response
import test_periodic_galerkin_sternheimer as fixtures


class JyAngularLadderTest(unittest.TestCase):
    def fixture(self):
        d = fixtures.PeriodicGalerkinSternheimerTest().complete_two_level_dataset()[0]
        blocks = tuple(PeriodicGalerkinPrimitiveBlock('C', 0, l, m, 2, 2*i)
            for i, (l, m) in enumerate((l, m) for l in range(3) for m in range(2*l+1)))
        n = sum(b.n_primitive for b in blocks)
        r = d.kpoints[0]
        occupied = torch.zeros((1, n), dtype=torch.complex128)
        occupied[0, 0] = 1
        h = torch.diag(torch.arange(1, n+1, dtype=torch.float64)).to(torch.complex128)
        h[0, 0] = -.5
        source = torch.full((1, 1, n), .1, dtype=torch.complex128)
        source[0, 0, 0] = 0
        r = replace(r, overlap=torch.eye(n, dtype=torch.complex128),
            hamiltonian_ha=h, occupied_projection=occupied, source=source,
            reference_projection=torch.empty(0, dtype=torch.complex128),
            occupied_projection_normalization=torch.eye(1, dtype=torch.complex128))
        return replace(d, primitive_count=n, primitive_blocks=blocks, kpoints=(r,))

    def test_nested_view_keeps_cross_blocks_weights_and_reference(self):
        d = self.fixture()
        r = d.kpoints[0]
        h = r.hamiltonian_ha.clone()
        h[1, 2] = .02j
        h[2, 1] = -.02j
        d = replace(d, kpoints=(replace(r, hamiltonian_ha=h),))
        small = ladder.angular_view(d, 1)
        self.assertEqual(small.primitive_count, 8)
        self.assertEqual(small.kpoints[0].hamiltonian_ha[1, 2], .02j)
        self.assertIs(small.reference_response, d.reference_response)
        self.assertEqual(small.q_weight, d.q_weight)
        self.assertIs(small.kpoints[0].occupied_projection_normalization,
                      r.occupied_projection_normalization)
        self.assertEqual(small.active_primitive_reduction.source_indices, tuple(range(8)))

    def test_full_view_reproduces_mother_and_added_channel_response(self):
        d = self.fixture()
        full = ladder.angular_view(d, 2)
        expected, _ = available_mother_response(d)
        actual, _ = available_mother_response(full)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        small, _ = available_mother_response(ladder.angular_view(d, 1))
        self.assertGreater(float(np.linalg.norm(actual-small)), 1e-5)

    def test_rejects_missing_angular_blocks(self):
        d = self.fixture()
        with self.assertRaisesRegex(ValueError, 'missing'):
            ladder.angular_view(d, 3)

    def test_compressed_candidates_keep_the_unreduced_mother_operator(self):
        d = self.fixture()
        compressed = ladder.candidate_evaluation_view(d, 2, compressed=True)
        self.assertIs(compressed, d)
        self.assertIsNone(compressed.active_primitive_reduction)
        reduced = ladder.candidate_evaluation_view(d, 1, compressed=False)
        self.assertIsNotNone(reduced.active_primitive_reduction)

    def test_subset_never_claims_full_q_acceptance(self):
        d = self.fixture()
        pi, _ = available_mother_response(d)
        summary = ladder.energy_summary(d, pi)
        self.assertFalse(summary['full_q_admitted'])
        self.assertEqual(summary['physical_release_gate'], 'hold')
        self.assertAlmostEqual(summary['q_weight'], d.q_weight)
        self.assertTrue(np.isfinite(summary['candidate_energy_ha']))


if __name__ == '__main__':
    unittest.main()
