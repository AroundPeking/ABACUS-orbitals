from dataclasses import replace
from pathlib import Path
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from IO.read_sternheimer_primitive_galerkin import (
    read_sternheimer_primitive_galerkin,
)
from primitive_projection import project_fixed_ao_to_primitives


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "sternheimer_galerkin_primitive_v1.dat"
)


class PrimitiveProjectionTest(unittest.TestCase):
    def setUp(self):
        self.data = read_sternheimer_primitive_galerkin(FIXTURE)

    def test_rank_revealing_projection_reproduces_fixed_ao_grid_matrices(self):
        result = project_fixed_ao_to_primitives(
            self.data, relative_rank_tolerance=1.0e-8
        )

        expected = torch.tensor(
            [[0.5, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(result.coefficients, expected)
        self.assertEqual(result.numerical_rank, 2)
        self.assertAlmostEqual(result.retained_condition, 2.0)
        self.assertLess(result.cross_overlap_relative_residual, 1.0e-14)
        self.assertLess(result.overlap_relative_residual, 1.0e-14)
        self.assertLess(max(result.hamiltonian_relative_residual), 1.0e-14)
        torch.testing.assert_close(
            result.projected_overlap, self.data.fixed_ao_grid_overlap
        )
        torch.testing.assert_close(
            result.projected_hamiltonian_ha,
            self.data.fixed_ao_grid_hamiltonian_ha,
        )

    def test_reports_material_fixed_ao_overlap_mismatch(self):
        changed = self.data.fixed_ao_grid_overlap.clone()
        changed[1, 1] += 0.25
        data = replace(self.data, fixed_ao_grid_overlap=changed)

        result = project_fixed_ao_to_primitives(
            data, relative_rank_tolerance=1.0e-8
        )

        self.assertGreater(result.overlap_relative_residual, 0.1)

    def test_rejects_materially_indefinite_primitive_overlap(self):
        overlap = self.data.overlap.clone()
        overlap[-1, -1] = -0.1
        data = replace(self.data, overlap=overlap)

        with self.assertRaisesRegex(RuntimeError, "materially indefinite"):
            project_fixed_ao_to_primitives(data)


if __name__ == "__main__":
    unittest.main()
