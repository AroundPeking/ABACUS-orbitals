import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from periodic_galerkin_basis import build_primitive_to_candidate
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock


class PeriodicGalerkinBasisTest(unittest.TestCase):
    def test_repeats_one_radial_channel_for_every_m_without_mixing_blocks(self):
        blocks = tuple(
            PeriodicGalerkinPrimitiveBlock("C", 0, 1, m, 2, 2 * (m + 1))
            for m in (-1, 0, 1)
        )
        coefficients = {
            "C": [
                torch.empty((2, 0), dtype=torch.float64),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
            ]
        }

        result = build_primitive_to_candidate(blocks, 6, coefficients)

        expected = torch.zeros((6, 6), dtype=torch.complex128)
        radial = coefficients["C"][1].to(torch.complex128)
        expected[0:2, 0:2] = radial
        expected[2:4, 2:4] = radial
        expected[4:6, 4:6] = radial
        torch.testing.assert_close(result.transform, expected)
        self.assertEqual(
            tuple((label.atom_index, label.l, label.m, label.zeta) for label in result.columns),
            ((0, 1, -1, 0), (0, 1, -1, 1),
             (0, 1, 0, 0), (0, 1, 0, 1),
             (0, 1, 1, 0), (0, 1, 1, 1)),
        )

    def test_reuses_element_radial_coefficients_on_each_atom(self):
        blocks = (
            PeriodicGalerkinPrimitiveBlock("C", 0, 0, 0, 2, 0),
            PeriodicGalerkinPrimitiveBlock("C", 1, 0, 0, 2, 2),
        )
        radial = torch.tensor([[0.5], [-0.2]], dtype=torch.float64)

        result = build_primitive_to_candidate(blocks, 4, {"C": [radial]})

        torch.testing.assert_close(
            result.transform[0:2, 0].contiguous(), radial[:, 0].to(torch.complex128)
        )
        torch.testing.assert_close(
            result.transform[2:4, 1].contiguous(), radial[:, 0].to(torch.complex128)
        )
        self.assertEqual(tuple(label.atom_index for label in result.columns), (0, 1))

    def test_rejects_coefficient_primitive_count_mismatch(self):
        blocks = (PeriodicGalerkinPrimitiveBlock("C", 0, 0, 0, 2, 0),)
        coefficients = {"C": [torch.ones((3, 1), dtype=torch.float64)]}

        with self.assertRaisesRegex(ValueError, "primitive count"):
            build_primitive_to_candidate(blocks, 2, coefficients)

    def test_candidate_transform_preserves_radial_coefficient_gradient(self):
        blocks = (PeriodicGalerkinPrimitiveBlock("C", 0, 0, 0, 2, 0),)
        radial = torch.tensor([[0.4], [-0.3]], dtype=torch.float64, requires_grad=True)

        result = build_primitive_to_candidate(blocks, 2, {"C": [radial]})
        loss = torch.sum(torch.abs(result.transform) ** 2)
        loss.backward()

        self.assertIsNotNone(radial.grad)
        self.assertTrue(bool(torch.isfinite(radial.grad).all()))
        self.assertGreater(float(torch.linalg.norm(radial.grad)), 0.0)


if __name__ == "__main__":
    unittest.main()
