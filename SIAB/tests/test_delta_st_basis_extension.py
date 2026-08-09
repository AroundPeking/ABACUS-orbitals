"""Tests for deterministic Delta-ST radial basis extension."""

import pathlib
import sys
import types
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT_DIR = ROOT / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from delta_st_basis_extension import select_metric_complement_shell
from sternheimer_data import PrimitiveBlock


class _TargetLastPrimitive:
    def evaluate(self, coefficients):
        channel = coefficients["H"][1]
        if channel.shape[1] == 1:
            loss = torch.tensor(1.0, dtype=torch.float64)
        else:
            target = torch.tensor([0.0, 0.0, 0.0, 0.5], dtype=torch.float64)
            overlap = torch.diag(
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
            )
            capture = torch.dot(channel[:, -1], overlap @ target) ** 2
            loss = 1.0 - capture
        return types.SimpleNamespace(loss=loss)


class _TargetFirstPrimitiveFromEmpty:
    def evaluate(self, coefficients):
        channel = coefficients["H"][2]
        if channel.shape[1] == 0:
            loss = torch.tensor(1.0, dtype=torch.float64)
        else:
            target = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            capture = torch.dot(channel[:, -1], target) ** 2
            loss = 1.0 - capture
        return types.SimpleNamespace(loss=loss)


class DeltaSTBasisExtensionTest(unittest.TestCase):
    def test_selects_the_best_metric_orthogonal_radial_complement(self):
        primitive = types.SimpleNamespace(
            blocks=(PrimitiveBlock("H", 0, 1, 0, 4, 0),),
            overlap=torch.diag(
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex128)
            ),
        )
        coefficients = {
            "H": [
                torch.empty((4, 0), dtype=torch.float64),
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64
                ).mT,
            ]
        }

        result = select_metric_complement_shell(
            primitive,
            _TargetLastPrimitive(),
            coefficients,
            element="H",
            l=1,
        )

        self.assertEqual(result.selected_mode, 2)
        self.assertEqual(result.coefficients["H"][1].shape, (4, 2))
        torch.testing.assert_close(
            result.coefficients["H"][1][:, 0],
            coefficients["H"][1][:, 0],
        )
        torch.testing.assert_close(
            result.coefficients["H"][1][:, 1],
            torch.tensor([0.0, 0.0, 0.0, 0.5], dtype=torch.float64),
        )
        self.assertEqual(result.initial_loss, 1.0)
        self.assertEqual(result.selected_loss, 0.0)
        self.assertLess(result.maximum_metric_orthogonality, 1.0e-14)
        self.assertLess(result.metric_normalization_error, 1.0e-14)

    def test_selects_the_first_radial_from_an_empty_angular_channel(self):
        primitive = types.SimpleNamespace(
            blocks=(PrimitiveBlock("H", 0, 2, 0, 3, 0),),
            overlap=torch.eye(3, dtype=torch.complex128),
        )
        coefficients = {
            "H": [
                torch.empty((3, 0), dtype=torch.float64),
                torch.empty((3, 0), dtype=torch.float64),
                torch.empty((3, 0), dtype=torch.float64),
            ]
        }

        result = select_metric_complement_shell(
            primitive,
            _TargetFirstPrimitiveFromEmpty(),
            coefficients,
            element="H",
            l=2,
        )

        self.assertEqual(result.selected_mode, 0)
        self.assertEqual(result.coefficients["H"][2].shape, (3, 1))
        self.assertEqual(result.initial_loss, 1.0)
        self.assertEqual(result.selected_loss, 0.0)

    def test_averages_the_magnetic_metrics_for_a_shared_radial(self):
        blocks = tuple(
            PrimitiveBlock("H", 0, 2, m, 3, 3 * index)
            for index, m in enumerate((-2, -1, 0, 1, 2))
        )
        diagonal_by_m = (
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.1),
            (1.0, 1.0, 0.9),
            (1.0, 1.0, 1.1),
            (1.0, 1.0, 0.9),
        )
        overlap = torch.block_diag(
            *(
                torch.diag(torch.tensor(diagonal, dtype=torch.complex128))
                for diagonal in diagonal_by_m
            )
        )
        primitive = types.SimpleNamespace(blocks=blocks, overlap=overlap)
        coefficients = {
            "H": [
                torch.empty((3, 0), dtype=torch.float64),
                torch.empty((3, 0), dtype=torch.float64),
                torch.empty((3, 0), dtype=torch.float64),
            ]
        }

        result = select_metric_complement_shell(
            primitive,
            _TargetFirstPrimitiveFromEmpty(),
            coefficients,
            element="H",
            l=2,
        )

        self.assertEqual(result.selected_mode, 0)
        self.assertAlmostEqual(
            result.maximum_magnetic_metric_relative_deviation,
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
