"""Tests for the bounded Delta-ST response gradient gate."""

import pathlib
import sys
import tempfile
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT_DIR = ROOT / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from delta_st_gradient_gate import (
    require_file_sha256,
    run_delta_st_gradient_gate,
)
from projected_pi_optimization import ProjectedPiOptimizationResult


class _QuadraticResponse:
    def evaluate(self, coefficients):
        value = coefficients["H"][0]
        target = torch.tensor(
            [[0.5, 0.4], [0.0, 0.6]], dtype=torch.float64
        )
        loss = torch.sum((value - target) ** 2)
        return ProjectedPiOptimizationResult(
            loss=loss,
            max_condition=1.0,
            frequency_ha=torch.tensor([0.2], dtype=torch.float64),
            frequency_loss=loss.reshape(1),
            family_results={"H": object()},
        )


class DeltaSTGradientGateTest(unittest.TestCase):
    def test_rejects_an_initial_orbital_from_a_different_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "H.orb"
            path.write_bytes(b"matched orbital\n")

            digest = require_file_sha256(path, None)
            self.assertEqual(len(digest), 64)
            self.assertEqual(require_file_sha256(path, digest), digest)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                require_file_sha256(path, "0" * 64)

    def test_masks_fixed_gradient_and_accepts_only_a_lower_loss_step(self):
        coefficients = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
            ]
        }
        fixed_before = coefficients["H"][0][:, 0].detach().clone()

        result = run_delta_st_gradient_gate(
            _QuadraticResponse(),
            coefficients,
            [{"element": "H", "l": 0, "zeta": 1}],
            step_sizes=(0.2, 0.1),
        )

        self.assertGreater(result.raw_fixed_gradient_norm, 0.0)
        self.assertEqual(result.masked_fixed_gradient_norm, 0.0)
        self.assertGreater(result.variable_gradient_norm, 0.0)
        self.assertLess(result.accepted_loss, result.initial_loss)
        self.assertTrue(
            torch.equal(result.coefficients["H"][0][:, 0], fixed_before)
        )
        self.assertFalse(
            torch.equal(
                result.coefficients["H"][0][:, 1],
                coefficients["H"][0][:, 1],
            )
        )


if __name__ == "__main__":
    unittest.main()
