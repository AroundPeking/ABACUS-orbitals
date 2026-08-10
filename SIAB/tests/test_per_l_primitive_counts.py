"""Tests for SIAB coefficient files with per-l primitive dimensions."""

import pathlib
import sys
import tempfile
import types
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT_DIR = ROOT / "opt_orb_pytorch_dpsi"
RUNNER_DIR = ROOT / "example_H_sternheimer" / "delta_st_response_compression"
sys.path.insert(0, str(OPT_DIR))
sys.path.insert(0, str(RUNNER_DIR))

from IO.func_C import read_C_init
from run_h_gradient_gate import _primitive_layout


class PerLPrimitiveCountsTest(unittest.TestCase):
    def test_primitive_layout_returns_one_radial_count_per_l(self):
        blocks = [types.SimpleNamespace(element="H", l=0, m=0, n_primitive=2)]
        blocks.extend(
            types.SimpleNamespace(element="H", l=1, m=m, n_primitive=3)
            for m in (-1, 0, 1)
        )

        element, radial_counts, lmax = _primitive_layout(
            types.SimpleNamespace(blocks=blocks)
        )

        self.assertEqual(element, "H")
        self.assertEqual(radial_counts, (2, 3))
        self.assertEqual(lmax, 1)

    def test_coefficient_reader_uses_the_radial_count_for_each_l(self):
        text = """<Coefficient>
2 Total number of radial orbitals.
Type L Zeta-Orbital
H 0 1
1.0
2.0
Type L Zeta-Orbital
H 1 1
3.0
4.0
5.0
</Coefficient>
"""
        info = {
            "H": types.SimpleNamespace(Nl=2, Ne=(2, 3), Nu=[1, 1])
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "coefficients.dat"
            path.write_text(text, encoding="ascii")

            coefficients, metadata = read_C_init(
                path, info, return_metadata=True
            )

        self.assertEqual(tuple(coefficients["H"][0].shape), (2, 1))
        self.assertEqual(tuple(coefficients["H"][1].shape), (3, 1))
        torch.testing.assert_close(
            coefficients["H"][0][:, 0],
            torch.tensor([1.0, 2.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            coefficients["H"][1][:, 0],
            torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64),
        )
        self.assertEqual(
            metadata.loaded_indices,
            frozenset({("H", 0, 0), ("H", 1, 0)}),
        )


if __name__ == "__main__":
    unittest.main()
