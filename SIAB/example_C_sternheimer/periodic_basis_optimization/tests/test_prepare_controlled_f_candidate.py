#!/usr/bin/env python3

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
SIAB_ROOT = MODULE_ROOT.parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(SIAB_ROOT / "opt_orb_pytorch_dpsi"))

from periodic_galerkin_basis import (
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)
from prepare_controlled_f_candidate import prepare_candidate


class PrepareControlledFCandidateTest(unittest.TestCase):
    def test_appends_lowest_f_primitive_without_changing_dzp(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "base.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(31)
            base = {
                "C": [
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 2, generator=generator, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                ]
            }
            write_periodic_optimizer_coefficients(source, base)

            result = prepare_candidate(source=source, root=root)

            self.assertEqual(result["profile"], "controlled_lowest_f")
            self.assertEqual(result["nu"], [3, 3, 2, 1, 0])
            self.assertEqual(result["ao_count_atom"], 29)
            self.assertEqual(result["seed_definition"], "lowest_l3_spherical_bessel_primitive")
            self.assertEqual(result["f_diagnostics"]["interior_node_count"], 0)
            self.assertAlmostEqual(result["f_diagnostics"]["radial_norm"], 1.0, places=12)
            self.assertGreater(result["f_diagnostics"]["kinetic_energy_ry"], 0.0)
            self.assertLess(result["f_diagnostics"]["tail_probability_r_ge_9_bohr"], 0.05)
            self.assertTrue(math.isfinite(result["f_diagnostics"]["mean_radius_bohr"]))

            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 1, 0),
            )
            for actual, expected in zip(restored["C"][:3], base["C"][:3]):
                self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(
                torch.equal(
                    restored["C"][3][:, 0],
                    torch.cat(
                        (
                            torch.ones(1, dtype=torch.float64),
                            torch.zeros(30, dtype=torch.float64),
                        )
                    ),
                )
            )
            self.assertEqual((root / "STATUS").read_text(encoding="ascii"), "success\n")
            self.assertEqual(
                json.loads((root / "CANDIDATE.json").read_text(encoding="ascii")),
                result,
            )

    def test_refuses_existing_root(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "base.txt"
            root = parent / "candidate"
            coefficients = {
                "C": [
                    torch.zeros(31, count, dtype=torch.float64)
                    for count in (3, 3, 2, 0, 0)
                ]
            }
            write_periodic_optimizer_coefficients(source, coefficients)
            root.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_candidate(source=source, root=root)


if __name__ == "__main__":
    unittest.main()
