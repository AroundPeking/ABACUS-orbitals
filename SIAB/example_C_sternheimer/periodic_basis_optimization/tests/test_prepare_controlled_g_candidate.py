#!/usr/bin/env python3

import json
import numpy as np
import sys
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
SIAB_ROOT = MODULE_ROOT.parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(SIAB_ROOT / "opt_orb_pytorch_dpsi"))

from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)
from prepare_controlled_g_candidate import (  # noqa: E402
    _count_interior_nodes,
    prepare_candidate,
)


class PrepareControlledGCandidateTest(unittest.TestCase):
    def test_node_counter_ignores_origin_recurrence_noise_but_keeps_physical_nodes(self):
        radius = np.arange(1001, dtype=float) * 0.01
        nodeless = radius**4 * np.exp(-radius)
        nodeless[1:6] *= -1.0e-4
        self.assertEqual(_count_interior_nodes(radius, nodeless), 0)

        one_node = nodeless * (radius - 2.0)
        self.assertEqual(_count_interior_nodes(radius, one_node), 1)

    def test_appends_lowest_g_primitive_without_changing_dzp(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "base.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(59)
            base = {
                "C": [
                    torch.randn(31, count, generator=generator, dtype=torch.float64)
                    if count
                    else torch.empty(31, 0, dtype=torch.float64)
                    for count in (3, 3, 2, 0, 0)
                ]
            }
            write_periodic_optimizer_coefficients(source, base)

            result = prepare_candidate(source=source, root=root)

            self.assertEqual(result["profile"], "controlled_lowest_g")
            self.assertEqual(result["nu"], [3, 3, 2, 0, 1])
            self.assertEqual(result["ao_count_atom"], 31)
            self.assertEqual(result["seed_definition"], "lowest_l4_spherical_bessel_primitive")
            self.assertEqual(result["g_diagnostics"]["interior_node_count"], 0)
            self.assertAlmostEqual(result["g_diagnostics"]["radial_norm"], 1.0, places=12)

            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 1),
            )
            for actual, expected in zip(restored["C"][:3], base["C"][:3]):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(restored["C"][3].shape, (31, 0))
            self.assertTrue(
                torch.equal(
                    restored["C"][4][:, 0],
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
            coefficients = {
                "C": [
                    torch.zeros(31, count, dtype=torch.float64)
                    for count in (3, 3, 2, 0, 0)
                ]
            }
            write_periodic_optimizer_coefficients(source, coefficients)
            root = parent / "candidate"
            root.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_candidate(source=source, root=root)


if __name__ == "__main__":
    unittest.main()
