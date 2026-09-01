#!/usr/bin/env python3

import hashlib
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

    def test_appends_nodeless_contracted_g_primitive(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "base.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(61)
            base = {
                "C": [
                    torch.randn(31, count, generator=generator, dtype=torch.float64)
                    if count
                    else torch.empty(31, 0, dtype=torch.float64)
                    for count in (3, 3, 2, 0, 0)
                ]
            }
            write_periodic_optimizer_coefficients(source, base)

            result = prepare_candidate(
                source=source,
                root=root,
                second_primitive_amplitude=0.2,
            )

            self.assertEqual(result["profile"], "controlled_contracted_g")
            self.assertEqual(result["second_primitive_amplitude"], 0.2)
            self.assertEqual(result["g_diagnostics"]["interior_node_count"], 0)
            self.assertLess(
                result["g_diagnostics"]["tail_probability_r_ge_9_bohr"],
                0.03,
            )
            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 1),
            )
            self.assertEqual(restored["C"][4][0, 0], 1.0)
            self.assertEqual(restored["C"][4][1, 0], 0.2)

    def test_reuses_only_optimized_g_without_changing_dzp_or_keeping_f(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "base.txt"
            optimized = parent / "joint-optimized.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(67)
            base = {
                "C": [
                    torch.randn(31, count, generator=generator, dtype=torch.float64)
                    if count
                    else torch.empty(31, 0, dtype=torch.float64)
                    for count in (3, 3, 2, 0, 0)
                ]
            }
            optimized_coefficients = {
                "C": [
                    torch.randn(31, count, generator=generator, dtype=torch.float64)
                    if count
                    else torch.empty(31, 0, dtype=torch.float64)
                    for count in (3, 3, 2, 1, 1)
                ]
            }
            optimized_coefficients["C"][4].zero_()
            optimized_coefficients["C"][4][0, 0] = 1.0
            write_periodic_optimizer_coefficients(source, base)
            write_periodic_optimizer_coefficients(optimized, optimized_coefficients)

            result = prepare_candidate(
                source=source,
                root=root,
                optimized_g_source=optimized,
            )

            self.assertEqual(result["profile"], "controlled_optimized_g")
            self.assertEqual(result["seed_definition"], "joint_atom_solid_optimized_g_only")
            self.assertEqual(
                result["optimized_g_source_sha256"],
                hashlib.sha256(optimized.read_bytes()).hexdigest(),
            )
            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 1),
            )
            for actual, expected in zip(restored["C"][:3], base["C"][:3]):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(restored["C"][3].shape, (31, 0))
            self.assertTrue(torch.equal(restored["C"][4], optimized_coefficients["C"][4]))

    def test_rejects_mixing_optimized_g_with_primitive_amplitude(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "base.txt"
            optimized = parent / "joint-optimized.txt"
            base = {"C": [torch.zeros(31, count, dtype=torch.float64) for count in (3, 3, 2, 0, 0)]}
            joint = {"C": [torch.zeros(31, count, dtype=torch.float64) for count in (3, 3, 2, 1, 1)]}
            joint["C"][4][0, 0] = 1.0
            write_periodic_optimizer_coefficients(source, base)
            write_periodic_optimizer_coefficients(optimized, joint)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                prepare_candidate(
                    source=source,
                    root=parent / "candidate",
                    second_primitive_amplitude=0.1,
                    optimized_g_source=optimized,
                )

    def test_lowpass_projects_optimized_g_to_requested_primitive_count(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "base.txt"
            optimized = parent / "joint-optimized.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(71)
            base = {
                "C": [
                    torch.randn(
                        31, count, generator=generator, dtype=torch.float64
                    )
                    for count in (3, 3, 2, 0, 0)
                ]
            }
            joint = {
                "C": [
                    torch.randn(
                        31, count, generator=generator, dtype=torch.float64
                    )
                    for count in (3, 3, 2, 1, 1)
                ]
            }
            joint["C"][4][:3, 0] = torch.tensor(
                [1.0, 0.01, 0.4], dtype=torch.float64
            )
            joint["C"][4][3:, 0] = 0.2
            write_periodic_optimizer_coefficients(source, base)
            write_periodic_optimizer_coefficients(optimized, joint)

            result = prepare_candidate(
                source=source,
                root=root,
                optimized_g_source=optimized,
                optimized_g_max_primitives=3,
            )

            self.assertEqual(result["profile"], "controlled_optimized_g_lowpass")
            self.assertEqual(result["optimized_g_max_primitives"], 3)
            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 1),
            )
            expected = joint["C"][4].clone()
            expected[3:, 0] = 0.0
            expected /= torch.linalg.norm(expected)
            self.assertTrue(torch.allclose(restored["C"][4], expected, atol=1.0e-15))
            self.assertEqual(result["g_diagnostics"]["interior_node_count"], 0)

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
