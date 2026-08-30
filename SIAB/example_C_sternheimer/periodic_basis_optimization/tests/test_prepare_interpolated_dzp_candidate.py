#!/usr/bin/env python3

import json
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
from prepare_interpolated_dzp_candidate import prepare_candidate


class PrepareInterpolatedDzpCandidateTest(unittest.TestCase):
    def test_writes_reverse_direction_candidate_and_locked_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent / "original.txt"
            optimized = parent / "optimized.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(7)
            initial = {
                "C": [
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 2, generator=generator, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                ]
            }
            selected = {
                "C": [channel + 0.01 for channel in initial["C"]]
            }
            write_periodic_optimizer_coefficients(original, initial)
            write_periodic_optimizer_coefficients(optimized, selected)

            result = prepare_candidate(
                original=original,
                optimized=optimized,
                root=root,
                alpha=-1.0,
            )

            self.assertEqual(result["profile"], "interpolated_dzp")
            self.assertEqual(result["alpha"], -1.0)
            self.assertEqual(result["nu"], [3, 3, 2, 0, 0])
            self.assertEqual(result["ao_count_atom"], 22)
            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 0),
            )
            for actual, initial_channel, selected_channel in zip(
                restored["C"], initial["C"], selected["C"]
            ):
                self.assertTrue(
                    torch.allclose(
                        actual,
                        initial_channel - (selected_channel - initial_channel),
                    )
                )
            self.assertTrue((root / result["orbital_filename"]).is_file())
            self.assertEqual((root / "STATUS").read_text(encoding="ascii"), "success\n")
            payload = json.loads((root / "CANDIDATE.json").read_text(encoding="ascii"))
            self.assertEqual(payload, result)

    def test_rejects_non_reverse_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "missing.txt"
            with self.assertRaisesRegex(ValueError, "negative"):
                prepare_candidate(
                    original=source,
                    optimized=source,
                    root=parent / "candidate",
                    alpha=0.0,
                )

    def test_writes_channel_resolved_reverse_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent / "original.txt"
            optimized = parent / "optimized.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(19)
            initial = {
                "C": [
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 2, generator=generator, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                ]
            }
            selected = {
                "C": [
                    channel
                    + 0.01
                    * torch.randn(
                        channel.shape,
                        generator=generator,
                        dtype=torch.float64,
                    )
                    for channel in initial["C"]
                ]
            }
            write_periodic_optimizer_coefficients(original, initial)
            write_periodic_optimizer_coefficients(optimized, selected)

            result = prepare_candidate(
                original=original,
                optimized=optimized,
                root=root,
                channel_alphas=(0.0, 0.0, -1.5, 0.0, 0.0),
            )

            self.assertIsNone(result["alpha"])
            self.assertEqual(result["channel_alphas"], [0.0, 0.0, -1.5, 0.0, 0.0])
            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 0),
            )
            self.assertTrue(torch.equal(restored["C"][0], initial["C"][0]))
            self.assertTrue(torch.equal(restored["C"][1], initial["C"][1]))
            self.assertTrue(
                torch.allclose(
                    restored["C"][2],
                    initial["C"][2] - 1.5 * (selected["C"][2] - initial["C"][2]),
                )
            )

    def test_accepts_signed_channel_search_with_positive_s_component(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent / "original.txt"
            optimized = parent / "optimized.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(29)
            initial = {
                "C": [
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 2, generator=generator, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                ]
            }
            selected = {
                "C": [
                    channel
                    + 0.01
                    * torch.randn(
                        channel.shape,
                        generator=generator,
                        dtype=torch.float64,
                    )
                    for channel in initial["C"]
                ]
            }
            write_periodic_optimizer_coefficients(original, initial)
            write_periodic_optimizer_coefficients(optimized, selected)

            result = prepare_candidate(
                original=original,
                optimized=optimized,
                root=root,
                channel_alphas=(1.0, -1.0, -1.5, 0.0, 0.0),
            )

            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 0),
            )
            self.assertTrue(torch.allclose(restored["C"][0], selected["C"][0]))
            self.assertTrue(
                torch.allclose(
                    restored["C"][1],
                    initial["C"][1] - (selected["C"][1] - initial["C"][1]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    restored["C"][2],
                    initial["C"][2]
                    - 1.5 * (selected["C"][2] - initial["C"][2]),
                )
            )

    def test_rejects_all_zero_channel_alphas(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "missing.txt"
            with self.assertRaisesRegex(ValueError, "nonzero"):
                prepare_candidate(
                    original=source,
                    optimized=source,
                    root=parent / "candidate",
                    channel_alphas=(0.0, 0.0, 0.0, 0.0, 0.0),
                )

    def test_combines_two_independent_channel_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent / "original.txt"
            primary = parent / "primary.txt"
            secondary = parent / "secondary.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(37)
            initial = {
                "C": [
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 2, generator=generator, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                ]
            }
            primary_selected = {
                "C": [channel + 0.01 for channel in initial["C"]]
            }
            secondary_selected = {
                "C": [channel - 0.02 for channel in initial["C"]]
            }
            write_periodic_optimizer_coefficients(original, initial)
            write_periodic_optimizer_coefficients(primary, primary_selected)
            write_periodic_optimizer_coefficients(secondary, secondary_selected)

            result = prepare_candidate(
                original=original,
                optimized=primary,
                root=root,
                channel_alphas=(0.5, -1.0, -1.5, 0.0, 0.0),
                secondary_optimized=secondary,
                secondary_channel_alphas=(0.25, 0.0, 0.0, 0.0, 0.0),
            )

            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 0),
            )
            expected_s = (
                initial["C"][0]
                + 0.5 * (primary_selected["C"][0] - initial["C"][0])
                + 0.25 * (secondary_selected["C"][0] - initial["C"][0])
            )
            self.assertTrue(torch.allclose(restored["C"][0], expected_s))
            self.assertEqual(
                result["secondary_channel_alphas"],
                [0.25, 0.0, 0.0, 0.0, 0.0],
            )
            self.assertEqual(
                result["secondary_optimized_coefficients"],
                str(secondary.resolve()),
            )

    def test_combines_one_secondary_zeta_without_moving_its_fixed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            original = parent / "original.txt"
            primary = parent / "primary.txt"
            secondary = parent / "secondary.txt"
            root = parent / "candidate"
            generator = torch.Generator().manual_seed(41)
            initial = {
                "C": [
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 3, generator=generator, dtype=torch.float64),
                    torch.randn(31, 2, generator=generator, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                    torch.empty(31, 0, dtype=torch.float64),
                ]
            }
            primary_selected = {
                "C": [channel + 0.01 for channel in initial["C"]]
            }
            secondary_selected = {
                "C": [channel - 0.02 for channel in initial["C"]]
            }
            write_periodic_optimizer_coefficients(original, initial)
            write_periodic_optimizer_coefficients(primary, primary_selected)
            write_periodic_optimizer_coefficients(secondary, secondary_selected)

            result = prepare_candidate(
                original=original,
                optimized=primary,
                root=root,
                channel_alphas=(0.5, -1.0, -1.5, 0.0, 0.0),
                secondary_optimized=secondary,
                secondary_zeta_alphas=(
                    (0.0, 0.0, 0.25),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0),
                    (),
                    (),
                ),
            )

            restored = read_periodic_optimizer_coefficients(
                root / result["coefficients_filename"],
                element="C",
                radial_rows=31,
                expected_nu=(3, 3, 2, 0, 0),
            )
            expected_s = initial["C"][0] + 0.5 * (
                primary_selected["C"][0] - initial["C"][0]
            )
            expected_s[:, 2] += 0.25 * (
                secondary_selected["C"][0][:, 2] - initial["C"][0][:, 2]
            )
            self.assertTrue(torch.allclose(restored["C"][0], expected_s))
            self.assertEqual(
                result["secondary_zeta_alphas"],
                [[0.0, 0.0, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0], [], []],
            )


if __name__ == "__main__":
    unittest.main()
