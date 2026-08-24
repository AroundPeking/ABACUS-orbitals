#!/usr/bin/env python3

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "analyze_residual_spectrum.py"
SPEC = importlib.util.spec_from_file_location("analyze_residual_spectrum", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def spectrum(eigenvalues, cumulative=None):
    values = torch.tensor(eigenvalues, dtype=torch.float64)
    if cumulative is None:
        cumulative = torch.cumsum(values, dim=0) / torch.sum(values)
    else:
        cumulative = torch.tensor(cumulative, dtype=torch.float64)
    return SimpleNamespace(
        eigenvalues=values,
        cumulative_capture=cumulative,
        numerical_rank=len(eigenvalues),
        overlap_relative_deviation=2.0e-8,
        coefficients=torch.eye(4, len(eigenvalues), dtype=torch.float64),
    )


class ResidualSpectrumAnalysisTest(unittest.TestCase):
    def test_cli_default_accepts_measured_high_l_fd8_anisotropy(self):
        args = ANALYSIS.parse_args(
            [
                "--coefficients",
                "coefficients.dat",
                "--atom-target",
                "target.dat",
                "--output",
                "seed.dat",
                "--report",
                "spectrum.json",
            ]
        )
        self.assertEqual(args.magnetic_overlap_tolerance, 3.0e-4)

    def test_spectrum_record_reports_weight_capture_and_ao_score(self):
        record = ANALYSIS.spectrum_record(spectrum([9.0, 3.0, 1.0]), l=2)

        self.assertEqual(record["l"], 2)
        self.assertEqual(record["total_weight"], 13.0)
        self.assertEqual(record["leading_eigenvalue"], 9.0)
        self.assertEqual(record["added_ao"], 5)
        self.assertEqual(record["score"], 1.8)
        self.assertEqual(record["numerical_rank"], 3)
        self.assertEqual(len(record["cumulative_capture_first_three"]), 3)

    def test_unique_maximum_selects_highest_gain_per_ao(self):
        records = [
            ANALYSIS.spectrum_record(spectrum([2.0]), l=0),
            ANALYSIS.spectrum_record(spectrum([5.0]), l=1),
            ANALYSIS.spectrum_record(spectrum([8.0]), l=2),
        ]

        selection = ANALYSIS.select_channel(records)

        self.assertEqual(selection["status"], "UNIQUE_SHELL_SELECTED")
        self.assertEqual(selection["selected_l"], 0)
        self.assertEqual(selection["score"], 2.0)

    def test_top_scores_within_one_percent_require_review(self):
        records = [
            ANALYSIS.spectrum_record(spectrum([2.0]), l=0),
            ANALYSIS.spectrum_record(spectrum([5.955]), l=1),
        ]

        selection = ANALYSIS.select_channel(records)

        self.assertEqual(selection["status"], "REVIEW_REQUIRED")
        self.assertIsNone(selection["selected_l"])

    def test_nonpositive_spectra_cannot_create_a_shell(self):
        with self.assertRaisesRegex(RuntimeError, "positive"):
            ANALYSIS.select_channel(
                [
                    {
                        "l": 0,
                        "leading_eigenvalue": 0.0,
                        "score": 0.0,
                    }
                ]
            )

    def test_append_mode_changes_only_one_requested_channel(self):
        coefficients = {
            "C": [
                torch.tensor([[1.0], [0.0]], dtype=torch.float64),
                torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                torch.empty((2, 0), dtype=torch.float64),
            ]
        }
        mode = torch.tensor([0.25, 0.75], dtype=torch.float64)

        candidate = ANALYSIS.append_leading_mode(coefficients, "C", 2, mode)

        self.assertTrue(torch.equal(candidate["C"][0], coefficients["C"][0]))
        self.assertTrue(torch.equal(candidate["C"][1], coefficients["C"][1]))
        self.assertEqual(candidate["C"][2].shape, (2, 1))
        self.assertTrue(torch.equal(candidate["C"][2][:, 0], mode))
        self.assertEqual(coefficients["C"][2].shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
