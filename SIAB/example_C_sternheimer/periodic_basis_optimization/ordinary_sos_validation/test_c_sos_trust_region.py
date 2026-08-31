#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c_sos_trust_region import (
    assess_surrogate,
    calibrate_high_frequency_tail,
    load_candidate,
    parse_frequency_decomposition,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="ascii")


def write_candidate_files(root, *, total, pbe=0.005):
    pbe_path = root / "PBE_GATE.json"
    binding_path = root / "RESULT.json"
    write_json(
        pbe_path,
        {
            "status": "success",
            "pbe_gate": "pass",
            "atom_energy_difference_ev": pbe,
            "solid_energy_difference_ev_per_c": pbe,
            "binding_energy_difference_ev_per_c": pbe,
        },
    )
    write_json(
        binding_path,
        {
            "status": "success",
            "zero_order_binding_ev_per_c": 5.0,
            "correlation_binding_ev_per_c": total - 5.0,
            "sos_total_binding_ev_per_c": total,
            "difference_from_delta_ev_per_c": total - 6.902326,
            "selected_orbital_sha256": "a" * 64,
        },
    )
    return pbe_path, binding_path


class CandidateDatasetTest(unittest.TestCase):
    def test_loads_locked_pbe_and_binding_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pbe_path, binding_path = write_candidate_files(root, total=7.1)
            result = load_candidate(
                {
                    "name": "stable",
                    "coordinates": {"s3": 0.25},
                    "pbe_gate": str(pbe_path),
                    "binding_result": str(binding_path),
                    "stability": "stable",
                },
                pbe_tolerance_ev=0.01,
            )

            self.assertTrue(result["pbe_pass"])
            self.assertAlmostEqual(result["sos_error_ev_per_c"], 0.197674)
            self.assertEqual(len(result["input_sha256"]["pbe_gate"]), 64)

    def test_rejects_claimed_stable_point_outside_pbe_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pbe_path, binding_path = write_candidate_files(root, total=7.1, pbe=0.011)
            with self.assertRaisesRegex(ValueError, "PBE"):
                load_candidate(
                    {
                        "name": "invalid",
                        "coordinates": {},
                        "pbe_gate": str(pbe_path),
                        "binding_result": str(binding_path),
                        "stability": "stable",
                    },
                    pbe_tolerance_ev=0.01,
                )


class SurrogateGateTest(unittest.TestCase):
    def test_rejects_sparse_axes_when_leave_one_out_loses_rank(self):
        rows = [
            {"name": "base", "coordinates": {"s": 0.0, "d": 0.0}, "sos_error_ev_per_c": 0.30, "stability": "stable"},
            {"name": "s1", "coordinates": {"s": 0.25, "d": 0.0}, "sos_error_ev_per_c": 0.27, "stability": "stable"},
            {"name": "s2", "coordinates": {"s": 0.50, "d": 0.0}, "sos_error_ev_per_c": 0.31, "stability": "stable"},
            {"name": "d1", "coordinates": {"s": 0.25, "d": 0.25}, "sos_error_ev_per_c": 0.26, "stability": "stable"},
            {"name": "barrier", "coordinates": {"s": 0.25, "d": 0.50}, "sos_error_ev_per_c": 6.4, "stability": "unstable"},
        ]
        terms = [
            {"name": "intercept", "powers": {}},
            {"name": "s", "powers": {"s": 1}},
            {"name": "s2", "powers": {"s": 2}},
            {"name": "d", "powers": {"d": 1}},
        ]

        result = assess_surrogate(rows, terms=terms, minimum_rank_concordance=0.8)

        self.assertEqual(result["model_gate"], "fail")
        self.assertEqual(result["unstable_excluded"], ["barrier"])
        self.assertIn("leave_one_out_rank_deficient", result["failure_reasons"])

    def test_accepts_redundant_quadratic_data_with_correct_loo_ranking(self):
        rows = []
        for index, x in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)):
            rows.append(
                {
                    "name": f"p{index}",
                    "coordinates": {"x": x},
                    "sos_error_ev_per_c": 0.20 + 0.04 * x + 0.03 * x * x,
                    "stability": "stable",
                }
            )
        terms = [
            {"name": "intercept", "powers": {}},
            {"name": "x", "powers": {"x": 1}},
            {"name": "x2", "powers": {"x": 2}},
        ]

        result = assess_surrogate(rows, terms=terms, minimum_rank_concordance=0.8)

        self.assertEqual(result["model_gate"], "pass")
        self.assertAlmostEqual(result["loo_mae_ev_per_c"], 0.0, places=10)
        self.assertAlmostEqual(result["loo_rank_concordance"], 1.0)


class HighFrequencyTailTest(unittest.TestCase):
    def test_extracts_q2_q6_cancellation_and_tail_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decomposition.tsv"
            lines = []
            for q_index, multiplicity in ((2, 8), (6, 6)):
                for frequency in range(6):
                    trace = -10.0 / (frequency + 1)
                    logdet = -trace - 0.1 / (frequency + 1)
                    raw = trace + logdet
                    weighted_raw = raw * (frequency + 1) * 0.01
                    lines.append(
                        "\t".join(
                            (
                                "RECORD",
                                "stable",
                                str(q_index),
                                str(multiplicity),
                                str(frequency),
                                str(frequency + 1.0),
                                "q=(0,0,0)",
                                str(trace),
                                str(logdet),
                                str(raw),
                                "0",
                                "0",
                                str(weighted_raw),
                            )
                        )
                    )
            path.write_text("\n".join(lines) + "\n", encoding="ascii")

            result = parse_frequency_decomposition(path)

            self.assertEqual(result["frequency_count_per_q"], 6)
            self.assertAlmostEqual(result["q2"]["highest_frequency_cancellation_ratio"], 0.01)
            self.assertAlmostEqual(result["q6"]["high_frequency_tail_fraction"], 1.0 / 6.0)

    def test_requires_multiple_stable_points_before_setting_a_threshold(self):
        metric = {
            "q2": {"highest_frequency_cancellation_ratio": 0.01, "high_frequency_tail_fraction": 0.02},
            "q6": {"highest_frequency_cancellation_ratio": 0.01, "high_frequency_tail_fraction": 0.02},
        }
        result = calibrate_high_frequency_tail([metric], minimum_stable_points=3)
        self.assertEqual(result["calibration_gate"], "insufficient")
        self.assertNotIn("thresholds", result)


if __name__ == "__main__":
    unittest.main()
