#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from c_sos_candidate_inventory import DEFAULT_CONTRACT, build_inventory, classify_candidate


def base_candidate():
    return {
        "status": "success",
        "profile": DEFAULT_CONTRACT["profile"],
        "nu": DEFAULT_CONTRACT["nu"],
        "original_coefficients_sha256": DEFAULT_CONTRACT["original_coefficients_sha256"],
        "optimized_coefficients_sha256": DEFAULT_CONTRACT[
            "primary_optimized_coefficients_sha256"
        ],
        "channel_alphas": DEFAULT_CONTRACT["primary_channel_alphas"],
        "direction": "original_plus_channel_alpha_times_optimized_minus_original",
    }


class CandidateClassificationTest(unittest.TestCase):
    def test_accepts_exact_origin(self):
        result = classify_candidate(base_candidate())
        self.assertTrue(result["in_current_subspace"])
        self.assertEqual(result["coordinates"], {})

    def test_rejects_different_primary_coordinates(self):
        candidate = base_candidate()
        candidate["channel_alphas"] = [0.0, -1.0, -1.5, 0.0, 0.0]
        result = classify_candidate(candidate)
        self.assertFalse(result["in_current_subspace"])
        self.assertEqual(result["reason"], "different_primary_coordinates")

    def test_accepts_supported_secondary_channel(self):
        candidate = base_candidate()
        candidate.update(
            {
                "direction": "original_plus_two_channel_resolved_directions",
                "secondary_optimized_coefficients_sha256": DEFAULT_CONTRACT[
                    "secondary_optimized_coefficients_sha256"
                ],
                "secondary_channel_alphas": [0.0, 0.25, 0.0, 0.0, 0.0],
            }
        )
        result = classify_candidate(candidate)
        self.assertTrue(result["in_current_subspace"])
        self.assertEqual(result["coordinates"], {"relaxed_p_all": 0.25})

    def test_rejects_secondary_s_channel(self):
        candidate = base_candidate()
        candidate.update(
            {
                "direction": "original_plus_two_channel_resolved_directions",
                "secondary_optimized_coefficients_sha256": DEFAULT_CONTRACT[
                    "secondary_optimized_coefficients_sha256"
                ],
                "secondary_channel_alphas": [0.25, 0.0, 0.0, 0.0, 0.0],
            }
        )
        result = classify_candidate(candidate)
        self.assertFalse(result["in_current_subspace"])
        self.assertEqual(result["reason"], "unsupported_secondary_channel")

    def test_accepts_supported_zeta_coordinates(self):
        candidate = base_candidate()
        candidate.update(
            {
                "direction": "original_plus_channel_and_zeta_resolved_directions",
                "secondary_optimized_coefficients_sha256": DEFAULT_CONTRACT[
                    "secondary_optimized_coefficients_sha256"
                ],
                "secondary_zeta_alphas": [
                    [0.0, 0.0, 0.25],
                    [0.0, 0.0, 0.25],
                    [0.0, 0.25],
                    [],
                    [],
                ],
            }
        )
        result = classify_candidate(candidate)
        self.assertTrue(result["in_current_subspace"])
        self.assertEqual(
            result["coordinates"],
            {"beta_s3": 0.25, "beta_p3": 0.25, "beta_d2": 0.25},
        )

    def test_rejects_different_secondary_direction_hash(self):
        candidate = base_candidate()
        candidate.update(
            {
                "direction": "original_plus_two_channel_resolved_directions",
                "secondary_optimized_coefficients_sha256": "0" * 64,
                "secondary_channel_alphas": [0.0, 0.25, 0.0, 0.0, 0.0],
            }
        )
        result = classify_candidate(candidate)
        self.assertFalse(result["in_current_subspace"])
        self.assertEqual(result["reason"], "different_secondary_direction")


class InventoryTest(unittest.TestCase):
    @staticmethod
    def _write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_reports_extra_completed_point_by_orbital_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            orbital_hash = "a" * 64
            candidate = base_candidate()
            candidate["orbital_sha256"] = orbital_hash
            self._write_json(root / "extra-candidate-test" / "CANDIDATE.json", candidate)
            self._write_json(
                root / "extra-pbe" / "pbe-gate" / "PBE_GATE.json",
                {
                    "status": "success",
                    "pbe_gate": "pass",
                    "selected_orbital_sha256": orbital_hash,
                },
            )
            self._write_json(
                root / "extra-sos" / "binding" / "RESULT.json",
                {"status": "success", "selected_orbital_sha256": orbital_hash},
            )
            trust_path = root / "trust" / "RESULT.json"
            self._write_json(
                trust_path,
                {
                    "candidates": [
                        {
                            "selected_orbital_sha256": "b" * 64,
                            "stability": "stable",
                            "coordinates": {"beta_d2": 0.25},
                        }
                    ],
                    "surrogate": {"model_gate": "fail"},
                },
            )

            result = build_inventory(root, trust_path)

            self.assertEqual(result["eligible_extra_stable_points"], ["extra-candidate-test"])
            self.assertFalse(result["strict_redundancy_blocker"])
            self.assertEqual(result["coordinate_nonzero_amplitudes"], {"beta_d2": [0.25]})


if __name__ == "__main__":
    unittest.main()
