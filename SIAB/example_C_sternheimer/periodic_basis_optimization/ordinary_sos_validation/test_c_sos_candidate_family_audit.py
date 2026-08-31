#!/usr/bin/env python3

import unittest

from c_sos_candidate_family_audit import (
    audit_low_order_models,
    candidate_family_key,
    extract_coordinates,
)


class CandidateFamilyTest(unittest.TestCase):
    def test_family_key_separates_direction_hashes(self):
        candidate = {
            "nu": [3, 3, 2, 0, 0],
            "profile": "interpolated_dzp",
            "original_coefficients_sha256": "a" * 64,
            "optimized_coefficients_sha256": "b" * 64,
            "secondary_optimized_coefficients_sha256": "c" * 64,
            "direction": "direction-a",
        }
        changed = dict(candidate)
        changed["secondary_optimized_coefficients_sha256"] = "d" * 64

        self.assertNotEqual(candidate_family_key(candidate), candidate_family_key(changed))

    def test_extracts_channel_and_zeta_coordinates(self):
        channel = {
            "direction": "original_plus_channel_alpha_times_optimized_minus_original",
            "channel_alphas": [0.5, -1.0, -1.5, 0.0, 0.0],
        }
        zeta = {
            "direction": "original_plus_channel_and_zeta_resolved_directions",
            "secondary_zeta_alphas": [
                [0.0, 0.0, 0.25],
                [0.0, 0.0, 0.5],
                [0.0, 0.25],
                [],
                [],
            ],
        }

        self.assertEqual(extract_coordinates(channel), {"alpha_s": 0.5, "alpha_p": -1.0, "alpha_d": -1.5})
        self.assertEqual(extract_coordinates(zeta), {"beta_s3": 0.25, "beta_p3": 0.5, "beta_d2": 0.25})


class LowOrderModelTest(unittest.TestCase):
    def test_accepts_redundant_exact_linear_family(self):
        rows = [
            {
                "name": f"p{index}",
                "coordinates": {"x": float(index)},
                "stability": "stable",
                "sos_error_ev_per_c": 1.0 + 2.0 * index,
            }
            for index in range(4)
        ]

        result = audit_low_order_models(rows, maximum_loo_mae_ev_per_c=1.0e-10)

        self.assertGreaterEqual(result["validated_model_count"], 1)
        self.assertEqual(result["model_gate"], "pass")

    def test_rejects_known_primary_family_ordering(self):
        reference = 6.902326
        values = [
            ("d-1.5", 0.0, 0.0, -1.5, 7.271557367754378),
            ("d-2.0", 0.0, 0.0, -2.0, 7.287566672396896),
            ("p-1", 0.0, -1.0, -1.5, 7.266547224614225),
            ("s-1", -1.0, 0.0, -1.5, 7.312039542371284),
            ("s0.5-p-1", 0.5, -1.0, -1.5, 7.206086206826756),
        ]
        rows = [
            {
                "name": name,
                "coordinates": {"alpha_s": s_value, "alpha_p": p_value, "alpha_d": d_value},
                "stability": "stable",
                "sos_error_ev_per_c": energy - reference,
            }
            for name, s_value, p_value, d_value, energy in values
        ]

        result = audit_low_order_models(rows, maximum_loo_mae_ev_per_c=0.05)

        self.assertEqual(result["validated_model_count"], 0)
        self.assertEqual(result["model_gate"], "fail")
        self.assertIn("no_leave_one_out_valid_model", result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
