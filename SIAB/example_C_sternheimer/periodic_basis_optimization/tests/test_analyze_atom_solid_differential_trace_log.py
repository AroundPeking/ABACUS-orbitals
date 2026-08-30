import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analyze_atom_solid_differential_trace_log.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_atom_solid_differential_trace_log", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalyzeAtomSolidDifferentialTraceLogTest(unittest.TestCase):
    def test_builds_raw_star_partial_and_star_normalized_proxies(self):
        proxies = MODULE.build_differential_proxies(
            atom_difference=0.20,
            solid_differences={1: 0.40, 22: 0.20, 43: -0.10},
            q_weights={1: 1.0 / 64.0, 22: 8.0 / 64.0, 43: 4.0 / 64.0},
            star_multiplicities={1: 1, 22: 8, 43: 4},
            full_q_count=64,
        )

        raw_solid = (0.40 + 8.0 * 0.20 - 4.0 * 0.10) / 64.0
        star_partial_solid = (0.40 + 8.0 * 0.20 - 4.0 * 0.10) / 64.0
        star_normalized_solid = (0.40 + 8.0 * 0.20 - 4.0 * 0.10) / 13.0
        self.assertAlmostEqual(proxies["raw_q_weight"]["solid_difference"], raw_solid)
        self.assertAlmostEqual(
            proxies["star_partial"]["solid_difference"], star_partial_solid
        )
        self.assertAlmostEqual(
            proxies["star_normalized_extrapolation"]["solid_difference"],
            star_normalized_solid,
        )
        for name, solid in (
            ("raw_q_weight", raw_solid),
            ("star_partial", star_partial_solid),
            ("star_normalized_extrapolation", star_normalized_solid),
        ):
            self.assertAlmostEqual(
                proxies[name]["atom_minus_half_solid_difference"],
                0.20 - 0.5 * solid,
            )

    def test_rejects_incomplete_or_inconsistent_q_metadata(self):
        common = {
            "atom_difference": 0.0,
            "solid_differences": {1: 0.0, 2: 0.0},
            "q_weights": {1: 1.0 / 64.0, 2: 1.0 / 64.0},
            "star_multiplicities": {1: 1, 2: 8},
            "full_q_count": 64,
        }
        with self.assertRaisesRegex(ValueError, "same q indices"):
            MODULE.build_differential_proxies(
                **{**common, "star_multiplicities": {1: 1}}
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            MODULE.build_differential_proxies(
                **{**common, "q_weights": {1: 1.0 / 64.0, 2: 0.0}}
            )
        with self.assertRaisesRegex(ValueError, "full q count"):
            MODULE.build_differential_proxies(**{**common, "full_q_count": 8})

    def test_scores_proxy_order_against_known_sos_errors(self):
        candidates = [
            {
                "label": "original",
                "known_sos_binding_error_ev": 0.87,
                "proxies": {
                    "star_normalized_extrapolation": {
                        "atom_minus_half_solid_difference": 0.10
                    }
                },
            },
            {
                "label": "fixed500",
                "known_sos_binding_error_ev": 1.15,
                "proxies": {
                    "star_normalized_extrapolation": {
                        "atom_minus_half_solid_difference": -0.20
                    }
                },
            },
            {
                "label": "relaxed",
                "known_sos_binding_error_ev": 1.28,
                "proxies": {
                    "star_normalized_extrapolation": {
                        "atom_minus_half_solid_difference": 0.30
                    }
                },
            },
            {
                "label": "early",
                "known_sos_binding_error_ev": None,
                "proxies": {
                    "star_normalized_extrapolation": {
                        "atom_minus_half_solid_difference": 0.01
                    }
                },
            },
        ]

        score = MODULE.score_proxy_order(
            candidates, "star_normalized_extrapolation"
        )

        self.assertEqual(score["candidate_count"], 3)
        self.assertEqual(score["pair_count"], 3)
        self.assertEqual(score["concordant_pairs"], 3)
        self.assertEqual(score["discordant_pairs"], 0)
        self.assertEqual(score["agreement_fraction"], 1.0)
        self.assertEqual(
            score["proxy_order_small_to_large"],
            ["original", "fixed500", "relaxed"],
        )

    def test_marks_ties_as_unresolved(self):
        candidates = [
            {
                "label": "a",
                "known_sos_binding_error_ev": 1.0,
                "proxies": {"raw_q_weight": {"atom_minus_half_solid_difference": 0.2}},
            },
            {
                "label": "b",
                "known_sos_binding_error_ev": 2.0,
                "proxies": {"raw_q_weight": {"atom_minus_half_solid_difference": -0.2}},
            },
        ]

        score = MODULE.score_proxy_order(candidates, "raw_q_weight")

        self.assertEqual(score["pair_count"], 1)
        self.assertEqual(score["unresolved_pairs"], 1)
        self.assertIsNone(score["agreement_fraction"])


if __name__ == "__main__":
    unittest.main()
