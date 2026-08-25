import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "select_periodic_candidate.py"
SPEC = importlib.util.spec_from_file_location("select_periodic_candidate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(label, trace_error, pi_error, ao_count, capture=0.999999):
    return {
        "label": label,
        "global_weighted_relative_trace_log_error": trace_error,
        "global_weighted_relative_pi_error": pi_error,
        "ao_count_cell": ao_count,
        "minimum_occupied_capture": capture,
        "coefficients": "/tmp/" + label + ".txt",
        "coefficients_sha256": "a" * 64,
        "nu": [3, 3, 2, 1, 2 if label == "joint-two-g" else 3],
    }


class SelectPeriodicCandidateTest(unittest.TestCase):
    def test_accuracy_precedes_basis_size(self):
        report = {
            "format_version": 1,
            "datasets": [{"selected_iq": 43}],
            "candidates": [
                candidate("joint-two-g", 0.08, 0.07, 94),
                candidate("joint-three-g", 0.06, 0.09, 112),
            ],
        }

        selected = MODULE.select_candidate(
            report,
            allowed_labels=("joint-two-g", "joint-three-g"),
            occupied_capture_floor=0.999898,
        )

        self.assertEqual(selected["label"], "joint-three-g")
        self.assertEqual(
            selected["selection_order"],
            [
                "global_weighted_relative_trace_log_error",
                "global_weighted_relative_pi_error",
                "ao_count_cell",
            ],
        )

    def test_pi_error_then_ao_count_break_exact_trace_log_ties(self):
        report = {
            "format_version": 1,
            "datasets": [{"selected_iq": 43}],
            "candidates": [
                candidate("joint-two-g", 0.06, 0.05, 94),
                candidate("joint-three-g", 0.06, 0.04, 112),
            ],
        }
        selected = MODULE.select_candidate(
            report,
            allowed_labels=("joint-two-g", "joint-three-g"),
            occupied_capture_floor=0.999898,
        )
        self.assertEqual(selected["label"], "joint-three-g")

        report["candidates"][0]["global_weighted_relative_pi_error"] = 0.04
        selected = MODULE.select_candidate(
            report,
            allowed_labels=("joint-two-g", "joint-three-g"),
            occupied_capture_floor=0.999898,
        )
        self.assertEqual(selected["label"], "joint-two-g")

    def test_rejects_missing_or_under_capture_joint_candidate(self):
        incomplete = {
            "format_version": 1,
            "datasets": [{"selected_iq": 43}],
            "candidates": [candidate("joint-two-g", 0.08, 0.07, 94)],
        }
        with self.assertRaisesRegex(ValueError, "exactly the allowed"):
            MODULE.select_candidate(
                incomplete,
                allowed_labels=("joint-two-g", "joint-three-g"),
                occupied_capture_floor=0.999898,
            )

        incomplete["candidates"].append(
            candidate("joint-three-g", 0.06, 0.04, 112, capture=0.99)
        )
        with self.assertRaisesRegex(ValueError, "occupied capture"):
            MODULE.select_candidate(
                incomplete,
                allowed_labels=("joint-two-g", "joint-three-g"),
                occupied_capture_floor=0.999898,
            )


if __name__ == "__main__":
    unittest.main()
