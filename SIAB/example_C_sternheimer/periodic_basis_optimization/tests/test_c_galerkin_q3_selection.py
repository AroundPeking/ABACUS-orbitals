import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "galerkin_binding_workflow"
SCRIPT = WORKFLOW / "select_c_candidate_q3.py"
CONFIG = WORKFLOW / "c_diamond.json"


def load_module():
    spec = importlib.util.spec_from_file_location("select_c_candidate_q3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_candidate(label, digest, pi_error, trace_error, *, capture=0.99995, condition=6.0e5):
    return {
        "label": label,
        "coefficients_sha256": digest,
        "global_weighted_relative_pi_error": pi_error,
        "global_weighted_relative_trace_log_error": trace_error,
        "minimum_occupied_capture": capture,
        "maximum_overlap_condition": condition,
    }


class CGalerkinQ3SelectionTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.config = json.loads(CONFIG.read_text(encoding="ascii"))
        self.bank = {
            "format_version": 1,
            "status": "success",
            "candidate_bank_gate": "pass",
            "input_sha256": {"initial": "0" * 64},
            "candidates": [
                {
                    "name": "pareto_w0p25",
                    "orbital_sha256": "1" * 64,
                    "family_tradeoff_gate": "pass",
                },
                {
                    "name": "pareto_w0p50",
                    "orbital_sha256": "2" * 64,
                    "family_tradeoff_gate": "pass",
                },
                {
                    "name": "pareto_w0p75",
                    "orbital_sha256": "3" * 64,
                    "family_tradeoff_gate": "pass",
                },
            ],
        }
        self.comparison = {
            "format_version": 1,
            "scope": "Galerkin Pi and trace-log screening; independent SOS validation required",
            "occupied_capture_floor": self.config["occupied_capture_floor"],
            "datasets": [{"selected_iq": 43, "physics_hash": "4" * 64}],
            "candidates": [
                comparison_candidate("initial", "0" * 64, 0.20, 0.10),
                comparison_candidate("pareto_w0p25", "1" * 64, 0.19, 0.099),
                comparison_candidate("pareto_w0p50", "2" * 64, 0.18, 0.098),
                comparison_candidate("pareto_w0p75", "3" * 64, 0.17, 0.101),
            ],
        }

    def test_selects_best_candidate_that_improves_both_q3_metrics(self):
        result = self.module.select_q3_candidate(
            config=self.config,
            bank=self.bank,
            comparison=self.comparison,
        )

        self.assertEqual(result["gate"], "pass")
        self.assertEqual(result["selected_candidate"], "pareto_w0p50")
        records = {record["name"]: record for record in result["candidates"]}
        self.assertEqual(records["pareto_w0p50"]["gate"], "pass")
        self.assertEqual(records["pareto_w0p75"]["gate"], "fail")
        self.assertIn(
            "trace_log_error_not_improved",
            records["pareto_w0p75"]["failure_reasons"],
        )

    def test_rejects_comparison_with_wrong_candidate_hash(self):
        self.comparison["candidates"][2]["coefficients_sha256"] = "9" * 64

        with self.assertRaisesRegex(ValueError, "hash"):
            self.module.select_q3_candidate(
                config=self.config,
                bank=self.bank,
                comparison=self.comparison,
            )

    def test_rejects_non_q3_dataset(self):
        self.comparison["datasets"][0]["selected_iq"] = 22

        with self.assertRaisesRegex(ValueError, "q3"):
            self.module.select_q3_candidate(
                config=self.config,
                bank=self.bank,
                comparison=self.comparison,
            )

    def test_fails_gate_when_no_candidate_improves_both_metrics(self):
        for candidate in self.comparison["candidates"][1:]:
            candidate["global_weighted_relative_trace_log_error"] = 0.11

        result = self.module.select_q3_candidate(
            config=self.config,
            bank=self.bank,
            comparison=self.comparison,
        )

        self.assertEqual(result["gate"], "fail")
        self.assertIsNone(result["selected_candidate"])


if __name__ == "__main__":
    unittest.main()
