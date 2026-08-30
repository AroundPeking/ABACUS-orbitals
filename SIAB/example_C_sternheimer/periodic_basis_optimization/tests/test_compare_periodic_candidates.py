import importlib.util
import math
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "compare_periodic_candidates.py"
SPEC = importlib.util.spec_from_file_location("compare_periodic_candidates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComparePeriodicCandidatesTest(unittest.TestCase):
    def test_parses_candidate_definition(self):
        label, path, nu = MODULE.parse_candidate("two-g:/tmp/C.txt:3,3,2,1,2")
        self.assertEqual(label, "two-g")
        self.assertEqual(path, Path("/tmp/C.txt"))
        self.assertEqual(nu, (3, 3, 2, 1, 2))

        with self.assertRaisesRegex(ValueError, "label:path:nu"):
            MODULE.parse_candidate("missing-fields")

    def test_trace_log_matches_diagonal_reference(self):
        response = torch.diag(
            torch.tensor([-0.5, -1.0], dtype=torch.complex128)
        )
        expected = math.log(1.5) - 0.5 + math.log(2.0) - 1.0
        self.assertAlmostEqual(MODULE.trace_log_value(response), expected, places=14)

        invalid = torch.diag(torch.tensor([1.25], dtype=torch.complex128))
        with self.assertRaisesRegex(
            RuntimeError,
            r"candidate trace-log argument is not positive.*"
            r"minimum_argument=-0.25.*maximum_eigenvalue=1.25",
        ):
            MODULE.trace_log_value(invalid, name="candidate")

    def test_response_metrics_validates_reference_before_candidate(self):
        invalid_reference = torch.tensor(
            [[[1.50 + 0.0j]]], dtype=torch.complex128
        )
        invalid_candidate = torch.tensor(
            [[[2.00 + 0.0j]]], dtype=torch.complex128
        )
        weights = torch.tensor([1.0], dtype=torch.float64)

        with self.assertRaisesRegex(
            RuntimeError,
            r"reference frequency 0 trace-log argument is not positive.*"
            r"maximum_eigenvalue=1.5",
        ):
            MODULE.response_metrics(
                invalid_candidate,
                invalid_reference,
                weights,
            )

    def test_reports_each_frequency_and_weighted_pi_error(self):
        reference = torch.tensor(
            [[[-2.0 + 0.0j]], [[-1.0 + 0.0j]]], dtype=torch.complex128
        )
        candidate = torch.tensor(
            [[[-1.0 + 0.0j]], [[-1.0 + 0.0j]]], dtype=torch.complex128
        )
        weights = torch.tensor([1.0, 3.0], dtype=torch.float64)

        metrics = MODULE.response_metrics(candidate, reference, weights)

        self.assertEqual(metrics["relative_pi_error_by_frequency"], [0.5, 0.0])
        self.assertAlmostEqual(
            metrics["weighted_relative_pi_error"], math.sqrt(1.0 / 7.0), places=14
        )
        self.assertEqual(len(metrics["trace_log_candidate"]), 2)
        self.assertEqual(len(metrics["trace_log_reference"]), 2)

    def test_validates_explicit_occupied_capture_floor(self):
        self.assertEqual(MODULE.validate_occupied_capture_floor(0.9998), 0.9998)

        for invalid in (0.0, -0.1, 1.0, 1.1, math.inf, math.nan, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "occupied capture floor"):
                    MODULE.validate_occupied_capture_floor(invalid)


if __name__ == "__main__":
    unittest.main()
