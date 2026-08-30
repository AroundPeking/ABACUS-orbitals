import hashlib
import json
import pathlib
import tempfile
import unittest

from prepare_relaxed_dzp_candidate import prepare_candidate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareRelaxedDzpCandidateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.coefficients = self.root / "ORBITAL_RESULTS.txt"
        self.coefficients.write_text("coefficients\n", encoding="ascii")
        self.orbital = self.root / "candidate.orb"
        self.orbital.write_text(
            "Energy Cutoff(Ry)           100.0\n"
            "Radius Cutoff(a.u.)         10.0\n"
            "Lmax                        2\n"
            "Number of Sorbital-->       3\n"
            "Number of Porbital-->       3\n"
            "Number of Dorbital-->       2\n",
            encoding="ascii",
        )
        sha = digest(self.coefficients)
        optimizer = {
            "format_version": 1,
            "nu": [3, 3, 2, 0, 0],
            "fixed_nu": [1, 1, 0, 0, 0],
            "occupied_capture_reference": "initial_candidate",
            "initial_family_losses": {"C_atom": 0.2, "C_solid": 0.1},
            "best_family_losses": {"C_atom": 0.1, "C_solid": 0.05},
            "output_coefficients": str(self.coefficients),
            "output_coefficients_sha256": sha,
            "best_checkpoint_sha256": sha,
        }
        self.optimizer = self.root / "OPTIMIZATION_RESULT.json"
        self.optimizer.write_text(json.dumps(optimizer), encoding="ascii")
        comparison = {
            "format_version": 1,
            "datasets": [{"selected_iq": 43}],
            "candidates": [
                {
                    "label": "original-tzdp",
                    "global_weighted_relative_pi_error": 0.2,
                    "global_weighted_relative_trace_log_error": 0.1,
                },
                {
                    "label": "relaxed-dzp",
                    "nu": [3, 3, 2, 0, 0],
                    "ao_count_cell": 44,
                    "coefficients_sha256": sha,
                    "global_weighted_relative_pi_error": 0.15,
                    "global_weighted_relative_trace_log_error": 0.08,
                    "minimum_occupied_capture": 0.99999,
                },
            ],
        }
        self.comparison = self.root / "COMPARISON_RESULT.json"
        self.comparison.write_text(json.dumps(comparison), encoding="ascii")
        self.original_spectrum = self.root / "ORIGINAL_SPECTRUM.json"
        self.candidate_spectrum = self.root / "CANDIDATE_SPECTRUM.json"
        self.original_spectrum.write_text(
            json.dumps({"format_version": 1, "label": "original-tzdp", "nu": [3, 3, 2, 0, 0], "ao_count_cell": 44, "maximum_overlap_condition": 10.0, "maximum_eigenvalue_ev": 100.0}),
            encoding="ascii",
        )
        self.candidate_spectrum.write_text(
            json.dumps({"format_version": 1, "label": "relaxed-dzp", "nu": [3, 3, 2, 0, 0], "ao_count_cell": 44, "maximum_overlap_condition": 20.0, "maximum_eigenvalue_ev": 120.0}),
            encoding="ascii",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        return prepare_candidate(
            optimizer_result_path=self.optimizer,
            comparison_path=self.comparison,
            original_spectrum_path=self.original_spectrum,
            candidate_spectrum_path=self.candidate_spectrum,
            orbital_path=self.orbital,
            output_directory=self.root / "selected",
        )

    def test_stages_candidate_with_pbe_gate_contract(self):
        result = self.prepare()
        self.assertEqual(result["profile"], "relaxed_dzp")
        self.assertEqual(result["ao_count_atom"], 22)
        self.assertEqual(result["pre_pbe_gate"], "pass")
        self.assertEqual(result["pbe_tolerance_ev"], 0.010)
        self.assertTrue((self.root / "selected" / "CANDIDATE.json").is_file())

    def test_rejects_q3_regression(self):
        payload = json.loads(self.comparison.read_text(encoding="ascii"))
        payload["candidates"][1]["global_weighted_relative_pi_error"] = 0.21
        self.comparison.write_text(json.dumps(payload), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "does not improve q3"):
            self.prepare()

    def test_rejects_spectrum_instability(self):
        payload = json.loads(self.candidate_spectrum.read_text(encoding="ascii"))
        payload["maximum_overlap_condition"] = 30.0
        self.candidate_spectrum.write_text(json.dumps(payload), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "overlap or virtual-spectrum"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
