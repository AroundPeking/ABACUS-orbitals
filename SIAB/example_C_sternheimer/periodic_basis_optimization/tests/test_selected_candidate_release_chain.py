from pathlib import Path
import unittest


class SelectedCandidateReleaseChainContractTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.validation = self.root / "ordinary_sos_validation"

    def test_initial_release_requires_heldout_then_submits_each_unique_stage(self):
        text = (
            self.validation
            / "release_selected_candidate_validation_after_heldout.sh"
        ).read_text(encoding="ascii")

        self.assertIn("OPTIMIZER_ARRAY_JOB_ID", text)
        self.assertIn("HELDOUT_JOB_ID", text)
        self.assertIn("COMPLETED", text)
        self.assertIn("COMPARISON_RESULT.json", text)
        self.assertIn("export_selected_product_pca_candidate.sh", text)
        self.assertIn("SELECTED_VALIDATION_CHAIN.txt", text)
        self.assertIn(".release.lock", text)
        self.assertIn("--array=1-64%8", text)
        self.assertIn("run_selected_candidate_full_bz_grid_coulomb.slurm", text)
        self.assertIn("run_selected_candidate_full_bz_reader_d4810f73.slurm", text)
        self.assertIn("run_selected_candidate_headwing_input.slurm", text)
        self.assertIn("run_selected_candidate_matched_delta_response.slurm", text)
        self.assertIn("--dependency=afterok:", text)
        self.assertIn("GRID_COULOMB_ARRAY_JOB_ID", text)
        self.assertIn("SELECTED_SOS_JOB_ID", text)

    def test_delta_release_submits_full_qavg_after_body_and_headwing(self):
        text = (
            self.validation
            / "release_selected_candidate_matched_delta_after_q1.sh"
        ).read_text(encoding="ascii")

        self.assertIn("HEADWING_JOB_ID", text)
        self.assertIn(
            "run_selected_candidate_matched_headwing_qavg_d4810f73.slurm",
            text,
        )
        self.assertIn("headwing_job=$headwing_job", text)
        self.assertIn("qavg_job=$qavg_job", text)
        self.assertIn('afterok:"$consumer_job":"$headwing_job"', text)
        self.assertIn("QAVG_TEST_ONLY.txt", text)


if __name__ == "__main__":
    unittest.main()
