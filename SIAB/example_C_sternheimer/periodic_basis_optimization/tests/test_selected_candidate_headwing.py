from pathlib import Path
import unittest


class SelectedCandidateHeadwingContractTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.validation = self.root / "ordinary_sos_validation"

    def test_headwing_producer_uses_selected_all_band_basis_and_pinned_pyatb(self):
        text = (
            self.validation / "run_selected_candidate_headwing_input.slurm"
        ).read_text(encoding="ascii")

        self.assertIn("#SBATCH --partition=p1", text)
        self.assertIn("#SBATCH --nodes=1", text)
        self.assertIn("#SBATCH --cpus-per-task=40", text)
        self.assertIn("#SBATCH --mem=190000", text)
        self.assertIn("C_gga_10au_100Ry_selected_product_pca.orb", text)
        self.assertIn('set_input_key nbands "$expected_ao_count"', text)
        self.assertIn("set_input_key symmetry 1", text)
        self.assertIn("set_input_key nx 24", text)
        self.assertIn("set_input_key out_mat_hs2 1", text)
        self.assertIn("set_input_key out_mat_r 1", text)
        self.assertIn("9fb9028c59b1dbaf9cf66965280961fc2225d9eb", text)
        self.assertIn("pyatb_librpa_df/velocity_matrix", text)
        self.assertIn("headwing_basis=$expected_ao_count", text)

    def test_qavg_consumer_runs_sos_and_delta_with_one_headwing_input(self):
        text = (
            self.validation
            / "run_selected_candidate_matched_headwing_qavg_d4810f73.slurm"
        ).read_text(encoding="ascii")

        self.assertIn("d4810f73aab20c36e69b1c353c945b77f40931c9", text)
        self.assertIn("ordinary-sos-selected-grid-full-bz-reader-fractional", text)
        self.assertIn("matched-delta-selected-grid-symmetry-reader", text)
        self.assertIn("pyatb_librpa_df", text)
        self.assertIn("replace_w_head = true", text)
        self.assertIn("option_dielect_func = 3", text)
        self.assertIn("rpa_headwing_mode = qavg", text)
        self.assertIn("rpa_headwing_body_start = 1", text)
        self.assertIn("sqrt_coulomb_threshold = 1e-5", text)
        self.assertIn("scope full_qavg_shared_selected_headwing", text)
        self.assertIn("headwing_convergence_gate pending", text)
        self.assertIn("basis_full_qavg_gate", text)


if __name__ == "__main__":
    unittest.main()
