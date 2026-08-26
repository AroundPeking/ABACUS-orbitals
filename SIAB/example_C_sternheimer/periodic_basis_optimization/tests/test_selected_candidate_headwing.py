import importlib.util
from pathlib import Path
import tempfile
import unittest


GET_DIEL = (
    Path(__file__).resolve().parents[1]
    / "ordinary_sos_validation"
    / "get_diel_selected_c_headwing.py"
)
SPEC = importlib.util.spec_from_file_location("get_diel_selected_c_headwing", GET_DIEL)
GET_DIEL_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GET_DIEL_MODULE)


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
        self.assertIn('cat "$pyatb_source/.git/HEAD"', text)
        self.assertNotIn('git -C "$pyatb_source"', text)
        self.assertIn('cp "$get_diel" "$work/get_diel.py"', text)
        self.assertIn('cp "$output_librpa" "$work/output_librpa.py"', text)
        self.assertNotIn('ln -s "$get_diel"', text)
        self.assertNotIn('ln -s "$output_librpa"', text)
        self.assertIn("pyatb_librpa_df/velocity_matrix", text)
        self.assertIn("headwing_basis=$expected_ao_count", text)
        get_diel = (self.validation / "get_diel_selected_c_headwing.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("OUT.ABACUS/eig_occ.txt", get_diel)
        self.assertNotIn("open(f_band", get_diel)

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

    def test_reads_consistent_occupied_band_count_from_abacus_eig_occ(self):
        text = """1 # ionic step
Spin number 1
spin=1 k-point=1/2 Cartesian=0 0 0
1 -5.0 0.125
2 -1.0 0.125
3 2.0 0.0
spin=1 k-point=2/2 Cartesian=0.5 0.5 0.5
1 -4.0 1.0
2 -0.5 1.0
3 2.5 0.0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eig_occ.txt"
            path.write_text(text, encoding="ascii")
            self.assertEqual(
                GET_DIEL_MODULE.read_occupied_band_count(path),
                2,
            )

            path.write_text(
                text.replace("2 -0.5 1.0", "2 -0.5 0.0"),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "same occupied band count"):
                GET_DIEL_MODULE.read_occupied_band_count(path)


if __name__ == "__main__":
    unittest.main()
