from pathlib import Path
import unittest


class SelectedCandidateFullBzSosContractTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.validation = self.root / "ordinary_sos_validation"

    def test_grid_producer_uses_selected_all_band_candidate_and_full_bz(self):
        text = (
            self.validation / "run_selected_candidate_full_bz_grid_coulomb.slurm"
        ).read_text(encoding="ascii")

        self.assertIn("#SBATCH --array=1-64%8", text)
        self.assertIn("SELECTED_CANDIDATE.json", text)
        self.assertIn("C_gga_10au_100Ry_selected_product_pca.orb", text)
        self.assertIn("expected_ao_count", text)
        self.assertIn("set_input_key nbands \"$expected_ao_count\"", text)
        self.assertIn("set_input_key exx_pca_threshold 1e-4", text)
        self.assertIn("set_input_key nx 24", text)
        self.assertIn("set_input_key symmetry -1", text)
        self.assertIn("ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1", text)
        self.assertIn("exact_rhs_full_periodic_poisson", text)
        self.assertNotIn("naux == 236", text)

    def test_login_export_closes_selection_with_exported_orbital_hash(self):
        text = (
            self.root / "export_selected_product_pca_candidate.sh"
        ).read_text(encoding="ascii")

        self.assertIn("select_periodic_candidate.py", text)
        self.assertIn("export_periodic_orbitals.py", text)
        self.assertIn("heldout-q3-fixed-prefix-layout-checkpoint-500", text)
        self.assertIn("--occupied-capture-floor 0.9998982409775239", text)
        self.assertIn("exported_orbital_sha256", text)
        self.assertIn("C_gga_10au_100Ry_selected_product_pca.orb", text)
        self.assertNotIn("#SBATCH", text)

    def test_rpa_consumer_uses_fractional_reader_fix_and_all_64_grid_matrices(self):
        text = (
            self.validation / "run_selected_candidate_full_bz_reader_d4810f73.slurm"
        ).read_text(encoding="ascii")

        self.assertIn("d4810f73aab20c36e69b1c353c945b77f40931c9", text)
        self.assertIn("SIAB_SOURCE_ROOT:?missing exact SIAB source deployment", text)
        self.assertIn("SELECTED_CANDIDATE.json", text)
        self.assertIn("for ordinal in $(seq 1 64)", text)
        self.assertIn('test "$grid_coulomb_file_count" -eq 64', text)
        self.assertIn("bz_coordinate_source=fractional_columns", text)
        self.assertIn("analytic_headwing=no_body_screen", text)
        self.assertIn("extract_librpa_frequency_grid.py", text)
        self.assertIn("/data/home/df_iopcas_ghj/app/miniconda3/bin/python", text)
        self.assertIn('"$frequency_python" "$frequency_extractor"', text)
        self.assertIn("SELECTED_SOS_FREQUENCY_GRID.dat", text)
        self.assertIn("SELECTED_SOS_FREQUENCY_GRID.json", text)
        self.assertIn("frequency_grid_source=selected_candidate_sos", text)
        self.assertNotIn("-0.501460253", text)


if __name__ == "__main__":
    unittest.main()
