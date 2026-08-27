from pathlib import Path
import unittest


class SelectedCandidateHighLMetricControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.script = (
            self.root
            / "ordinary_sos_validation"
            / "run_selected_candidate_highl_metric_control.slurm"
        )

    def test_control_changes_only_the_high_l_ewald_implementation(self) -> None:
        text = self.script.read_text(encoding="ascii")
        self.assertIn("dbb7671896910155bd9b4983ce72ca7caa451f19", text)
        self.assertIn("d05b3088acd5ec170f62027c35de69c22205dba82ac6799eb4afe877c8943708", text)
        self.assertIn("exx_pca_threshold 1e-4", text)
        self.assertIn("nbands 112", text)
        self.assertIn("symmetry -1", text)
        self.assertIn("nx 24", text)
        self.assertIn("ny 24", text)
        self.assertIn("nz 24", text)
        self.assertIn("remove_input_key out_sternheimer_basis_opt", text)
        self.assertNotIn("set_input_key out_sternheimer_basis_opt", text)
        self.assertIn("out_sternheimer_librpa 0", text)
        self.assertNotIn("ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY", text)
        self.assertIn("v1_coulomb_full_iq_1_rank0.dat", text)
        self.assertIn("v1_Cs_data_0.txt", text)
        self.assertIn("HIGH_L_METRIC_CONTROL.json", text)


if __name__ == "__main__":
    unittest.main()
