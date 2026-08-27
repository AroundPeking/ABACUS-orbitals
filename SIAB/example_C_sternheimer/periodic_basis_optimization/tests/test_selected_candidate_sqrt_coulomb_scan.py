from pathlib import Path
import unittest


class SelectedCandidateSqrtCoulombScanTest(unittest.TestCase):
    def test_scan_changes_only_positive_coulomb_eigenvalue_threshold(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (
            root
            / "ordinary_sos_validation"
            / "run_selected_candidate_sqrt_coulomb_scan_d4810f73.slurm"
        )
        text = script.read_text(encoding="ascii")
        self.assertIn("#SBATCH --array=1-2", text)
        self.assertIn('thresholds=("1e-4" "1e-3")', text)
        self.assertIn("sqrt_coulomb_threshold = $threshold", text)
        self.assertIn("exx_pca_threshold=1e-4", text)
        self.assertIn("nfreq=6", text)
        self.assertIn("input_dir = $reader", text)
        self.assertIn("v1_coulomb_grid_iq_", text)
        self.assertIn("v1_Cs_data_", text)
        self.assertIn("use_symmetry_rpa", text)
        self.assertIn("replace_w_head", text)
        self.assertIn("sqrt_coulomb_threshold_scan", text)


if __name__ == "__main__":
    unittest.main()
