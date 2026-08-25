import pathlib
import unittest


HERE = pathlib.Path(__file__).resolve().parent
CAMPAIGN = HERE.parent


class ProductPcaJobProvenanceTest(unittest.TestCase):
    def test_compute_jobs_do_not_require_git_executable(self):
        scripts = (
            "run_optimize_product_pca_nfreq6_q1_q2_efc24c2c.slurm",
            "run_compare_product_pca_heldout_q3_efc24c2c.slurm",
            "export_product_pca_candidates_efc24c2c.slurm",
        )
        for script in scripts:
            text = (CAMPAIGN / script).read_text(encoding="ascii")
            self.assertNotIn("git -C", text)
            self.assertIn('test "$(cat "$code/.git/HEAD")" = "$siab_commit"', text)


if __name__ == "__main__":
    unittest.main()
