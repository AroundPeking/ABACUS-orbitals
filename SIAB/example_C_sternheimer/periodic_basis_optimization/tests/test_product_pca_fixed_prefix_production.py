import pathlib
import unittest


class ProductPcaFixedPrefixProductionContractTest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(__file__).resolve().parents[1]

    def test_joint_optimization_pins_validated_layout_and_physics(self):
        script = self.root / (
            "run_optimize_product_pca_fixed_prefix_checkpoint_06f61e4c.slurm"
        )
        text = script.read_text(encoding="ascii")

        self.assertIn("06f61e4c5d8aa514a391b268206d99a2e70c6075", text)
        self.assertIn("periodic-c-06f61e4c", text)
        self.assertIn("--occupied-capture-reference fixed_prefix", text)
        self.assertIn("--omitted-reference-projection-validation layout", text)
        self.assertIn("--block-cache-workers 8", text)
        self.assertIn("--fixed-nu 2,2,1,0,0", text)
        self.assertIn("--max-steps 500", text)
        self.assertIn("--minimum-steps 200", text)
        self.assertIn("--plateau-patience 250", text)
        self.assertIn("python3 -u", text)
        self.assertNotIn(" +  ", text)
        self.assertIn("joint-fixed-prefix-layout-checkpoint-500-", text)
        self.assertIn("BEST_ORBITAL_CHECKPOINT.txt", text)
        self.assertIn("BEST_CHECKPOINT.json", text)
        self.assertIn("producer_jobs=3118459_1+3119622,3119900_2", text)

    def test_heldout_q3_uses_only_completed_joint_results(self):
        script = self.root / (
            "run_compare_product_pca_fixed_prefix_heldout_q3_06f61e4c.slurm"
        )
        text = script.read_text(encoding="ascii")

        self.assertIn("06f61e4c5d8aa514a391b268206d99a2e70c6075", text)
        self.assertIn("SIAB_SOURCE_ROOT", text)
        self.assertIn("SIAB_SOURCE_COMMIT", text)
        self.assertIn("optimizer_siab_commit", text)
        self.assertIn("comparison_source_commit", text)
        self.assertIn("selected_iq 43", text)
        self.assertIn("producer_job=3119906_3", text)
        self.assertIn("joint-fixed-prefix-layout-checkpoint-500-2g", text)
        self.assertIn("joint-fixed-prefix-layout-checkpoint-500-3g", text)
        self.assertIn("heldout-q3-fixed-prefix-layout-checkpoint-500", text)
        self.assertIn("BEST_ORBITAL_CHECKPOINT.txt", text)
        self.assertIn(
            "--occupied-capture-floor 0.9998982409775239", text
        )
        self.assertNotIn(" +  ", text)


if __name__ == "__main__":
    unittest.main()
