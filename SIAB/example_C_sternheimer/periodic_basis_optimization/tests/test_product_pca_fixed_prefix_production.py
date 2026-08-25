import pathlib
import unittest


class ProductPcaFixedPrefixProductionContractTest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(__file__).resolve().parents[1]

    def test_joint_optimization_pins_validated_layout_and_physics(self):
        script = self.root / (
            "run_optimize_product_pca_fixed_prefix_layout_c5b09b89.slurm"
        )
        text = script.read_text(encoding="ascii")

        self.assertIn("c5b09b892eff6bb38f98bc3659052f358c5f4c1b", text)
        self.assertIn("periodic-c-c5b09b89", text)
        self.assertIn("--occupied-capture-reference fixed_prefix", text)
        self.assertIn("--omitted-reference-projection-validation layout", text)
        self.assertIn("--block-cache-workers 8", text)
        self.assertIn("--fixed-nu 2,2,1,0,0", text)
        self.assertIn("--max-steps 1000", text)
        self.assertIn("--minimum-steps 200", text)
        self.assertIn("--plateau-patience 250", text)
        self.assertIn("python3 -u", text)
        self.assertNotIn(" +  ", text)
        self.assertIn("joint-fixed-prefix-layout-", text)
        self.assertIn("producer_jobs=3118459_1+3119622,3119900_2", text)

    def test_heldout_q3_uses_only_completed_joint_results(self):
        script = self.root / (
            "run_compare_product_pca_fixed_prefix_heldout_q3_c5b09b89.slurm"
        )
        text = script.read_text(encoding="ascii")

        self.assertIn("c5b09b892eff6bb38f98bc3659052f358c5f4c1b", text)
        self.assertIn("selected_iq 43", text)
        self.assertIn("producer_job=3119906_3", text)
        self.assertIn("joint-fixed-prefix-layout-2g", text)
        self.assertIn("joint-fixed-prefix-layout-3g", text)
        self.assertIn("heldout-q3-fixed-prefix-layout", text)
        self.assertNotIn(" +  ", text)


if __name__ == "__main__":
    unittest.main()
