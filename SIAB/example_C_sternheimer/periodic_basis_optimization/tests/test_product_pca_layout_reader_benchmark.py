import pathlib
import unittest


class ProductPcaLayoutReaderBenchmarkContractTest(unittest.TestCase):
    def test_benchmark_pins_layout_reader_and_fixed_prefix_reference(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        script = root / "run_benchmark_product_pca_layout_reader_c5b09b89.slurm"
        text = script.read_text(encoding="ascii")

        self.assertIn("c5b09b892eff6bb38f98bc3659052f358c5f4c1b", text)
        self.assertIn("periodic-c-c5b09b89", text)
        self.assertIn("--omitted-reference-projection-validation layout", text)
        self.assertIn("--occupied-capture-reference fixed_prefix", text)
        self.assertIn("--block-cache-workers 8", text)
        self.assertIn("--max-steps 2", text)
        self.assertIn("fixed-prefix-capture-ae3a98ca-2g", text)
        self.assertIn("layout-reader-c5b09b89-2g", text)
        self.assertIn("ORBITAL_RESULTS.txt", text)


if __name__ == "__main__":
    unittest.main()
