import pathlib
import unittest


class ProductPcaFixedPrefixCapturePilotContractTest(unittest.TestCase):
    def test_pilot_pins_fixed_prefix_capture_and_parallel_cache(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        script = root / "run_pilot_product_pca_fixed_prefix_capture_ae3a98ca.slurm"
        text = script.read_text(encoding="ascii")

        self.assertIn("ae3a98ca62486c6a8a3df94186eed93a3012cbdb", text)
        self.assertIn("periodic-c-ae3a98ca", text)
        self.assertIn("--occupied-capture-reference fixed_prefix", text)
        self.assertIn("--block-cache-workers 8", text)
        self.assertIn("--max-steps 2", text)
        self.assertIn("fixed-prefix-capture-ae3a98ca-2g", text)


if __name__ == "__main__":
    unittest.main()
