import pathlib
import unittest


class ProductPcaBlockCacheBenchmarkContractTest(unittest.TestCase):
    def test_benchmark_pins_cached_optimizer_and_short_trajectory(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        script = root / "run_benchmark_product_pca_block_cache_de137884.slurm"
        text = script.read_text(encoding="ascii")

        self.assertIn("de137884c4796a2d21916fcd9caad3305c9d8ee3", text)
        self.assertIn("periodic-c-de137884", text)
        self.assertIn("--max-steps 2", text)
        self.assertIn("--minimum-steps 0", text)
        self.assertIn("block-cache-de137884-2g", text)
        self.assertIn("OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK", text)
        self.assertIn("MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK", text)


if __name__ == "__main__":
    unittest.main()
