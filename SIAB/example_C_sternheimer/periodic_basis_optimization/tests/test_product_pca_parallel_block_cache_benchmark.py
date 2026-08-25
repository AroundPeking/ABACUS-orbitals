import pathlib
import unittest


class ProductPcaParallelBlockCacheBenchmarkContractTest(unittest.TestCase):
    def test_benchmark_pins_parallel_cache_optimizer_and_short_trajectory(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        script = root / "run_benchmark_product_pca_parallel_block_cache_efff82ac.slurm"
        text = script.read_text(encoding="ascii")

        self.assertIn("efff82ac0ea893e59dfa1e5518e899875fbb405c", text)
        self.assertIn("periodic-c-efff82ac", text)
        self.assertIn("--max-steps 2", text)
        self.assertIn("--minimum-steps 0", text)
        self.assertIn("--block-cache-workers 8", text)
        self.assertIn("parallel-block-cache-efff82ac-2g", text)
        self.assertIn("OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK", text)
        self.assertIn("MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK", text)


if __name__ == "__main__":
    unittest.main()
