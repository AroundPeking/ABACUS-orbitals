"""Static production contract for the compact H response-basis campaign."""

import json
from pathlib import Path
import unittest


SIAB_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = (
    SIAB_ROOT / "example_H_sternheimer" / "compact_response_selection"
)


class CompactResponseSelectionContractTest(unittest.TestCase):
    def test_predeclares_control_and_two_locality_lanes(self):
        expected = {
            "selection_config_tail_0p00.json": 0.0,
            "selection_config_tail_0p10.json": 0.1,
            "selection_config_tail_0p30.json": 0.3,
        }
        for name, weight in expected.items():
            value = json.loads((CAMPAIGN / name).read_text(encoding="utf-8"))
            self.assertEqual(value["selection_mode"], "ao_budget_frontier")
            self.assertEqual(value["max_ao_per_atom"], 48)
            self.assertEqual(value["max_l"], 4)
            self.assertNotIn("global_capture", value)
            self.assertNotIn("per_l_residual_limit", value)
            self.assertEqual(
                value["optimizer_loss"]["radial_tail_weight"], weight
            )
            self.assertEqual(
                value["optimizer_loss"]["radial_tail_radius"], 4.0
            )
            self.assertEqual(
                value["optimizer_loss"]["radial_tail_condition_limit"],
                1.0e10,
            )

    def test_runner_uses_full_normal_node_and_physical_targets_only(self):
        runner = (CAMPAIGN / "run_selection.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --partition=normal", runner)
        self.assertIn("#SBATCH --array=0-2", runner)
        self.assertIn("#SBATCH --nodes=1", runner)
        self.assertIn("#SBATCH --ntasks=1", runner)
        self.assertIn("#SBATCH --cpus-per-task=30", runner)
        self.assertIn("#SBATCH --mem=110610M", runner)
        self.assertIn("#SBATCH --time=1-00:00:00", runner)
        self.assertIn("OMP_NUM_THREADS", runner)
        self.assertIn("MKL_NUM_THREADS", runner)
        self.assertIn("OPENBLAS_NUM_THREADS", runner)
        self.assertIn("-m unittest discover -v", runner)
        self.assertIn("SOURCE_SHA256SUMS", runner)
        self.assertIn("H_TZDP_8au_ORBITAL_RESULTS.txt", runner)
        self.assertIn("--atom-target", runner)
        self.assertIn("--multicenter-target", runner)
        self.assertIn("max_ao_per_atom", runner)
        self.assertNotIn("producer_h2_fragment_ghost", runner)
        self.assertNotIn("ghost_target", runner)
        self.assertNotIn("#SBATCH --partition=debug", runner)


if __name__ == "__main__":
    unittest.main()
