from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parents[1]
RUNNER = HERE / "run_c_atom_solid_balanced_one_g_df.slurm"
SUBMITTER = HERE / "submit_c_atom_solid_balanced_one_g_df.sh"


class CAtomSolidBalancedJobTest(unittest.TestCase):
    def test_runner_balances_atom_and_solid_families(self):
        text = RUNNER.read_text(encoding="ascii")
        self.assertIn("#SBATCH --partition=p1", text)
        self.assertIn("#SBATCH --nodes=1", text)
        self.assertIn("#SBATCH --cpus-per-task=40", text)
        self.assertIn("--nu 3,3,2,1,1", text)
        self.assertIn("--fixed-nu 2,2,1,0,0", text)
        self.assertEqual(text.count("--dataset-family C_solid"), 2)
        self.assertIn("--atomic-family C_atom", text)
        self.assertIn("--atomic-response", text)
        self.assertIn("--atomic-source", text)
        self.assertIn("initial_family_losses", text)
        self.assertIn("best_family_losses", text)
        self.assertIn("fixed_prefix", text)
        self.assertIn("source-only pair gate is not successful", text)

    def test_submitter_is_duplicate_safe_and_has_two_modes(self):
        text = SUBMITTER.read_text(encoding="ascii")
        self.assertIn("pilot|production", text)
        self.assertIn('test ! -e "$receipt"', text)
        self.assertIn('test ! -e "$run_root"', text)
        self.assertIn("sbatch --test-only", text)
        self.assertIn("sbatch --parsable", text)


if __name__ == "__main__":
    unittest.main()
