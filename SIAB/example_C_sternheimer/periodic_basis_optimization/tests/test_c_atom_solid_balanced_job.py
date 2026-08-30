from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parents[1]
RUNNER = HERE / "run_c_atom_solid_balanced_one_g_df.slurm"
SUBMITTER = HERE / "submit_c_atom_solid_balanced_one_g_df.sh"
PROJECTED_PI = HERE.parents[1] / "opt_orb_pytorch_dpsi" / "projected_pi.py"


class CAtomSolidBalancedJobTest(unittest.TestCase):
    def test_runner_balances_atom_and_solid_families(self):
        text = RUNNER.read_text(encoding="ascii")
        self.assertIn("#SBATCH --partition=p1", text)
        self.assertIn("#SBATCH --nodes=1", text)
        self.assertIn("#SBATCH --cpus-per-task=40", text)
        self.assertIn("module load python/3.9.22", text)
        self.assertIn("/data/home/df_iopcas_ghj/app/python/siab-torch19-py39", text)
        self.assertIn("torch.optim.Adam", text)
        self.assertIn("runpy.run_path", text)
        self.assertIn("optimizer_nu=3,3,2,1,1", text)
        self.assertIn("optimizer_fixed_nu=2,2,1,0,0", text)
        self.assertEqual(text.count("--dataset-family C_solid"), 2)
        self.assertIn("--atomic-family C_atom", text)
        self.assertIn("--atomic-response", text)
        self.assertIn("--atomic-source", text)
        self.assertIn("initial_family_losses", text)
        self.assertIn("best_family_losses", text)
        self.assertIn("fixed_prefix", text)
        self.assertIn("source-only pair gate is not successful", text)

    def test_runner_supports_direct_no_f_optimization(self):
        text = RUNNER.read_text(encoding="ascii")
        self.assertIn("CANDIDATE_PROFILE", text)
        self.assertIn("no_f)", text)
        self.assertIn("candidate_label=3s3p2d", text)
        self.assertIn("optimizer_nu=3,3,2,0,0", text)
        self.assertIn("optimizer_fixed_nu=2,2,1,0,0", text)
        self.assertIn("optimizer_max_l=4", text)
        self.assertIn("truncate_periodic_coefficients.py", text)
        self.assertIn("--preserve-channel-layout", text)
        self.assertIn("#SBATCH --time=1-00:00:00", text)

    def test_submitter_is_duplicate_safe_and_has_two_modes(self):
        text = SUBMITTER.read_text(encoding="ascii")
        self.assertIn("pilot|production", text)
        self.assertIn("one_g|no_f", text)
        self.assertIn("CANDIDATE_PROFILE", text)
        self.assertIn('test ! -e "$receipt"', text)
        self.assertIn('test ! -e "$run_root"', text)
        self.assertIn("sbatch --test-only", text)
        self.assertIn("sbatch --parsable", text)

    def test_atomic_evaluator_defers_annotations_for_python39(self):
        text = PROJECTED_PI.read_text(encoding="ascii")
        self.assertTrue(text.startswith("from __future__ import annotations\n"))


if __name__ == "__main__":
    unittest.main()
