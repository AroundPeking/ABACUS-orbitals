from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parents[1]
RUNNER = HERE / "run_c_atom_source_only_df.slurm"
SUBMITTER = HERE / "submit_c_atom_source_only_df.sh"


class CAtomSourceOnlyTest(unittest.TestCase):
    def test_runner_uses_full_nodes_without_solving_delta_st(self):
        text = RUNNER.read_text(encoding="ascii")
        self.assertIn("#SBATCH --partition=p1", text)
        self.assertIn("#SBATCH --nodes=4", text)
        self.assertIn("#SBATCH --ntasks-per-node=1", text)
        self.assertIn("#SBATCH --cpus-per-task=40", text)
        self.assertIn("#SBATCH --mem=190000M", text)
        self.assertIn("sternheimer_siab_source_only 1", text)
        self.assertIn("out_sternheimer_basis_opt 1", text)
        self.assertNotIn("out_sternheimer_librpa 1", text)
        self.assertIn('/usr/bin/git -C "$REPO_ROOT"', text)
        self.assertIn("pair_response_and_source", text)
        self.assertIn("SOURCE_ONLY_COMPLETE.json", text)

    def test_submitter_refuses_duplicate_receipt_or_result(self):
        text = SUBMITTER.read_text(encoding="ascii")
        self.assertIn("test ! -e \"$receipt\"", text)
        self.assertIn("test ! -e \"$run_root\"", text)
        self.assertIn("sbatch --test-only", text)
        self.assertIn("sbatch --parsable", text)


if __name__ == "__main__":
    unittest.main()
