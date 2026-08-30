from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run_atom_solid_differential_trace_log_df.slurm"
)


class AtomSolidDifferentialTraceLogJobTest(unittest.TestCase):
    def test_job_is_a_single_node_offline_replay_with_locked_inputs(self):
        text = SCRIPT.read_text(encoding="ascii")

        for expected in (
            "#SBATCH --partition=p1",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=40",
            "#SBATCH --mem=190000M",
            "module load python/3.9.22",
            "siab-torch19-py39",
            "--q-star-multiplicity 1:1",
            "--q-star-multiplicity 22:8",
            "--q-star-multiplicity 43:4",
            "--full-q-count 64",
            "--known-sos-binding-error-ev original:0.870347342670",
            "--known-sos-binding-error-ev fixed500:1.145074039398",
            "--known-sos-binding-error-ev relaxed:1.275306726142",
            "STATUS.json",
            "PROVENANCE.json",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

        self.assertNotIn("sbatch ", text)
        self.assertNotIn("mpirun", text.lower())
        self.assertNotIn("abacus_work", text.lower())
        self.assertNotIn("librpa_work", text.lower())

    def test_job_requires_an_immutable_deployment_and_new_run_root(self):
        text = SCRIPT.read_text(encoding="ascii")

        self.assertIn('test "$(cat "$REPO_ROOT/.git/HEAD")" = "$SIAB_COMMIT"', text)
        self.assertIn('test ! -e "$RUN_ROOT/result"', text)
        self.assertIn('mkdir "$RUN_ROOT/result"', text)


if __name__ == "__main__":
    unittest.main()
