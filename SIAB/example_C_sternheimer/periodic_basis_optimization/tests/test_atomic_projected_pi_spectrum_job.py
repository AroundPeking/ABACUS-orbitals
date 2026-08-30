from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run_atomic_projected_pi_spectrum_df.slurm"
)


class AtomicProjectedPiSpectrumJobTest(unittest.TestCase):
    def test_is_a_low_memory_read_only_single_frequency_diagnostic(self):
        text = SCRIPT.read_text(encoding="ascii")
        commands = "\n".join(
            line for line in text.lower().splitlines()
            if not line.lstrip().startswith("#")
        )
        for expected in (
            "#SBATCH --partition=48cp1",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=40",
            "#SBATCH --mem=32000M",
            "extract_sternheimer_frequency_subset.py",
            "analyze_atomic_projected_pi_spectrum.py",
            "--frequency-index 0",
            "reference_i_minus_pi_positive",
            "candidate_i_minus_pi_positive",
            "STATUS.json",
            "PROVENANCE.json",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)
        for forbidden in ("sbatch ", "mpirun", "abacus_work", "librpa_work"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, commands)

    def test_requires_immutable_deployment_and_new_result(self):
        text = SCRIPT.read_text(encoding="ascii")
        self.assertIn('test "$(cat "$REPO_ROOT/.git/HEAD")" = "$SIAB_COMMIT"', text)
        self.assertIn('test ! -e "$RUN_ROOT/result"', text)


if __name__ == "__main__":
    unittest.main()
