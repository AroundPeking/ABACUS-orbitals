import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_relaxed_dzp_promotion_df.slurm"


class RelaxedDzpPromotionJobTest(unittest.TestCase):
    def test_job_runs_independent_gates_and_exports_once(self):
        text = SCRIPT.read_text(encoding="ascii")
        self.assertIn("compare_periodic_candidates.py", text)
        self.assertIn("analyze_periodic_candidate_spectrum.py", text)
        self.assertIn("export_periodic_orbitals.py", text)
        self.assertIn("prepare_relaxed_dzp_candidate.py", text)
        self.assertIn("selected_iq 43", text)
        self.assertIn("--candidate original-tzdp:", text)
        self.assertIn("--candidate relaxed-dzp:", text)
        self.assertIn("--nu 3,3,2,0,0", text)
        self.assertIn("--occupied-capture-floor 0.9998982409775239", text)
        self.assertIn('test ! -e "$CANDIDATE_ROOT"', text)

    def test_job_accepts_optimizer_key_value_provenance(self):
        text = SCRIPT.read_text(encoding="ascii")
        self.assertIn(
            "grep -qx 'status=success' \"$OPTIMIZER_ROOT/provenance.txt\"",
            text,
        )
        self.assertNotIn(
            "grep -qx 'status success' \"$OPTIMIZER_ROOT/provenance.txt\"",
            text,
        )


if __name__ == "__main__":
    unittest.main()
