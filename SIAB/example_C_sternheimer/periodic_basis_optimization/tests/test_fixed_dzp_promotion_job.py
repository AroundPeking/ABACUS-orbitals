import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_fixed_dzp_promotion_df.slurm"


class FixedDzpPromotionJobTest(unittest.TestCase):
    def test_job_uses_fixed_prefix_candidate_contract(self):
        text = SCRIPT.read_text(encoding="ascii")
        self.assertIn("prepare_relaxed_dzp_candidate.py", text)
        self.assertIn("--profile fixed_dzp", text)
        self.assertIn("--candidate fixed-dzp:", text)
        self.assertIn("fixed_dzp_independent_q3_spectrum_and_export_gate", text)
        self.assertIn("selected_iq 43", text)
        self.assertIn("--occupied-capture-floor 0.9998982409775239", text)
        self.assertIn("grep -qx 'status=success'", text)
        self.assertIn('test ! -e "$CANDIDATE_ROOT"', text)


if __name__ == "__main__":
    unittest.main()
