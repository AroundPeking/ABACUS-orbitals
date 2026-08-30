import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
SCRIPT = ROOT / "run_relaxed_dzp_pbe_gate_55d25e3c9.slurm"


class RelaxedDzpPbeJobContractTest(unittest.TestCase):
    def test_job_runs_atom_and_solid_before_rpa(self):
        text = SCRIPT.read_text(encoding="ascii")
        self.assertIn("CANDIDATE.json", text)
        self.assertIn("collect_relaxed_dzp_pbe_gate.py", text)
        self.assertIn("reference_basis=original_unoptimized_sg15_tzdp", text)
        self.assertIn('set_input_key "$input" rpa 0', text)
        self.assertIn("prepare_side atom", text)
        self.assertIn("C_RELAXED_DZP_PBE_ATOM 22", text)
        self.assertIn("prepare_side solid", text)
        self.assertIn("C_RELAXED_DZP_PBE_SOLID 44", text)
        self.assertIn('set_input_key "$input" init_wfc file', text)
        self.assertIn('set_input_key "$input" init_chg file', text)
        self.assertIn("#SCF IS CONVERGED#", text)
        self.assertIn("!FINAL_ETOT_IS", text)
        self.assertIn("--tolerance-ev 0.010", text)
        self.assertIn("grep -q '\"pbe_gate\": \"pass\"'", text)


if __name__ == "__main__":
    unittest.main()
