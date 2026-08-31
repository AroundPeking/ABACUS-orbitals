import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
SCRIPT = ROOT / "run_relaxed_dzp_pbe_gate_55d25e3c9.slurm"


class RelaxedDzpPbeJobContractTest(unittest.TestCase):
    def test_job_runs_dynamic_candidate_atom_and_solid_before_rpa(self):
        text = SCRIPT.read_text(encoding="ascii")
        self.assertIn('case "${SLURM_JOB_PARTITION:?}" in', text)
        self.assertIn("p1|48cp2)", text)
        self.assertNotIn('test "${SLURM_JOB_PARTITION:?}" = p1', text)
        self.assertIn("CANDIDATE.json", text)
        self.assertIn("read_periodic_candidate_manifest.py", text)
        self.assertIn("collect_relaxed_dzp_pbe_gate.py", text)
        self.assertIn("reference_basis=original_unoptimized_sg15_tzdp", text)
        self.assertIn('set_input_key "$input" rpa 0', text)
        self.assertIn("prepare_side atom", text)
        self.assertIn('C_CANDIDATE_PBE_ATOM "$nao_atom"', text)
        self.assertIn("prepare_side solid", text)
        self.assertIn('C_CANDIDATE_PBE_SOLID "$((2 * nao_atom))"', text)
        self.assertIn('set_input_key "$input" init_wfc atomic', text)
        self.assertIn('set_input_key "$input" init_chg atomic', text)
        self.assertIn('set_input_key "$input" ocp_set "3*1 $((nbands - 3))*0 1*1 $((nbands - 1))*0"', text)
        self.assertIn("#SCF IS CONVERGED#", text)
        self.assertIn("!FINAL_ETOT_IS", text)
        self.assertIn("--tolerance-ev 0.010", text)
        self.assertIn("grep -q '\"pbe_gate\": \"pass\"'", text)


if __name__ == "__main__":
    unittest.main()
