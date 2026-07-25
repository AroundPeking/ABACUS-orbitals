"""Static contracts for the H2 response-shell producer family."""

from pathlib import Path
import unittest


SIAB_ROOT = Path(__file__).resolve().parents[1]
GREEDY = SIAB_ROOT / "example_H_sternheimer" / "greedy_response_selection"


def input_values(path):
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) >= 2 and fields[0] != "INPUT_PARAMETERS":
            values[fields[0]] = " ".join(fields[1:])
    return values


class H2GreedyResponseTemplatesTest(unittest.TestCase):
    def test_h2_and_h2_ghost_replace_the_h3_target_contract(self):
        h2 = GREEDY / "producer_h2"
        ghost = GREEDY / "producer_h2_fragment_ghost"
        self.assertTrue(h2.is_dir())
        self.assertTrue(ghost.is_dir())

        h2_input = input_values(h2 / "INPUT")
        self.assertEqual(h2_input["suffix"], "H2_SIAB_GREEDY_R074085_WFULL_NF16_E50")
        self.assertEqual(h2_input["ntype"], "1")
        self.assertEqual(h2_input["nelec"], "2")
        self.assertEqual(h2_input["nspin"], "1")
        self.assertEqual(h2_input["ecutwfc"], "50")
        self.assertEqual(h2_input["sternheimer_nfreq"], "16")
        self.assertEqual(h2_input["rpa_ccp_rmesh_times"], "5")

        ghost_input = input_values(ghost / "INPUT")
        self.assertEqual(ghost_input["suffix"], "H_GHOST1_SIAB_GREEDY_R074085_WFULL_NF16_E50")
        self.assertEqual(ghost_input["ntype"], "2")
        self.assertEqual(ghost_input["nelec"], "1")
        self.assertEqual(ghost_input["nspin"], "2")

        h2_stru = (h2 / "STRU").read_text(encoding="utf-8")
        ghost_stru = (ghost / "STRU").read_text(encoding="utf-8")
        self.assertIn("0.48147879757009904 0.5 0.5", h2_stru)
        self.assertIn("0.518521202429901 0.5 0.5", h2_stru)
        self.assertIn("H_empty", ghost_stru)
        self.assertIn("0.48147879757009904 0.5 0.5", ghost_stru)
        self.assertIn("0.518521202429901 0.5 0.5", ghost_stru)

        runner = (GREEDY / "run_targets.slurm").read_text(encoding="utf-8")
        self.assertIn(
            "case_names=(producer_atom producer_h2 producer_h2_fragment_ghost)",
            runner,
        )
        self.assertIn("expected_atoms=(1 2 2)", runner)
        self.assertIn("expected_auxiliary_channels=(214 428 428)", runner)
        self.assertIn("ABACUS_STERNHEIMER_THREADS:-16", runner)

        selection = (GREEDY / "run_selection.slurm").read_text(encoding="utf-8")
        self.assertIn("producer_h2", selection)
        self.assertIn("producer_h2_fragment_ghost", selection)
        self.assertNotIn("producer_h3", selection)
        self.assertNotIn("$h3_target", selection)


if __name__ == "__main__":
    unittest.main()
