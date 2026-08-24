#!/usr/bin/env python3

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CAMPAIGN_DIR = HERE.parent
sys.path.insert(0, str(CAMPAIGN_DIR))
MODULE_PATH = CAMPAIGN_DIR / "f_shell_campaign.py"
SPEC = importlib.util.spec_from_file_location("f_shell_campaign", MODULE_PATH)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class FShellCampaignTest(unittest.TestCase):
    def setUp(self):
        self.template = json.loads(
            (CAMPAIGN_DIR / "SIAB_INPUT.f_shell_optimization.json").read_text(
                encoding="ascii"
            )
        )

    def test_template_varies_only_tzdp_excess_and_first_f_shell(self):
        CAMPAIGN.validate_template(self.template)
        self.assertEqual(self.template["element"]["Nu"]["C"], [3, 3, 2, 1, 0])
        self.assertEqual(
            CAMPAIGN.freeze_keys(self.template["freeze_orbitals"]),
            CAMPAIGN.FIXED_DZP,
        )
        self.assertEqual(
            CAMPAIGN.variable_keys(self.template), CAMPAIGN.EXPECTED_VARIABLE
        )
        self.assertEqual(self.template["loss"]["mode"], "st_only")

    def test_rejects_missing_f_or_unfrozen_dzp(self):
        missing_f = copy.deepcopy(self.template)
        missing_f["element"]["Nu"]["C"] = [3, 3, 2, 0, 0]
        with self.assertRaisesRegex(ValueError, "3s3p2d1f"):
            CAMPAIGN.validate_template(missing_f)
        unfrozen = copy.deepcopy(self.template)
        unfrozen["freeze_orbitals"] = unfrozen["freeze_orbitals"][:-1]
        with self.assertRaisesRegex(ValueError, "frozen DZP"):
            CAMPAIGN.validate_template(unfrozen)

    def test_build_input_points_to_f_seed_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sternheimer_matrix.dat"
            seed = root / "ORBITAL_RESULTS.next_shell_seed.txt"
            target.write_bytes(b"target")
            seed.write_bytes(b"seed")
            result = CAMPAIGN.build_input(self.template, target, seed)

        self.assertEqual(
            result["file_list"]["sternheimer"][0]["path"], str(target.resolve())
        )
        self.assertEqual(result["C_init_info"]["C_init_file"], str(seed.resolve()))

    def test_material_gain_requires_convergence_and_one_percent_reduction(self):
        report = CAMPAIGN.assess_f_shell_gain(
            [0.704, 0.700] + [0.701] * 51,
            optimizer_rows=52,
            maximum_condition=3.0,
        )
        self.assertEqual(report["status"], "F_SHELL_MATERIAL_GAIN")
        self.assertTrue(report["advance_to_multicenter_projected_pi"])
        self.assertGreaterEqual(report["relative_gain_to_3s3p2d"], 0.01)

    def test_converged_subpercent_gain_is_marginal(self):
        report = CAMPAIGN.assess_f_shell_gain(
            [0.710, 0.707] + [0.708] * 51,
            optimizer_rows=52,
            maximum_condition=3.0,
        )
        self.assertEqual(report["status"], "F_SHELL_MARGINAL_GAIN")
        self.assertFalse(report["advance_to_multicenter_projected_pi"])

    def test_unconverged_gain_does_not_advance(self):
        losses = [0.70 - index * 1.0e-4 for index in range(301)]
        report = CAMPAIGN.assess_f_shell_gain(
            losses,
            optimizer_rows=300,
            maximum_condition=3.0,
        )
        self.assertEqual(report["status"], "CONTINUE_REQUIRED")
        self.assertFalse(report["advance_to_multicenter_projected_pi"])

    def test_seed_must_improve_the_converged_3s3p2d_space(self):
        with self.assertRaisesRegex(RuntimeError, "f-shell seed"):
            CAMPAIGN.assess_f_shell_gain(
                [CAMPAIGN.PRIOR_BASIS_LOSS, 0.70] + [0.701] * 51,
                optimizer_rows=52,
                maximum_condition=3.0,
            )

    def test_df_runner_is_single_node_and_stops_before_next_shell_search(self):
        text = (CAMPAIGN_DIR / "run_f_shell_optimization_df.slurm").read_text(
            encoding="ascii"
        )
        for marker in (
            "#SBATCH --partition=p1",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=16G",
            "#SBATCH --time=UNLIMITED",
            'SOURCE_HEAD_FILE="$REPO_ROOT/.git/HEAD"',
            'test -x "$PYTHON_EXE"',
            '"$PYTHON_EXE" "$CAMPAIGN_SCRIPT" audit',
        ):
            self.assertIn(marker, text)
        self.assertNotIn("analyze_residual_spectrum.py", text)
        self.assertNotIn("debug", text.lower())
        self.assertNotIn("git rev-parse", text)

    def test_submitter_preflights_and_keeps_slurm_files_in_campaign(self):
        text = (CAMPAIGN_DIR / "submit_f_shell_optimization_df.sh").read_text(
            encoding="ascii"
        )
        self.assertIn('for path in "$SCRIPT" "$TARGET" "$SEED"; do', text)
        self.assertIn('test -x "$PYTHON_EXE"', text)
        self.assertIn('test ! -e "$RECEIPT"', text)
        self.assertIn('test ! -e "$RESULT"', text)
        self.assertIn('cd "$CAMPAIGN_ROOT"', text)
        self.assertIn("sbatch --test-only", text)
        self.assertIn("sbatch --parsable", text)


if __name__ == "__main__":
    unittest.main()
