#!/usr/bin/env python3

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CAMPAIGN_DIR = HERE.parent
MODULE_PATH = CAMPAIGN_DIR / "continuation_campaign.py"
SPEC = importlib.util.spec_from_file_location("continuation_campaign", MODULE_PATH)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def write_spillage(path, losses, conditions=None, accepted=None):
    conditions = conditions or [2.0] * len(losses)
    accepted = accepted or [True] * len(losses)
    lines = [
        "istep_big\tistep_small\tistep_all\tsternheimer\t"
        "max_st_condition\taccepted"
    ]
    for index, (loss, condition, is_accepted) in enumerate(
        zip(losses, conditions, accepted)
    ):
        lines.append(
            f"{index - 1}\t0\t{index}\t{loss}\t{condition}\t"
            f"{str(is_accepted).lower()}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class ContinuationCampaignTest(unittest.TestCase):
    def setUp(self):
        self.template = json.loads(
            (CAMPAIGN_DIR / "SIAB_INPUT.tzdp_continuation.json").read_text(
                encoding="ascii"
            )
        )

    def test_template_preserves_physics_and_runs_long_continuation(self):
        CAMPAIGN.validate_template(self.template)
        self.assertEqual(self.template["element"]["Nu"]["C"], [3, 3, 2, 0, 0])
        self.assertEqual(
            CAMPAIGN.freeze_keys(self.template["freeze_orbitals"]),
            CAMPAIGN.FIXED_DZP,
        )
        self.assertEqual(
            CAMPAIGN.variable_keys(self.template), CAMPAIGN.EXPECTED_VARIABLE
        )
        self.assertEqual(
            self.template["optimize"],
            [
                {
                    "optimizer": "Adam",
                    "kwargs": {"lr": 0.001},
                    "cal_T": False,
                    "norm": "element",
                    "max_steps": 3000,
                }
            ],
        )
        self.assertEqual(self.template["loss"]["mode"], "st_only")

    def test_rejects_changed_optimizer_or_shell_contract(self):
        wrong_steps = copy.deepcopy(self.template)
        wrong_steps["optimize"][0]["max_steps"] = 20
        with self.assertRaisesRegex(ValueError, "3000 steps"):
            CAMPAIGN.validate_template(wrong_steps)
        wrong_shells = copy.deepcopy(self.template)
        wrong_shells["element"]["Nu"]["C"] = [3, 3, 2]
        with self.assertRaisesRegex(ValueError, "zero f/g"):
            CAMPAIGN.validate_template(wrong_shells)

    def test_build_input_points_to_checkpoint_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sternheimer_matrix.dat"
            checkpoint = root / "ORBITAL_RESULTS.txt"
            target.write_bytes(b"target")
            checkpoint.write_bytes(b"checkpoint")
            result = CAMPAIGN.build_input(self.template, target, checkpoint)

        self.assertEqual(
            result["file_list"]["sternheimer"][0]["path"],
            str(target.resolve()),
        )
        self.assertEqual(
            result["C_init_info"]["C_init_file"], str(checkpoint.resolve())
        )

    def test_parses_only_accepted_finite_optimizer_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Spillage.dat"
            write_spillage(
                path,
                [0.75, 0.74, 0.73],
                conditions=[2.0, 3.0, 4.0],
                accepted=[True, False, True],
            )
            rows = CAMPAIGN.read_spillage_rows(path)

        self.assertEqual([row.step for row in rows], [-1, 1])
        self.assertEqual([row.loss for row in rows], [0.75, 0.73])
        self.assertEqual([row.condition for row in rows], [2.0, 4.0])

    def test_final_window_convergence_uses_best_loss_not_last_loss(self):
        losses = [1.0, 0.7] + [0.7] * 199
        converged, relative_drop = CAMPAIGN.final_window_converged(
            losses, window=100, tolerance=1.0e-4
        )
        self.assertTrue(converged)
        self.assertEqual(relative_drop, 0.0)

    def test_final_window_detects_continued_progress(self):
        losses = [1.0 - index * 0.001 for index in range(201)]
        converged, relative_drop = CAMPAIGN.final_window_converged(
            losses, window=100, tolerance=1.0e-4
        )
        self.assertFalse(converged)
        self.assertGreater(relative_drop, 1.0e-4)

    def test_detects_only_a_real_51_step_nonimprovement_stop(self):
        plateau = [1.0, 0.8] + [0.81 + index * 1.0e-5 for index in range(51)]
        self.assertTrue(
            CAMPAIGN.detect_nonimprovement_stop(plateau, optimizer_rows=52)
        )
        self.assertFalse(
            CAMPAIGN.detect_nonimprovement_stop(plateau[:-1], optimizer_rows=51)
        )
        self.assertFalse(
            CAMPAIGN.detect_nonimprovement_stop(plateau, optimizer_rows=3000)
        )

    def test_detects_51_nonimprovements_after_the_initial_point(self):
        losses = [0.8] + [0.81 + index * 1.0e-5 for index in range(51)]
        self.assertTrue(
            CAMPAIGN.detect_nonimprovement_stop(losses, optimizer_rows=51)
        )

    def test_assess_convergence_distinguishes_pass_and_continue(self):
        plateau = [1.0, 0.8] + [0.81] * 51
        passed = CAMPAIGN.assess_convergence(
            plateau,
            optimizer_rows=52,
            maximum_condition=3.0,
            checkpoint_loss=0.9,
        )
        self.assertEqual(passed["status"], "TZDP_CONVERGED")
        progress = [1.0 - index * 1.0e-4 for index in range(301)]
        pending = CAMPAIGN.assess_convergence(
            progress,
            optimizer_rows=300,
            maximum_condition=3.0,
            checkpoint_loss=1.1,
        )
        self.assertEqual(pending["status"], "CONTINUE_REQUIRED")

    def test_assess_convergence_rejects_condition_limit(self):
        with self.assertRaisesRegex(RuntimeError, "condition"):
            CAMPAIGN.assess_convergence(
                [1.0, 0.9],
                optimizer_rows=1,
                maximum_condition=1.0e12,
                checkpoint_loss=1.1,
            )

    def test_df_runner_uses_partial_p1_node_and_guards_spectrum(self):
        text = (CAMPAIGN_DIR / "run_tzdp_continuation_df.slurm").read_text(
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
            'SOURCE_REF_FILE="$REPO_ROOT/.git/$source_ref"',
            'test -x "$PYTHON_EXE"',
            '"$PYTHON_EXE" "$CAMPAIGN_SCRIPT" audit',
            'grep -q \'"status": "TZDP_CONVERGED"\'',
            '"$PYTHON_EXE" "$SPECTRUM_SCRIPT"',
        ):
            self.assertIn(marker, text)
        self.assertNotIn("#SBATCH --exclusive", text)
        self.assertNotIn("debug", text.lower())
        self.assertNotIn("git rev-parse", text)

    def test_submitter_preflights_and_refuses_duplicate_outputs(self):
        text = (CAMPAIGN_DIR / "submit_tzdp_continuation_df.sh").read_text(
            encoding="ascii"
        )
        self.assertIn('for path in "$SCRIPT" "$TARGET" "$CHECKPOINT"; do', text)
        self.assertNotIn(
            'for path in "$SCRIPT" "$TARGET" "$CHECKPOINT" "$PYTHON_EXE"; do',
            text,
        )
        self.assertIn('test -x "$PYTHON_EXE"', text)
        self.assertIn('test ! -e "$RECEIPT"', text)
        self.assertIn('test ! -e "$RESULT"', text)
        self.assertIn('test ! -e "$SPECTRUM"', text)
        self.assertIn("sbatch --test-only", text)
        self.assertIn("sbatch --parsable", text)


if __name__ == "__main__":
    unittest.main()
