#!/usr/bin/env python3

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "atomic_optimization_gate.py"
SPEC = importlib.util.spec_from_file_location("atomic_optimization_gate", MODULE_PATH)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class AtomicOptimizationGateTest(unittest.TestCase):
    def setUp(self):
        self.template = json.loads(
            (HERE.parent / "SIAB_INPUT.atomic_gradient_gate.json").read_text(
                encoding="ascii"
            )
        )

    def test_template_freezes_dzp_and_varies_only_tzdp_excess(self):
        GATE.validate_template(self.template)
        self.assertEqual(
            self.template["element"]["Nu"]["C"], [3, 3, 2, 0, 0]
        )
        self.assertEqual(
            GATE.freeze_keys(self.template["freeze_orbitals"]),
            frozenset(
                {
                    ("C", 0, 1),
                    ("C", 0, 2),
                    ("C", 1, 1),
                    ("C", 1, 2),
                    ("C", 2, 1),
                }
            ),
        )
        self.assertEqual(
            GATE.variable_keys(self.template),
            frozenset({("C", 0, 3), ("C", 1, 3), ("C", 2, 2)}),
        )
        self.assertEqual(self.template["loss"]["mode"], "st_only")
        self.assertEqual(self.template["optimize"][0]["max_steps"], 20)

    def test_rejects_freezing_a_tzdp_excess_shell(self):
        value = copy.deepcopy(self.template)
        value["freeze_orbitals"].append(
            {"element": "C", "l": 2, "zeta": 2}
        )
        with self.assertRaisesRegex(ValueError, "DZP freeze set"):
            GATE.validate_template(value)

    def test_build_input_uses_absolute_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sternheimer_matrix.dat"
            coefficients = root / "ORBITAL_RESULTS.txt"
            target.write_bytes(b"target")
            coefficients.write_bytes(b"coefficients")
            result = GATE.build_input(self.template, target, coefficients)

        self.assertEqual(
            result["file_list"],
            {
                "sternheimer": [
                    {
                        "path": str(target.resolve()),
                        "family": "C_atom",
                        "role": "physical",
                    }
                ]
            },
        )
        self.assertEqual(
            result["C_init_info"]["C_init_file"],
            str(coefficients.resolve()),
        )

    def test_validate_source_hashes_rejects_wrong_orbital(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sternheimer_matrix.dat"
            coefficients = root / "ORBITAL_RESULTS.txt"
            orbital = root / "C.orb"
            target.write_bytes(b"target")
            coefficients.write_bytes(b"coefficients")
            orbital.write_bytes(b"wrong")
            expected = {
                "target": GATE.sha256(target),
                "coefficients": GATE.sha256(coefficients),
                "orbital": "0" * 64,
            }
            with self.assertRaisesRegex(ValueError, "orbital SHA256"):
                GATE.validate_source_hashes(
                    target, coefficients, orbital, expected=expected
                )

    def test_server66_job_uses_one_full_normal_compute_node(self):
        text = (HERE.parent / "run_atomic_gradient_gate_server66.slurm").read_text(
            encoding="ascii"
        )
        for marker in (
            "#SBATCH --partition=640",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=40",
            "#SBATCH --mem=110G",
            "#SBATCH --time=UNLIMITED",
            "export OMP_MAX_ACTIVE_LEVELS=1",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("debug", text.lower())

    def test_submitter_preflights_and_refuses_duplicate_receipt(self):
        text = (
            HERE.parent / "submit_atomic_gradient_gate_server66.sh"
        ).read_text(encoding="ascii")
        self.assertIn('test ! -e "$RECEIPT"', text)
        self.assertIn('test ! -e "$RESULT"', text)
        self.assertIn("sbatch --test-only", text)
        self.assertIn("sbatch --parsable", text)

    def test_df_job_uses_one_full_p1_node(self):
        text = (HERE.parent / "run_atomic_gradient_gate_df.slurm").read_text(
            encoding="ascii"
        )
        for marker in (
            "#SBATCH --partition=p1",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=40",
            "#SBATCH --mem=190000M",
            "#SBATCH --exclusive",
            "#SBATCH --time=UNLIMITED",
            "export OMP_MAX_ACTIVE_LEVELS=1",
            'test -x "$PYTHON_EXE"',
            'SOURCE_HEAD_FILE="$REPO_ROOT/.git/HEAD"',
            'SOURCE_REF_FILE="$REPO_ROOT/.git/$source_ref"',
        ):
            self.assertIn(marker, text)
        self.assertNotIn("debug", text.lower())
        self.assertNotIn("git rev-parse", text)
        self.assertNotIn('"$ORBITAL" "$TARGET" "$PYTHON_EXE"', text)

    def test_df_submitter_requires_zero_elapsed_server66_migration(self):
        text = (HERE.parent / "submit_atomic_gradient_gate_df.sh").read_text(
            encoding="ascii"
        )
        self.assertIn('grep -qx "server66_job_id=410776"', text)
        self.assertIn('grep -qx "server66_state=CANCELLED"', text)
        self.assertIn('grep -qx "server66_elapsed=00:00:00"', text)
        self.assertIn("sbatch --test-only", text)
        self.assertIn("sbatch --parsable", text)


if __name__ == "__main__":
    unittest.main()
