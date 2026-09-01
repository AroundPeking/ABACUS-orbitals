#!/usr/bin/env python3

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SIAB_ROOT = ROOT.parents[1]
WORKFLOW = ROOT / "galerkin_binding_workflow"
SCRIPT = WORKFLOW / "stage_selected_c_candidate.py"
sys.path.insert(0, str(SIAB_ROOT / "opt_orb_pytorch_dpsi"))

from periodic_galerkin_basis import write_periodic_optimizer_coefficients


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("stage_selected_c_candidate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageSelectedCGalerkinCandidateTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def _inputs(self, parent):
        bank_root = parent / "bank"
        candidate_dir = bank_root / "candidates" / "pareto_w0p25"
        candidate_dir.mkdir(parents=True)
        coefficients = {
            "C": [
                torch.eye(31, 3, dtype=torch.float64),
                torch.eye(31, 3, dtype=torch.float64),
                torch.eye(31, 2, dtype=torch.float64),
                torch.empty(31, 0, dtype=torch.float64),
                torch.empty(31, 0, dtype=torch.float64),
            ]
        }
        coefficient_path = candidate_dir / "ORBITAL_RESULTS.txt"
        write_periodic_optimizer_coefficients(coefficient_path, coefficients)
        coefficient_sha = sha256(coefficient_path)
        bank = {
            "format_version": 1,
            "status": "success",
            "candidate_bank_gate": "pass",
            "source_commit": "1" * 40,
            "input_sha256": {"initial": "2" * 64},
            "candidates": [
                {
                    "name": "pareto_w0p25",
                    "family_tradeoff_gate": "pass",
                    "orbital_file": str(coefficient_path),
                    "orbital_sha256": coefficient_sha,
                }
            ],
        }
        bank_path = bank_root / "CANDIDATE_BANK.json"
        bank_path.write_text(
            json.dumps(bank, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        (bank_root / "STATUS.json").write_text(
            json.dumps({"status": "success", "candidate_bank_sha256": sha256(bank_path)}),
            encoding="ascii",
        )
        (bank_root / "PROVENANCE.json").write_text(
            json.dumps({"status": "success", "candidate_bank_sha256": sha256(bank_path)}),
            encoding="ascii",
        )

        q3_root = parent / "q3"
        q3_root.mkdir()
        selection = {
            "format_version": 1,
            "status": "success",
            "gate": "pass",
            "selected_candidate": "pareto_w0p25",
            "selected_orbital_sha256": coefficient_sha,
        }
        selection_path = q3_root / "SELECTION_RESULT.json"
        selection_path.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        (q3_root / "STATUS.json").write_text(
            json.dumps({"status": "success", "gate": "pass", "selection_sha256": sha256(selection_path)}),
            encoding="ascii",
        )
        (q3_root / "PROVENANCE.json").write_text(
            json.dumps({"status": "success", "selection_sha256": sha256(selection_path)}),
            encoding="ascii",
        )
        return bank_root, q3_root, coefficient_path, coefficient_sha

    def test_stages_a_self_contained_hash_locked_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bank_root, q3_root, coefficient_path, coefficient_sha = self._inputs(parent)
            output = parent / "staged"

            result = self.module.stage_selected_candidate(
                bank_root=bank_root,
                q3_root=q3_root,
                output_directory=output,
                source_commit="3" * 40,
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["profile"], "galerkin_pareto_dzp")
            self.assertEqual(result["selected_candidate"], "pareto_w0p25")
            self.assertEqual(result["source_coefficients_sha256"], coefficient_sha)
            self.assertEqual(result["nu"], [3, 3, 2, 0, 0])
            self.assertEqual(result["ao_count_atom"], 22)
            self.assertNotEqual(output / result["coefficients_filename"], coefficient_path)
            self.assertEqual(
                sha256(output / result["coefficients_filename"]),
                coefficient_sha,
            )
            self.assertEqual(
                sha256(output / result["orbital_filename"]),
                result["orbital_sha256"],
            )
            self.assertEqual(
                json.loads((output / "CANDIDATE.json").read_text(encoding="ascii")),
                result,
            )
            self.assertIn(
                "status=success\n",
                (output / "provenance.txt").read_text(encoding="ascii"),
            )

    def test_rejects_selection_hash_that_differs_from_candidate_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bank_root, q3_root, _, _ = self._inputs(parent)
            selection_path = q3_root / "SELECTION_RESULT.json"
            selection = json.loads(selection_path.read_text(encoding="ascii"))
            selection["selected_orbital_sha256"] = "9" * 64
            selection_path.write_text(json.dumps(selection), encoding="ascii")
            status = json.loads((q3_root / "STATUS.json").read_text(encoding="ascii"))
            status["selection_sha256"] = sha256(selection_path)
            (q3_root / "STATUS.json").write_text(json.dumps(status), encoding="ascii")
            provenance = json.loads((q3_root / "PROVENANCE.json").read_text(encoding="ascii"))
            provenance["selection_sha256"] = sha256(selection_path)
            (q3_root / "PROVENANCE.json").write_text(json.dumps(provenance), encoding="ascii")

            with self.assertRaisesRegex(ValueError, "selection hash differs"):
                self.module.stage_selected_candidate(
                    bank_root=bank_root,
                    q3_root=q3_root,
                    output_directory=parent / "staged",
                    source_commit="3" * 40,
                )


if __name__ == "__main__":
    unittest.main()
