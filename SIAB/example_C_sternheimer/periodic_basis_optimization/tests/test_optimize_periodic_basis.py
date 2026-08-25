import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "optimize_periodic_basis.py"
SPEC = importlib.util.spec_from_file_location("optimize_periodic_basis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OptimizePeriodicBasisTest(unittest.TestCase):
    def test_parses_explicit_block_cache_worker_count(self):
        args = MODULE.parse_args(
            [
                "--dataset",
                "q1",
                "--initial",
                "initial.txt",
                "--output-directory",
                "result",
                "--siab-commit",
                "a" * 40,
                "--block-cache-workers",
                "8",
            ]
        )

        self.assertEqual(args.block_cache_workers, 8)

    def test_parses_fixed_prefix_occupied_capture_reference(self):
        args = MODULE.parse_args(
            [
                "--dataset",
                "q1",
                "--initial",
                "initial.txt",
                "--output-directory",
                "result",
                "--siab-commit",
                "a" * 40,
                "--occupied-capture-reference",
                "fixed_prefix",
            ]
        )

        self.assertEqual(args.occupied_capture_reference, "fixed_prefix")

    def test_parses_layout_only_validation_for_unused_projection_chunks(self):
        args = MODULE.parse_args(
            [
                "--dataset",
                "q1",
                "--initial",
                "initial.txt",
                "--output-directory",
                "result",
                "--siab-commit",
                "a" * 40,
                "--omitted-reference-projection-validation",
                "layout",
            ]
        )

        self.assertEqual(args.omitted_reference_projection_validation, "layout")

    def test_fixed_prefix_must_not_exceed_candidate_counts(self):
        self.assertEqual(
            MODULE.parse_channel_counts("2,2,1,0,0", (3, 3, 2, 0, 0)),
            (2, 2, 1, 0, 0),
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            MODULE.parse_channel_counts("4,2,1,0,0", (3, 3, 2, 0, 0))

    def test_requires_full_commit_hash(self):
        self.assertEqual(MODULE.validate_commit("a" * 40), "a" * 40)
        with self.assertRaisesRegex(ValueError, "commit"):
            MODULE.validate_commit("a1129b06")

    def test_allows_q_dependent_whitened_auxiliary_rank(self):
        common = {
            "abacus_commit": "a" * 40,
            "executable_sha256": "b" * 64,
            "orbital_sha256": "c" * 64,
            "pseudopotential_sha256": "d" * 64,
            "auxiliary_basis_sha256": "e" * 64,
            "primitive_blocks_sha256": "f" * 64,
            "primitive_count": 10,
            "raw_auxiliary_dimension": 8,
            "primitive_blocks": (("C", 0, 0, 10),),
        }
        q1 = SimpleNamespace(**common, whitened_auxiliary_rank=6)
        q2 = SimpleNamespace(**common, whitened_auxiliary_rank=7)
        MODULE.validate_dataset_contract((q1, q2))

        changed = dict(common)
        changed["auxiliary_basis_sha256"] = "0" * 64
        q2_bad = SimpleNamespace(**changed, whitened_auxiliary_rank=7)
        with self.assertRaisesRegex(ValueError, "basis/provenance"):
            MODULE.validate_dataset_contract((q1, q2_bad))

    def test_best_checkpoint_records_hash_and_replaces_previous_basis(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            first = {"C": [torch.eye(3, 2, dtype=torch.float64)]}
            second = {
                "C": [
                    torch.tensor(
                        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
                        dtype=torch.float64,
                    )
                ]
            }

            MODULE.write_best_checkpoint(output, 2, 0.5, first)
            MODULE.write_best_checkpoint(output, 7, 0.25, second)

            metadata = json.loads(
                (output / "BEST_CHECKPOINT.json").read_text(encoding="ascii")
            )
            orbital = output / "BEST_ORBITAL_CHECKPOINT.txt"
            self.assertEqual(metadata["step"], 7)
            self.assertEqual(metadata["loss"], 0.25)
            self.assertEqual(metadata["orbital_sha256"], MODULE.sha256(orbital))
            restored = MODULE.read_periodic_optimizer_coefficients(
                orbital,
                element="C",
                radial_rows=3,
                expected_nu=(2,),
            )
            self.assertTrue(torch.equal(restored["C"][0], second["C"][0]))
            self.assertFalse((output / ".BEST_ORBITAL_CHECKPOINT.txt.tmp").exists())
            self.assertFalse((output / ".BEST_CHECKPOINT.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
