import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


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


if __name__ == "__main__":
    unittest.main()
