import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "optimize_periodic_basis.py"
SPEC = importlib.util.spec_from_file_location("optimize_periodic_basis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OptimizePeriodicBasisTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
