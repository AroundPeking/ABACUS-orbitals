import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_periodic_basis.py"
SPEC = importlib.util.spec_from_file_location("evaluate_periodic_basis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluatePeriodicBasisTest(unittest.TestCase):
    def test_parses_explicit_zero_high_l_channels(self):
        self.assertEqual(MODULE.parse_nu("3,3,2,0,0", max_l=4), (3, 3, 2, 0, 0))

    def test_rejects_wrong_number_of_channels(self):
        with self.assertRaisesRegex(ValueError, "five"):
            MODULE.parse_nu("3,3,2", max_l=4)

    def test_rejects_empty_basis(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            MODULE.parse_nu("0,0,0,0,0", max_l=4)


if __name__ == "__main__":
    unittest.main()
