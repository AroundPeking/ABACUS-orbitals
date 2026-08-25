import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / "opt_orb_pytorch_dpsi" / "periodic_galerkin_data.py"
SPEC = importlib.util.spec_from_file_location("periodic_galerkin_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PeriodicGalerkinDataTest(unittest.TestCase):
    def test_whitening_limit_accounts_for_retained_rank(self):
        declared_max_element_error = 4.5e-9
        self.assertAlmostEqual(
            MODULE._whitening_consistency_limit(235, declared_max_element_error),
            235 * declared_max_element_error + 1.0e-12,
        )

    def test_whitening_limit_keeps_absolute_floor(self):
        self.assertEqual(MODULE._whitening_consistency_limit(3, 1.0e-12), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
