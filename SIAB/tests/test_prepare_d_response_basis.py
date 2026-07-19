import sys
from pathlib import Path
from unittest import mock
import unittest

import common  # noqa: F401 - configures the optimizer import path


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example_H_sternheimer"
sys.path.insert(0, str(EXAMPLE_DIR))

import prepare_d_response_basis


class PrepareDResponseBasisTest(unittest.TestCase):
    def parse_args(self, *extra):
        argv = [
            "prepare_d_response_basis.py",
            "--target",
            "target.dat",
            "--baseline",
            "baseline.txt",
            "--output-dir",
            "output",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return prepare_d_response_basis.parse_args()

    def test_uses_stable_production_rank_tolerance_by_default(self):
        args = self.parse_args()

        self.assertEqual(args.relative_rank_tolerance, 1.0e-4)

    def test_accepts_an_explicit_rank_tolerance(self):
        args = self.parse_args("--relative-rank-tolerance", "2e-5")

        self.assertEqual(args.relative_rank_tolerance, 2.0e-5)


if __name__ == "__main__":
    unittest.main()
