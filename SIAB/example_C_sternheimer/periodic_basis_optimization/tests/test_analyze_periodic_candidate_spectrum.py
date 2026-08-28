import importlib.util
from pathlib import Path
import sys
import unittest

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analyze_periodic_candidate_spectrum.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_periodic_candidate_spectrum", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AnalyzePeriodicCandidateSpectrumTest(unittest.TestCase):
    def test_solves_generalized_hermitian_eigenproblem(self):
        overlap = torch.diag(
            torch.tensor([2.0, 0.5], dtype=torch.complex128)
        )
        hamiltonian = torch.diag(
            torch.tensor([6.0, 1.0], dtype=torch.complex128)
        )

        result = MODULE.solve_generalized_spectrum(
            overlap,
            hamiltonian,
            relative_rank_tolerance=1.0e-12,
        )

        self.assertEqual(result.rank, 2)
        self.assertAlmostEqual(result.condition, 4.0)
        self.assertTrue(
            torch.allclose(
                result.eigenvalue_ha,
                torch.tensor([2.0, 3.0], dtype=torch.float64),
            )
        )

    def test_rejects_rank_deficient_candidate_overlap(self):
        overlap = torch.diag(
            torch.tensor([1.0, 1.0e-14], dtype=torch.complex128)
        )
        hamiltonian = torch.eye(2, dtype=torch.complex128)

        with self.assertRaisesRegex(RuntimeError, "rank deficient"):
            MODULE.solve_generalized_spectrum(
                overlap,
                hamiltonian,
                relative_rank_tolerance=1.0e-12,
            )


if __name__ == "__main__":
    unittest.main()
