import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analyze_atomic_projected_pi_spectrum.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_atomic_projected_pi_spectrum", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalyzeAtomicProjectedPiSpectrumTest(unittest.TestCase):
    def test_reports_causal_and_noncausal_spectra_without_taking_log(self):
        matrices = torch.stack(
            (
                torch.diag(torch.tensor([-2.0, 0.5], dtype=torch.complex128)),
                torch.diag(torch.tensor([-1.0, 1.25], dtype=torch.complex128)),
            )
        )

        report = MODULE.matrix_spectrum(matrices)

        self.assertEqual(report[0]["minimum_eigenvalue"], -2.0)
        self.assertEqual(report[0]["maximum_eigenvalue"], 0.5)
        self.assertEqual(report[0]["minimum_i_minus_pi_eigenvalue"], 0.5)
        self.assertTrue(report[0]["i_minus_pi_positive"])
        self.assertEqual(report[1]["minimum_i_minus_pi_eigenvalue"], -0.25)
        self.assertFalse(report[1]["i_minus_pi_positive"])

    def test_rejects_nonfinite_or_nonsquare_input(self):
        with self.assertRaisesRegex(ValueError, "frequency matrices"):
            MODULE.matrix_spectrum(torch.zeros((2, 3), dtype=torch.complex128))
        invalid = torch.tensor([[[float("nan") + 0.0j]]], dtype=torch.complex128)
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.matrix_spectrum(invalid)


if __name__ == "__main__":
    unittest.main()
