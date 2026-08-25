import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "expand_periodic_coefficients.py"
SPEC = importlib.util.spec_from_file_location("expand_periodic_coefficients", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExpandPeriodicCoefficientsTest(unittest.TestCase):
    def test_appends_orthogonal_deterministic_complement(self):
        channel = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        coefficients = {"C": [channel, torch.empty((4, 0), dtype=torch.float64)]}

        expanded = MODULE.expand_coefficients(
            coefficients,
            (3, 1),
            element="C",
        )

        self.assertTrue(torch.equal(expanded["C"][0][:, :2], channel))
        added = expanded["C"][0][:, 2]
        torch.testing.assert_close(
            channel.transpose(0, 1).matmul(added),
            torch.zeros(2, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            torch.linalg.norm(added),
            torch.tensor(1.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            expanded["C"][1][:, 0],
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )

        repeated = MODULE.expand_coefficients(
            coefficients,
            (3, 1),
            element="C",
        )
        self.assertTrue(torch.equal(repeated["C"][0], expanded["C"][0]))
        self.assertTrue(torch.equal(repeated["C"][1], expanded["C"][1]))

    def test_rejects_removing_existing_orbitals(self):
        coefficients = {"C": [torch.eye(3, 2, dtype=torch.float64)]}

        with self.assertRaisesRegex(ValueError, "remove"):
            MODULE.expand_coefficients(coefficients, (1,), element="C")


if __name__ == "__main__":
    unittest.main()
