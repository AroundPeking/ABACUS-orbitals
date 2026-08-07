import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from galerkin_sternheimer import evaluate_galerkin_response


class GalerkinSternheimerTest(unittest.TestCase):
    def test_two_level_analytic_response(self):
        overlap = torch.eye(2, dtype=torch.complex128)
        hamiltonian = torch.diag(
            torch.tensor([-0.5, 0.7], dtype=torch.float64)
        ).to(torch.complex128)
        perturbation = torch.tensor(
            [[[0.0, 0.3], [0.3, 0.0]]],
            dtype=torch.complex128,
        )
        occupation = torch.tensor([2.0, 0.0], dtype=torch.float64)
        frequency = torch.tensor([0.4], dtype=torch.float64)

        result = evaluate_galerkin_response(
            overlap,
            hamiltonian,
            perturbation,
            occupation,
            frequency,
        )

        delta = -0.3 / (0.7 - (-0.5) + 0.4j)
        expected_half = 2.0 * 0.3 * delta
        expected = expected_half + expected_half.conjugate()
        self.assertEqual(result.response_half.shape, (1, 1, 1))
        self.assertEqual(result.response.shape, (1, 1, 1))
        torch.testing.assert_close(
            result.response_half[0, 0, 0],
            torch.tensor(expected_half, dtype=torch.complex128),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            result.response[0, 0, 0],
            torch.tensor(expected, dtype=torch.complex128),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            result.response,
            result.response.mH,
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
