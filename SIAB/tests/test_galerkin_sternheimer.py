import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from galerkin_sternheimer import evaluate_galerkin_response


class GalerkinSternheimerTest(unittest.TestCase):
    def two_level_inputs(self):
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
        return (
            overlap,
            hamiltonian,
            perturbation,
            occupation,
            frequency,
        )

    def test_two_level_analytic_response(self):
        inputs = self.two_level_inputs()

        result = evaluate_galerkin_response(*inputs)

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

    def test_rejects_non_complex128_overlap(self):
        inputs = list(self.two_level_inputs())
        inputs[0] = inputs[0].to(torch.complex64)
        with self.assertRaisesRegex(ValueError, "overlap must have dtype"):
            evaluate_galerkin_response(*inputs)

    def test_rejects_non_hermitian_hamiltonian(self):
        inputs = list(self.two_level_inputs())
        inputs[1][0, 1] = 0.2j
        with self.assertRaisesRegex(ValueError, "hamiltonian must be Hermitian"):
            evaluate_galerkin_response(*inputs)

    def test_rejects_non_hermitian_perturbation(self):
        inputs = list(self.two_level_inputs())
        inputs[2][0, 0, 1] = 0.2j
        with self.assertRaisesRegex(ValueError, "perturbation must be Hermitian"):
            evaluate_galerkin_response(*inputs)

    def test_rejects_nonpositive_frequency(self):
        inputs = list(self.two_level_inputs())
        inputs[4] = torch.tensor([0.0], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "frequency_ha must be positive"):
            evaluate_galerkin_response(*inputs)

    def test_rejects_basis_without_virtual_state(self):
        inputs = list(self.two_level_inputs())
        inputs[3] = torch.tensor([2.0, 2.0], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "at least one virtual state"):
            evaluate_galerkin_response(*inputs)

    def test_rejects_rank_deficient_overlap(self):
        inputs = list(self.two_level_inputs())
        inputs[0] = torch.diag(
            torch.tensor([1.0, 0.0], dtype=torch.float64)
        ).to(torch.complex128)
        with self.assertRaisesRegex(RuntimeError, "overlap is rank deficient"):
            evaluate_galerkin_response(*inputs)

    def test_rejects_overlap_condition_above_limit(self):
        inputs = list(self.two_level_inputs())
        inputs[0] = torch.diag(
            torch.tensor([1.0, 1.0e-8], dtype=torch.float64)
        ).to(torch.complex128)
        with self.assertRaisesRegex(
            RuntimeError,
            "overlap condition number exceeds limit",
        ):
            evaluate_galerkin_response(*inputs, condition_limit=1.0e6)


if __name__ == "__main__":
    unittest.main()
