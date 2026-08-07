import pathlib
import sys
import unittest

import torch


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from galerkin_sternheimer import evaluate_galerkin_response
from primitive_galerkin import (
    contract_primitive_matrices,
    evaluate_primitive_galerkin,
)


def hermitian(value):
    return value + value.mH


class PrimitiveGalerkinTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(41)
        raw = torch.randn((4, 4), dtype=torch.complex128, generator=generator)
        self.overlap = raw.mH @ raw + 2.0 * torch.eye(4, dtype=torch.complex128)
        self.hamiltonian = torch.stack(
            tuple(
                hermitian(
                    torch.randn(
                        (4, 4), dtype=torch.complex128, generator=generator
                    )
                )
                for _ in range(2)
            )
        )
        self.perturbation = torch.stack(
            tuple(
                hermitian(
                    torch.randn(
                        (4, 4), dtype=torch.complex128, generator=generator
                    )
                )
                for _ in range(3)
            )
        )
        self.coefficients = torch.tensor(
            [
                [1.0, 0.0],
                [0.2, 0.1],
                [0.0, 1.0],
                [-0.1, 0.3],
            ],
            dtype=torch.complex128,
        )

    def test_contracts_every_operator_by_the_same_congruence(self):
        actual = contract_primitive_matrices(
            self.overlap,
            self.hamiltonian,
            self.perturbation,
            self.coefficients,
        )
        c = self.coefficients

        torch.testing.assert_close(actual.overlap, c.mH @ self.overlap @ c)
        torch.testing.assert_close(
            actual.hamiltonian,
            torch.stack(tuple(c.mH @ value @ c for value in self.hamiltonian)),
        )
        torch.testing.assert_close(
            actual.perturbation,
            torch.stack(tuple(c.mH @ value @ c for value in self.perturbation)),
        )

    def test_response_matches_direct_finite_ao_evaluation(self):
        occupation = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        frequency = torch.tensor([0.2, 0.8], dtype=torch.float64)

        actual = evaluate_primitive_galerkin(
            self.overlap,
            self.hamiltonian,
            self.perturbation,
            self.coefficients,
            occupation,
            frequency,
        )
        contracted = contract_primitive_matrices(
            self.overlap,
            self.hamiltonian,
            self.perturbation,
            self.coefficients,
        )
        expected = evaluate_galerkin_response(
            contracted.overlap,
            contracted.hamiltonian[0],
            contracted.perturbation,
            occupation[0],
            frequency,
        )

        torch.testing.assert_close(actual.response, expected.response)
        self.assertEqual(actual.active_spin_count, 1)
        self.assertEqual(len(actual.overlap_condition_by_spin), 2)

    def test_response_has_gradient_with_respect_to_primitive_coefficients(self):
        coefficients = self.coefficients.clone().requires_grad_(True)
        occupation = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        frequency = torch.tensor([0.3], dtype=torch.float64)

        result = evaluate_primitive_galerkin(
            self.overlap,
            self.hamiltonian,
            self.perturbation,
            coefficients,
            occupation,
            frequency,
        )
        loss = torch.linalg.vector_norm(result.response)
        loss.backward()

        self.assertIsNotNone(coefficients.grad)
        self.assertTrue(bool(torch.all(torch.isfinite(coefficients.grad))))
        self.assertGreater(float(torch.linalg.vector_norm(coefficients.grad)), 0.0)

    def test_rejects_operator_or_occupation_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "hamiltonian shape"):
            contract_primitive_matrices(
                self.overlap,
                self.hamiltonian[:, :3, :3],
                self.perturbation,
                self.coefficients,
            )
        with self.assertRaisesRegex(ValueError, "occupation shape"):
            evaluate_primitive_galerkin(
                self.overlap,
                self.hamiltonian,
                self.perturbation,
                self.coefficients,
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
                torch.tensor([0.2], dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
