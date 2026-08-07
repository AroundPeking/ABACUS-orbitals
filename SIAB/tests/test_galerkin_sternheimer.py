import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from galerkin_sternheimer import (
    evaluate_galerkin_response,
    evaluate_sos_response,
)


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

    def dense_inputs(self):
        generator = torch.Generator().manual_seed(731)

        def random_complex(*shape):
            real = torch.randn(*shape, generator=generator, dtype=torch.float64)
            imag = torch.randn(*shape, generator=generator, dtype=torch.float64)
            return torch.complex(real, imag)

        basis_transform = torch.eye(4, dtype=torch.complex128)
        basis_transform = basis_transform + 0.12 * random_complex(4, 4)
        overlap = basis_transform.mH @ basis_transform

        unitary, _ = torch.linalg.qr(random_complex(4, 4))
        energy = torch.tensor([-0.8, -0.3, 0.4, 1.2], dtype=torch.float64)
        orthonormal_hamiltonian = (
            unitary @ torch.diag(energy).to(torch.complex128) @ unitary.mH
        )
        hamiltonian = (
            basis_transform.mH
            @ orthonormal_hamiltonian
            @ basis_transform
        )

        perturbation = random_complex(3, 4, 4)
        perturbation = (perturbation + perturbation.mH) / 2.0
        perturbation = torch.stack(
            tuple(
                basis_transform.mH @ value @ basis_transform
                for value in perturbation
            )
        )
        occupation = torch.tensor([2.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        frequency = torch.tensor([0.2, 1.3], dtype=torch.float64)
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

    def test_dense_complex_galerkin_matches_full_virtual_sos(self):
        inputs = self.dense_inputs()

        galerkin = evaluate_galerkin_response(*inputs)
        sos = evaluate_sos_response(*inputs)

        difference = galerkin.response - sos.response
        relative = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            sos.response
        )
        maximum_absolute = torch.max(torch.abs(difference))
        self.assertLess(float(relative), 1.0e-11)
        self.assertLess(float(maximum_absolute), 1.0e-12)
        torch.testing.assert_close(
            galerkin.response_half,
            sos.response_half,
            rtol=1.0e-11,
            atol=1.0e-12,
        )
        for result in (galerkin, sos):
            torch.testing.assert_close(
                result.response,
                result.response.mH,
                rtol=0.0,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
