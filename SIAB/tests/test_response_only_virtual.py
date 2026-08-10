import pathlib
import sys
import unittest

import torch


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

try:
    import response_only_virtual
except ImportError:
    response_only_virtual = None

solve_response_only_virtual_eigensystem = (
    None
    if response_only_virtual is None
    else getattr(
        response_only_virtual,
        "solve_response_only_virtual_eigensystem",
        None,
    )
)
evaluate_response_only_sos = (
    None
    if response_only_virtual is None
    else getattr(response_only_virtual, "evaluate_response_only_sos", None)
)


class ResponseOnlyVirtualTest(unittest.TestCase):
    def test_exposes_response_only_virtual_solver(self):
        self.assertTrue(callable(solve_response_only_virtual_eigensystem))

    def test_exposes_response_only_spectral_response(self):
        self.assertTrue(callable(evaluate_response_only_sos))

    def test_keeps_fixed_occupied_state_and_diagonalizes_only_s_complement(self):
        overlap = torch.diag(
            torch.tensor([2.0, 1.0, 0.5], dtype=torch.complex128)
        )
        hamiltonian = torch.tensor(
            [
                [-1.0, 0.6, 0.0],
                [0.6, 0.3, 0.1],
                [0.0, 0.1, 0.4],
            ],
            dtype=torch.complex128,
        )
        occupied = torch.tensor(
            [[2.0**-0.5], [0.0], [0.0]], dtype=torch.complex128
        )
        occupied_energy = torch.tensor([-0.5], dtype=torch.float64)

        result = solve_response_only_virtual_eigensystem(
            overlap,
            hamiltonian,
            occupied,
            occupied_energy,
        )
        self.assertIsNotNone(result)

        expected_virtual_hamiltonian = torch.tensor(
            [[0.3, 2.0**0.5 * 0.1], [2.0**0.5 * 0.1, 0.8]],
            dtype=torch.complex128,
        )
        expected_virtual_energy = torch.linalg.eigvalsh(
            expected_virtual_hamiltonian
        )
        self.assertTrue(torch.equal(result.coefficient[:, :1], occupied))
        self.assertTrue(torch.equal(result.energy_ha[:1], occupied_energy))
        torch.testing.assert_close(
            result.virtual_energy_ha,
            expected_virtual_energy,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        torch.testing.assert_close(
            occupied.mH @ overlap @ result.virtual_coefficient,
            torch.zeros((1, 2), dtype=torch.complex128),
            rtol=0.0,
            atol=1.0e-13,
        )
        torch.testing.assert_close(
            result.virtual_coefficient.mH
            @ overlap
            @ result.virtual_coefficient,
            torch.eye(2, dtype=torch.complex128),
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        self.assertEqual(result.retained_virtual_rank, 2)
        self.assertEqual(result.dropped_trial_rank, 1)

    def test_spectral_response_has_sternheimer_sign_and_conjugate_factor(self):
        overlap = torch.eye(2, dtype=torch.complex128)
        hamiltonian = torch.tensor(
            [[-0.5, 0.7], [0.7, 0.3]], dtype=torch.complex128
        )
        occupied = torch.tensor([[1.0], [0.0]], dtype=torch.complex128)
        occupied_energy = torch.tensor([-0.5], dtype=torch.float64)
        eigensystem = solve_response_only_virtual_eigensystem(
            overlap,
            hamiltonian,
            occupied,
            occupied_energy,
        )
        perturbation = torch.tensor(
            [[[0.0, 0.4], [0.4, 0.0]]], dtype=torch.complex128
        )
        occupation = torch.tensor([1.0], dtype=torch.float64)
        frequency = torch.tensor([0.2], dtype=torch.float64)

        result = evaluate_response_only_sos(
            eigensystem,
            perturbation,
            occupation,
            frequency,
        )
        self.assertIsNotNone(result)

        denominator = torch.tensor(0.8 + 0.2j, dtype=torch.complex128)
        expected_half = (-0.16 / denominator).reshape(1, 1, 1)
        torch.testing.assert_close(result.response_half, expected_half)
        torch.testing.assert_close(
            result.response,
            expected_half + expected_half.mH,
        )


if __name__ == "__main__":
    unittest.main()
