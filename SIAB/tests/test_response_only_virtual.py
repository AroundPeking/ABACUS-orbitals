import pathlib
import sys
import unittest

import torch


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

try:
    from response_only_virtual import solve_response_only_virtual_eigensystem
except ImportError:
    solve_response_only_virtual_eigensystem = None


class ResponseOnlyVirtualTest(unittest.TestCase):
    def test_exposes_response_only_virtual_solver(self):
        self.assertTrue(callable(solve_response_only_virtual_eigensystem))

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


if __name__ == "__main__":
    unittest.main()
