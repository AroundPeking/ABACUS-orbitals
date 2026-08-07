from dataclasses import replace
from pathlib import Path
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from fixed_ao_galerkin import evaluate_fixed_ao_sidecar
from galerkin_sternheimer import evaluate_galerkin_response, evaluate_sos_response
from IO.read_sternheimer_fixed_ao import read_sternheimer_fixed_ao


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "sternheimer_galerkin_fixed_ao_v1.dat"
)


class FixedAOGalerkinTest(unittest.TestCase):
    def test_sums_spin_channels_without_an_extra_spin_factor(self):
        data = read_sternheimer_fixed_ao(FIXTURE)

        result = evaluate_fixed_ao_sidecar(data)

        manual_galerkin = sum(
            (
                evaluate_galerkin_response(
                    data.overlap,
                    data.hamiltonian_ha[spin],
                    data.perturbation_ha,
                    data.occupation[spin],
                    data.frequency_ha,
                ).response
                for spin in range(data.hamiltonian_ha.shape[0])
            ),
            torch.zeros_like(result.galerkin_response),
        )
        manual_sos = sum(
            (
                evaluate_sos_response(
                    data.overlap,
                    data.hamiltonian_ha[spin],
                    data.perturbation_ha,
                    data.occupation[spin],
                    data.frequency_ha,
                ).response
                for spin in range(data.hamiltonian_ha.shape[0])
            ),
            torch.zeros_like(result.sos_response),
        )
        torch.testing.assert_close(result.galerkin_response, manual_galerkin)
        torch.testing.assert_close(result.sos_response, manual_sos)
        self.assertLess(result.galerkin_sos_relative_error, 1.0e-12)
        self.assertLess(result.galerkin_sos_max_abs_error, 1.0e-12)
        self.assertGreater(result.eigenvalue_max_abs_error_ha, 0.0)
        self.assertEqual(len(result.overlap_condition_by_spin), 2)

    def test_empty_spin_channel_contributes_zero_response(self):
        data = read_sternheimer_fixed_ao(FIXTURE)
        occupation = data.occupation.clone()
        occupation[1] = 0.0
        data = replace(data, occupation=occupation)

        result = evaluate_fixed_ao_sidecar(data)
        spin_zero = evaluate_galerkin_response(
            data.overlap,
            data.hamiltonian_ha[0],
            data.perturbation_ha,
            data.occupation[0],
            data.frequency_ha,
        )

        torch.testing.assert_close(result.galerkin_response, spin_zero.response)
        torch.testing.assert_close(result.sos_response, spin_zero.response)
        self.assertEqual(len(result.overlap_condition_by_spin), 2)


if __name__ == "__main__":
    unittest.main()
