from pathlib import Path
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from IO.read_sternheimer_fixed_ao import read_sternheimer_fixed_ao


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "sternheimer_galerkin_fixed_ao_v1.dat"
)


class ReadSternheimerFixedAOTest(unittest.TestCase):
    def test_reads_version_one_fixed_ao_data(self):
        data = read_sternheimer_fixed_ao(FIXTURE)

        self.assertEqual(data.format_version, 1)
        self.assertEqual(data.representation, "fixed_lcao_gamma")
        self.assertEqual(data.energy_unit, "Ha")
        self.assertEqual(data.channels[0].label, "H0_l1_n2_m-1")
        self.assertEqual(data.eigenvalue_ha.shape, (2, 2))
        self.assertEqual(data.occupation.tolist(), [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(data.overlap.shape, (2, 2))
        self.assertEqual(data.hamiltonian_ha.shape, (2, 2, 2))
        self.assertEqual(data.perturbation_ha.shape, (1, 2, 2))
        self.assertEqual(data.frequency_ha.tolist(), [0.2, 0.8])
        self.assertEqual(data.frequency_weight_ha.tolist(), [0.3, 0.7])
        self.assertEqual(data.eigenvalue_ha.dtype, torch.float64)
        self.assertEqual(data.occupation.dtype, torch.float64)
        self.assertEqual(data.frequency_ha.dtype, torch.float64)
        self.assertEqual(data.frequency_weight_ha.dtype, torch.float64)
        self.assertEqual(data.overlap.dtype, torch.complex128)
        self.assertEqual(data.hamiltonian_ha.dtype, torch.complex128)
        self.assertEqual(data.perturbation_ha.dtype, torch.complex128)
        self.assertEqual(data.overlap.device.type, "cpu")
        self.assertEqual(data.provenance["kernel"], "full_coulomb")


if __name__ == "__main__":
    unittest.main()
