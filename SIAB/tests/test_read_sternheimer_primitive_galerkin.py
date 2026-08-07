from pathlib import Path
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from IO.read_sternheimer_primitive_galerkin import (
    read_sternheimer_primitive_galerkin,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "sternheimer_galerkin_primitive_v1.dat"
)


class ReadSternheimerPrimitiveGalerkinTest(unittest.TestCase):
    def test_reads_version_one_primitive_data(self):
        data = read_sternheimer_primitive_galerkin(FIXTURE)

        self.assertEqual(data.format_version, 1)
        self.assertEqual(
            data.representation, "bessel_primitive_uniform_grid_gamma"
        )
        self.assertEqual(data.energy_unit, "Ha")
        self.assertEqual(data.blocks[0].key, ("H", 0, 0, 0))
        self.assertEqual(data.channels[0].label, "H0_l1_n2_m-1")
        self.assertEqual(data.occupation.tolist(), [[1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(data.overlap.shape, (3, 3))
        self.assertEqual(data.hamiltonian_ha.shape, (2, 3, 3))
        self.assertEqual(data.perturbation_ha.shape, (1, 3, 3))
        self.assertEqual(data.primitive_ao_overlap.shape, (3, 2))
        self.assertEqual(data.fixed_ao_grid_overlap.shape, (2, 2))
        self.assertEqual(data.fixed_ao_grid_hamiltonian_ha.shape, (2, 2, 2))
        self.assertEqual(data.frequency_ha.tolist(), [0.2, 0.8])
        self.assertEqual(data.frequency_weight_ha.tolist(), [0.3, 0.7])
        for value in (
            data.overlap,
            data.hamiltonian_ha,
            data.perturbation_ha,
            data.primitive_ao_overlap,
            data.fixed_ao_grid_overlap,
            data.fixed_ao_grid_hamiltonian_ha,
        ):
            self.assertEqual(value.dtype, torch.complex128)
            self.assertEqual(value.device.type, "cpu")
        self.assertEqual(data.provenance["kernel"], "full_coulomb")


if __name__ == "__main__":
    unittest.main()
