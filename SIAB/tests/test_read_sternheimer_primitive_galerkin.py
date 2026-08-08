from pathlib import Path
import tempfile
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
        self.assertIsNone(data.primitive_ao_hamiltonian_ha)
        self.assertIsNone(data.primitive_ao_perturbation_ha)
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

    def test_reads_optional_frozen_occupied_cross_matrices(self):
        text = FIXTURE.read_text(encoding="ascii")
        hamiltonian = "\n".join(
            f"{value} 0.0"
            for value in (
                -0.5,
                0.0,
                0.0,
                0.6,
                0.0,
                0.0,
                -0.4,
                0.0,
                0.0,
                0.7,
                0.0,
                0.0,
            )
        )
        perturbation = "\n".join(
            f"{value} 0.0" for value in (0.2, 0.0, 0.0, -0.1, 0.0, 0.0)
        )
        text = text.replace(
            "</HAMILTONIAN_H>",
            "</HAMILTONIAN_H>\n<PRIMITIVE_AO_HAMILTONIAN>\n"
            + hamiltonian
            + "\n</PRIMITIVE_AO_HAMILTONIAN>",
        )
        text = text.replace(
            "</PERTURBATION_V>",
            "</PERTURBATION_V>\n<PRIMITIVE_AO_PERTURBATION>\n"
            + perturbation
            + "\n</PRIMITIVE_AO_PERTURBATION>",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.dat"
            path.write_text(text, encoding="ascii")
            data = read_sternheimer_primitive_galerkin(path)

        self.assertEqual(data.primitive_ao_hamiltonian_ha.shape, (2, 3, 2))
        self.assertEqual(data.primitive_ao_perturbation_ha.shape, (1, 3, 2))
        self.assertEqual(data.primitive_ao_hamiltonian_ha[0, 0, 0], -0.5)
        self.assertEqual(data.primitive_ao_perturbation_ha[0, 1, 1], -0.1)


if __name__ == "__main__":
    unittest.main()
