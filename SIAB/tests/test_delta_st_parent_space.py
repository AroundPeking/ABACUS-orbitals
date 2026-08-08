"""Tests for full-grid Delta-ST parent-space convergence analysis."""

import dataclasses
import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest

import torch


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from delta_st_parent_space import (
    DeltaSTReference,
    FullCoulombMatrix,
    analyze_frozen_occupied_parent_response,
    analyze_parent_response,
    build_parent_coefficients,
    build_parent_occupation,
    integrate_trace_log_integrand,
    load_delta_st_reference,
    rpa_correlation_energy,
    symmetric_response,
    validate_parent_space_protocol,
)
from primitive_galerkin import evaluate_primitive_galerkin
from sternheimer_data import PrimitiveBlock
from sternheimer_fixed_ao_data import AuxiliaryChannel, SternheimerFixedAOData
from sternheimer_primitive_galerkin_data import SternheimerPrimitiveGalerkinData


def _provenance():
    return {
        "abacus_commit": "1" * 40,
        "auxiliary_basis_sha256": "a" * 64,
        "cell_bohr": [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0],
        "ecut_ry": 100.0,
        "kernel": "full_coulomb",
        "orbital_sha256": "b" * 64,
        "pseudopotential_sha256": "c" * 64,
        "spin_convention": "occupation_in_metadata",
    }


def _channels():
    return (
        AuxiliaryChannel(0, 0, 0, 0, 0, "H0_l0_n0_m0"),
        AuxiliaryChannel(1, 0, 1, 0, 0, "H0_l1_n0_m0"),
    )


def _primitive_data():
    overlap = torch.eye(4, dtype=torch.complex128)
    hamiltonian = torch.stack(
        (
            torch.diag(
                torch.tensor([-0.5, 0.2, 0.8, 1.4], dtype=torch.complex128)
            ),
            torch.diag(
                torch.tensor([-0.4, 0.3, 0.9, 1.5], dtype=torch.complex128)
            ),
        )
    )
    perturbation = torch.zeros((2, 4, 4), dtype=torch.complex128)
    perturbation[0, 0, 1] = perturbation[0, 1, 0] = 0.4
    perturbation[1, 0, 2] = perturbation[1, 2, 0] = 0.3
    return SternheimerPrimitiveGalerkinData(
        format_version=1,
        representation="bessel_primitive_uniform_grid_gamma",
        energy_unit="Ha",
        blocks=(
            PrimitiveBlock("H", 0, 0, 0, 2, 0),
            PrimitiveBlock("H", 0, 1, 0, 2, 2),
        ),
        channels=_channels(),
        occupation=torch.tensor(
            [[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64
        ),
        overlap=overlap,
        hamiltonian_ha=hamiltonian,
        perturbation_ha=perturbation,
        primitive_ao_overlap=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=torch.complex128,
        ),
        fixed_ao_grid_overlap=torch.eye(2, dtype=torch.complex128),
        fixed_ao_grid_hamiltonian_ha=hamiltonian[:, :2, :2].clone(),
        frequency_ha=torch.tensor([0.2, 0.8], dtype=torch.float64),
        frequency_weight_ha=torch.tensor([0.3, 0.7], dtype=torch.float64),
        provenance=_provenance(),
        primitive_ao_hamiltonian_ha=hamiltonian[:, :, :2].clone(),
        primitive_ao_perturbation_ha=perturbation[:, :, :2].clone(),
    )


def _reference(data):
    occupation = build_parent_occupation(data, data.overlap.shape[0])
    identity = torch.eye(data.overlap.shape[0], dtype=torch.complex128)
    response = evaluate_primitive_galerkin(
        data.overlap,
        data.hamiltonian_ha,
        data.perturbation_ha,
        identity,
        occupation,
        data.frequency_ha,
    ).response
    return DeltaSTReference(
        response_m=response,
        frequency_ha=data.frequency_ha,
        frequency_weight_ha=data.frequency_weight_ha,
        channels=data.channels,
        atom_naux=(2,),
        occupied_occupation_by_spin=((1.0,), ()),
        provenance=dict(data.provenance),
    )


def _fixed_ao_data(data):
    return SternheimerFixedAOData(
        format_version=1,
        representation="fixed_lcao_gamma",
        energy_unit="Ha",
        channels=data.channels,
        eigenvalue_ha=torch.tensor(
            [[-0.5, 0.2], [-0.4, 0.3]], dtype=torch.float64
        ),
        occupation=data.occupation,
        overlap=torch.eye(2, dtype=torch.complex128),
        hamiltonian_ha=data.fixed_ao_grid_hamiltonian_ha,
        perturbation_ha=data.perturbation_ha[:, :2, :2].clone(),
        frequency_ha=data.frequency_ha,
        frequency_weight_ha=data.frequency_weight_ha,
        provenance=dict(data.provenance),
    )


class DeltaSTParentSpaceTest(unittest.TestCase):
    def test_protocol_requires_identical_physical_metadata(self):
        data = _primitive_data()
        reference = _reference(data)
        coulomb = FullCoulombMatrix(
            matrix=torch.diag(
                torch.tensor([2.0, 4.0], dtype=torch.complex128)
            ),
            atom_naux=(2,),
            provenance=dict(data.provenance),
        )
        validate_parent_space_protocol(reference, data, coulomb)

        cases = (
            (
                "frequency",
                dataclasses.replace(
                    reference,
                    frequency_ha=torch.tensor([0.2, 0.9], dtype=torch.float64),
                ),
                data,
                coulomb,
            ),
            (
                "auxiliary channel",
                dataclasses.replace(reference, channels=tuple(reversed(_channels()))),
                data,
                coulomb,
            ),
            (
                "spin occupation",
                dataclasses.replace(
                    reference,
                    occupied_occupation_by_spin=((0.5,), ()),
                ),
                data,
                coulomb,
            ),
            (
                "Coulomb dimension",
                reference,
                data,
                dataclasses.replace(
                    coulomb,
                    matrix=torch.eye(3, dtype=torch.complex128),
                    atom_naux=(3,),
                ),
            ),
            (
                "kernel",
                dataclasses.replace(
                    reference,
                    provenance={**reference.provenance, "kernel": "cut_coulomb"},
                ),
                data,
                coulomb,
            ),
            (
                "physical provenance",
                reference,
                dataclasses.replace(
                    data,
                    provenance={**data.provenance, "ecut_ry": 80.0},
                ),
                coulomb,
            ),
        )
        for message, changed_reference, changed_data, changed_coulomb in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_parent_space_protocol(
                        changed_reference, changed_data, changed_coulomb
                    )

    def test_symmetric_response_and_trace_log_use_retained_coulomb_space(self):
        coulomb = torch.diag(
            torch.tensor([0.0, 4.0], dtype=torch.complex128)
        )
        response_m = torch.zeros((2, 2, 2), dtype=torch.complex128)
        response_m[0, 1, 1] = -0.8
        response_m[1, 1, 1] = -0.4

        pi, metadata = symmetric_response(
            coulomb, response_m, eigenvalue_threshold=1.0e-12
        )

        self.assertEqual(tuple(pi.shape), (2, 1, 1))
        self.assertEqual(metadata.retained_rank, 1)
        self.assertEqual(metadata.dropped_rank, 1)
        torch.testing.assert_close(
            pi[:, 0, 0], torch.tensor([-0.2, -0.1], dtype=torch.complex128)
        )
        weight = torch.tensor([0.3, 0.7], dtype=torch.float64)
        expected = (
            0.3 * (math.log(1.2) - 0.2)
            + 0.7 * (math.log(1.1) - 0.1)
        ) / (2.0 * math.pi)
        self.assertAlmostEqual(rpa_correlation_energy(pi, weight), expected, 14)

    def test_identity_parent_matches_direct_primitive_galerkin_response(self):
        data = _primitive_data()
        reference = _reference(data)
        coulomb = FullCoulombMatrix(
            matrix=torch.diag(
                torch.tensor([2.0, 4.0], dtype=torch.complex128)
            ),
            atom_naux=(2,),
            provenance=dict(data.provenance),
        )

        result = analyze_parent_response(
            reference,
            data,
            coulomb,
            radial_count=None,
            lmax=1,
        )

        self.assertEqual(result.parent_dimension, 4)
        self.assertLess(result.maximum_pi_relative_frobenius, 1.0e-13)
        self.assertLess(abs(result.energy_error_ha), 1.0e-14)

    def test_frozen_occupied_parent_matches_full_reference(self):
        data = _primitive_data()
        reference = _reference(data)
        coulomb = FullCoulombMatrix(
            matrix=torch.diag(
                torch.tensor([2.0, 4.0], dtype=torch.complex128)
            ),
            atom_naux=(2,),
            provenance=dict(data.provenance),
        )

        result = analyze_frozen_occupied_parent_response(
            reference,
            data,
            _fixed_ao_data(data),
            coulomb,
            radial_count=None,
            lmax=1,
        )

        self.assertEqual(result.parent_dimension, 4)
        self.assertLess(result.maximum_pi_relative_frobenius, 1.0e-13)
        self.assertLess(abs(result.energy_error_ha), 1.0e-14)

    def test_trace_log_integration_reproduces_existing_fixed_ao_librpa_result(self):
        weight = torch.tensor(
            [
                0.1387508132744379,
                0.1552451503581690,
                0.1907006349403112,
                0.2503660767714552,
                0.3430536112358677,
                0.4825994565315100,
                0.6903117179895282,
                0.9989890446239748,
                1.4597439214684820,
                2.1545449390627964,
                3.2224752322022949,
                4.9252429196544769,
                7.8468534661184357,
                13.6580038013319722,
                29.2016164564687166,
                110.6229955908527671,
            ],
            dtype=torch.float64,
        )
        integrand = torch.tensor(
            [
                -1.6077188242274640e-01,
                -1.2212679223171916e-01,
                -7.7232900322834100e-02,
                -4.6156502711458505e-02,
                -2.7976404939615621e-02,
                -1.6517965414786309e-02,
                -8.5627596803143149e-03,
                -3.5934618549848152e-03,
                -1.2050741515642260e-03,
                -3.3638438389903638e-04,
                -8.2308272844396015e-05,
                -1.8236796166142748e-05,
                -3.6576117832542454e-06,
                -6.2989181684219751e-07,
                -7.8095339668350934e-08,
                -3.6414671488141971e-09,
            ],
            dtype=torch.float64,
        )

        energy = integrate_trace_log_integrand(integrand, weight)

        self.assertAlmostEqual(energy, -0.015517587319549305, 15)
        self.assertEqual(f"{energy:.9f}", "-0.015517587")

    def test_parent_selection_is_block_local_in_radial_index_and_lmax(self):
        data = _primitive_data()
        coefficients = build_parent_coefficients(
            data, radial_count=1, lmax=1
        )
        expected = torch.tensor(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(coefficients, expected)
        coefficients_l0 = build_parent_coefficients(
            data, radial_count=2, lmax=0
        )
        torch.testing.assert_close(
            coefficients_l0, torch.eye(4, dtype=torch.complex128)[:, :2]
        )

    def test_binary_reference_parser_preserves_frequency_and_channel_order(self):
        provenance = _provenance()
        manifest = {
            "occupied_occupation_by_spin": [[1.0], []],
            "provenance": provenance,
        }
        with tempfile.TemporaryDirectory() as directory_text:
            directory = pathlib.Path(directory_text)
            (directory / "reference_protocol.json").write_text(
                json.dumps(manifest), encoding="ascii"
            )
            (directory / "STERNHEIMER_ABFS_CHANNELS.dat").write_text(
                "# channel atom atom_local type l radial m label max_abs\n"
                "0 0 0 0 0 0 0 H0_l0_n0_m0 1.0\n"
                "1 0 1 0 1 0 1 H0_l1_n0_m0 1.0\n",
                encoding="ascii",
            )
            _write_response_shard(directory, 1, 0.2, 0.3, [-0.2, -0.1])
            _write_response_shard(directory, 2, 0.8, 0.7, [-0.1, -0.05])

            reference = load_delta_st_reference(directory)

        torch.testing.assert_close(
            reference.frequency_ha, torch.tensor([0.2, 0.8], dtype=torch.float64)
        )
        torch.testing.assert_close(
            reference.frequency_weight_ha,
            torch.tensor([0.3, 0.7], dtype=torch.float64),
        )
        self.assertEqual(reference.channels, _channels())
        self.assertEqual(reference.atom_naux, (2,))
        torch.testing.assert_close(
            reference.response_m[:, 0, 0],
            torch.tensor([-0.2, -0.1], dtype=torch.complex128),
        )


def _write_response_shard(directory, ifreq, omega, weight, diagonal):
    path = directory / f"v1_sternheimer_chi0_iq_1_ifreq_{ifreq}_rank0.dat"
    header = struct.pack(
        "<6i2di", -41073291, 1, ifreq, 2, 1, 1, omega, weight, 1
    )
    atom_naux = struct.pack("<i", 2)
    table_size = struct.calcsize("<iq")
    payload_offset = len(header) + len(atom_naux) + table_size
    table = struct.pack("<iq", 0, payload_offset)
    matrix = (complex(diagonal[0]), 0.0j, 0.0j, complex(diagonal[1]))
    payload = b"".join(struct.pack("<2d", value.real, value.imag) for value in matrix)
    path.write_bytes(header + atom_naux + table + payload)


if __name__ == "__main__":
    unittest.main()
