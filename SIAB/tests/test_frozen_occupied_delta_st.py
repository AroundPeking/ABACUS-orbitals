"""Tests for Delta-ST with fixed LCAO occupied states."""

import dataclasses
import pathlib
import sys
import unittest

import torch


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from frozen_occupied_delta_st import (
    _positive_metric_coordinate_transform,
    evaluate_frozen_occupied_delta_st,
)
from galerkin_sternheimer import evaluate_galerkin_response
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


def _inputs():
    frequency = torch.tensor([0.2, 0.8], dtype=torch.float64)
    weight = torch.tensor([0.3, 0.7], dtype=torch.float64)
    overlap = torch.eye(3, dtype=torch.complex128)
    grid_hamiltonian = torch.diag(
        torch.tensor([5.0, 0.2, 0.8], dtype=torch.complex128)
    ).reshape(1, 3, 3)
    perturbation = torch.zeros((2, 3, 3), dtype=torch.complex128)
    perturbation[0, 0, 1] = perturbation[0, 1, 0] = 0.4
    perturbation[1, 0, 2] = perturbation[1, 2, 0] = 0.3

    primitive = SternheimerPrimitiveGalerkinData(
        format_version=1,
        representation="bessel_primitive_uniform_grid_gamma",
        energy_unit="Ha",
        blocks=(PrimitiveBlock("H", 0, 0, 0, 3, 0),),
        channels=_channels(),
        occupation=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        overlap=overlap,
        hamiltonian_ha=grid_hamiltonian,
        perturbation_ha=perturbation,
        primitive_ao_overlap=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.complex128,
        ),
        fixed_ao_grid_overlap=torch.eye(2, dtype=torch.complex128),
        fixed_ao_grid_hamiltonian_ha=grid_hamiltonian[:, :2, :2].clone(),
        frequency_ha=frequency,
        frequency_weight_ha=weight,
        provenance=_provenance(),
        primitive_ao_hamiltonian_ha=grid_hamiltonian[:, :, :2].clone(),
        primitive_ao_perturbation_ha=perturbation[:, :, :2].clone(),
    )
    fixed = SternheimerFixedAOData(
        format_version=1,
        representation="fixed_lcao_gamma",
        energy_unit="Ha",
        channels=_channels(),
        eigenvalue_ha=torch.tensor([[-0.5, 0.2]], dtype=torch.float64),
        occupation=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        overlap=torch.eye(2, dtype=torch.complex128),
        hamiltonian_ha=torch.diag(
            torch.tensor([-0.5, 0.2], dtype=torch.complex128)
        ).reshape(1, 2, 2),
        perturbation_ha=perturbation[:, :2, :2].clone(),
        frequency_ha=frequency,
        frequency_weight_ha=weight,
        provenance=_provenance(),
    )
    return primitive, fixed, perturbation


class FrozenOccupiedDeltaSTTest(unittest.TestCase):
    def test_coordinate_metric_transform_has_finite_gradient_at_exact_degeneracy(self):
        scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        metric = scale.to(torch.complex128) * torch.diag(
            torch.tensor([1.0, 1.0, 0.0], dtype=torch.complex128)
        )

        transform, rank, dropped, condition = (
            _positive_metric_coordinate_transform(metric, 1.0e-8, 1.0e8)
        )
        transformed_metric = transform.mH @ metric @ transform
        loss = torch.sum(torch.abs(transform) ** 2)
        loss.backward()

        self.assertEqual(rank, 2)
        self.assertEqual(dropped, 1)
        self.assertEqual(condition, 1.0)
        torch.testing.assert_close(
            transformed_metric,
            torch.eye(2, dtype=torch.complex128),
        )
        self.assertTrue(torch.isfinite(scale.grad))

    def test_keeps_lcao_occupied_state_and_solves_only_its_orthogonal_complement(self):
        primitive, fixed, perturbation = _inputs()
        coefficients = torch.eye(3, dtype=torch.complex128)

        result = evaluate_frozen_occupied_delta_st(
            primitive,
            fixed,
            coefficients,
        )

        expected = evaluate_galerkin_response(
            torch.eye(3, dtype=torch.complex128),
            torch.diag(
                torch.tensor([-0.5, 0.2, 0.8], dtype=torch.complex128)
            ),
            perturbation,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            primitive.frequency_ha,
        )
        torch.testing.assert_close(result.response_half, expected.response_half)
        torch.testing.assert_close(result.response, expected.response)
        self.assertEqual(result.active_spin_count, 1)
        self.assertEqual(result.retained_parent_rank_by_spin, (2,))
        self.assertEqual(result.dropped_parent_rank_by_spin, (1,))
        self.assertLess(result.fixed_ao_eigenvalue_max_abs_error_ha, 1.0e-14)

    def test_requires_exact_cross_matrices(self):
        primitive, fixed, _ = _inputs()
        primitive = dataclasses.replace(
            primitive,
            primitive_ao_hamiltonian_ha=None,
            primitive_ao_perturbation_ha=None,
        )

        with self.assertRaisesRegex(ValueError, "cross matrices"):
            evaluate_frozen_occupied_delta_st(
                primitive,
                fixed,
                torch.eye(3, dtype=torch.complex128),
            )

    def test_can_retain_fixed_lcao_virtual_states_alongside_bessel_parent(self):
        primitive, fixed, perturbation = _inputs()
        coefficients = torch.eye(3, dtype=torch.complex128)[:, [0, 2]]

        bessel_only = evaluate_frozen_occupied_delta_st(
            primitive,
            fixed,
            coefficients,
        )
        augmented = evaluate_frozen_occupied_delta_st(
            primitive,
            fixed,
            coefficients,
            include_fixed_ao_virtual=True,
        )
        expected = evaluate_galerkin_response(
            torch.eye(3, dtype=torch.complex128),
            torch.diag(
                torch.tensor([-0.5, 0.2, 0.8], dtype=torch.complex128)
            ),
            perturbation,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            primitive.frequency_ha,
        )

        self.assertGreater(
            float(torch.max(torch.abs(bessel_only.response - expected.response))),
            1.0e-3,
        )
        torch.testing.assert_close(augmented.response_half, expected.response_half)
        torch.testing.assert_close(augmented.response, expected.response)
        self.assertEqual(augmented.retained_parent_rank_by_spin, (2,))
        self.assertEqual(augmented.dropped_parent_rank_by_spin, (1,))


if __name__ == "__main__":
    unittest.main()
