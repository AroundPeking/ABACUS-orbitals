"""Tests for compact LCAO compression of a grid Delta-ST response."""

import dataclasses
import pathlib
import sys
import unittest

import torch


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from delta_st_parent_space import DeltaSTReference, FullCoulombMatrix
from delta_st_response_compression import FrozenOccupiedDeltaSTCompression
from frozen_occupied_delta_st import evaluate_frozen_occupied_delta_st
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


def _fixture(overlap_coupling=0.0):
    s = float(overlap_coupling)
    overlap = torch.tensor(
        [[1.0, s, 0.0], [s, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.complex128,
    )
    hamiltonian = torch.tensor(
        [
            [-0.5, -0.5 * s, 0.0],
            [-0.5 * s, 0.2, 0.0],
            [0.0, 0.0, 0.8],
        ],
        dtype=torch.complex128,
    ).reshape(1, 3, 3)
    perturbation = torch.zeros((2, 3, 3), dtype=torch.complex128)
    perturbation[0, 0, 1] = perturbation[0, 1, 0] = 0.4
    perturbation[1, 0, 2] = perturbation[1, 2, 0] = 0.3
    channels = (
        AuxiliaryChannel(0, 0, 0, 0, 0, "H0_l0_n0_m0"),
        AuxiliaryChannel(1, 0, 1, 0, 0, "H0_l1_n0_m0"),
    )
    frequency = torch.tensor([0.2, 0.8], dtype=torch.float64)
    weight = torch.tensor([0.3, 0.7], dtype=torch.float64)
    primitive = SternheimerPrimitiveGalerkinData(
        format_version=1,
        representation="bessel_primitive_uniform_grid_gamma",
        energy_unit="Ha",
        blocks=(PrimitiveBlock("H", 0, 0, 0, 3, 0),),
        channels=channels,
        occupation=torch.tensor([[1.0]], dtype=torch.float64),
        overlap=overlap,
        hamiltonian_ha=hamiltonian,
        perturbation_ha=perturbation,
        primitive_ao_overlap=overlap[:, :1].clone(),
        fixed_ao_grid_overlap=torch.eye(1, dtype=torch.complex128),
        fixed_ao_grid_hamiltonian_ha=torch.tensor(
            [[[-0.5]]], dtype=torch.complex128
        ),
        frequency_ha=frequency,
        frequency_weight_ha=weight,
        provenance=_provenance(),
        primitive_ao_hamiltonian_ha=hamiltonian[:, :, :1].clone(),
        primitive_ao_perturbation_ha=perturbation[:, :, :1].clone(),
    )
    fixed = SternheimerFixedAOData(
        format_version=1,
        representation="fixed_lcao_gamma",
        energy_unit="Ha",
        channels=channels,
        eigenvalue_ha=torch.tensor([[-0.5]], dtype=torch.float64),
        occupation=torch.tensor([[1.0]], dtype=torch.float64),
        overlap=torch.eye(1, dtype=torch.complex128),
        hamiltonian_ha=torch.tensor([[[-0.5]]], dtype=torch.complex128),
        perturbation_ha=torch.zeros((2, 1, 1), dtype=torch.complex128),
        frequency_ha=frequency,
        frequency_weight_ha=weight,
        provenance=_provenance(),
    )
    identity = torch.eye(3, dtype=torch.complex128)
    full = evaluate_frozen_occupied_delta_st(primitive, fixed, identity)
    reference = DeltaSTReference(
        response_m=full.response,
        frequency_ha=frequency,
        frequency_weight_ha=weight,
        channels=channels,
        atom_naux=(2,),
        occupied_occupation_by_spin=((1.0,),),
        provenance=_provenance(),
    )
    coulomb = FullCoulombMatrix(
        matrix=torch.diag(
            torch.tensor([2.0, 4.0], dtype=torch.complex128)
        ),
        atom_naux=(2,),
        provenance=_provenance(),
    )
    return primitive, fixed, reference, coulomb


def _compact_coefficients(theta):
    zero = theta * 0.0
    one = zero + 1.0
    occupied = torch.stack((one, zero, zero)).to(torch.complex128)
    variable = torch.stack(
        (zero, torch.cos(theta), torch.sin(theta))
    ).to(torch.complex128)
    return torch.stack((occupied, variable), dim=1)


class DeltaSTResponseCompressionTest(unittest.TestCase):
    def test_variable_lcao_has_descent_gradient_toward_missing_response(self):
        primitive, fixed, reference, coulomb = _fixture()
        objective = FrozenOccupiedDeltaSTCompression(
            reference, primitive, fixed, coulomb
        )
        theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

        initial = objective.evaluate_matrix(_compact_coefficients(theta))
        initial.loss.backward()

        self.assertTrue(bool(torch.isfinite(theta.grad)))
        self.assertGreater(abs(float(theta.grad)), 1.0e-8)
        accepted = False
        for step in (0.2, 0.1, 0.05, 0.02, 0.01):
            trial_theta = theta.detach() - step * theta.grad.detach()
            trial = objective.evaluate_matrix(
                _compact_coefficients(trial_theta)
            )
            if float(trial.loss) < float(initial.loss):
                accepted = True
                break
        self.assertTrue(accepted)

    def test_occupied_projection_uses_hamiltonian_cross_block(self):
        primitive, fixed, reference, coulomb = _fixture(overlap_coupling=0.15)
        objective = FrozenOccupiedDeltaSTCompression(
            reference, primitive, fixed, coulomb
        )
        theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        baseline = objective.evaluate_matrix(_compact_coefficients(theta))
        baseline.loss.backward()
        baseline_gradient = theta.grad.detach().clone()

        changed_cross = primitive.primitive_ao_hamiltonian_ha.clone()
        changed_cross[0, 1, 0] += 0.25
        changed = dataclasses.replace(
            primitive,
            primitive_ao_hamiltonian_ha=changed_cross,
        )
        changed_objective = FrozenOccupiedDeltaSTCompression(
            reference, changed, fixed, coulomb
        )
        changed_theta = torch.tensor(
            0.3, dtype=torch.float64, requires_grad=True
        )
        changed_result = changed_objective.evaluate_matrix(
            _compact_coefficients(changed_theta)
        )
        changed_result.loss.backward()

        self.assertFalse(
            torch.allclose(
                changed_result.loss,
                baseline.loss,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        )
        self.assertFalse(
            torch.allclose(
                changed_theta.grad,
                baseline_gradient,
                rtol=1.0e-8,
                atol=1.0e-10,
            )
        )


if __name__ == "__main__":
    unittest.main()
