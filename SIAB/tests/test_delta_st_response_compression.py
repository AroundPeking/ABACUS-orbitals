"""Tests for compact LCAO compression of a grid Delta-ST response."""

import dataclasses
import io
import pathlib
import sys
import unittest

import torch


TEST_DIR = pathlib.Path(__file__).resolve().parent
OPT_DIR = TEST_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from delta_st_parent_space import DeltaSTReference, FullCoulombMatrix
from delta_st_response_compression import (
    FrozenOccupiedDeltaSTCompression,
    anchor_atomic_occupied_radial,
)
from frozen_occupied_delta_st import evaluate_frozen_occupied_delta_st
from common import info
from optimization_loss import normalize_loss_config
from opt_orbital_converge import Opt_Orbital_Converge
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


def _radial_coefficients(theta=0.2):
    return {
        "H": [
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, torch.cos(torch.tensor(theta)).item()],
                    [0.0, torch.sin(torch.tensor(theta)).item()],
                ],
                dtype=torch.float64,
                requires_grad=True,
            )
        ]
    }


def _converge(objective, max_steps=8):
    info_stru = [info(Na={"H": 1}, Nb_true=1, weight=torch.tensor([1.0]))]
    info_element = {"H": info(Nl=1, Ne=3, Nu=[2])}
    source = {"H": [torch.tensor([[0.8, 0.4, 0.2]], dtype=torch.complex128)]}
    overlap = {
        ("H", "H"): [
            [torch.eye(3, dtype=torch.complex128).reshape(1, 1, 3, 1, 1, 3)]
        ]
    }
    stage = {
        "optimizer": "Adam",
        "kwargs": {"lr": 0.03},
        "cal_T": False,
        "norm": "one",
        "max_steps": max_steps,
        "loss": normalize_loss_config(
            {
                "mode": "pi_dpsi_joint",
                "joint_dpsi_weight": 0.0,
                "tau_dft": 1.0e6,
                "tau_dpsi": 1.0e6,
                "constraint_penalty_dft": 0.0,
                "constraint_penalty_dpsi": 0.0,
            }
        ),
    }
    converge = Opt_Orbital_Converge()
    converge.set_info(
        {"origin": ["synthetic"]},
        [stage],
        info_stru,
        {
            "init_from_file": False,
            "freeze_orbitals": [{"element": "H", "l": 0, "zeta": 1}],
        },
        {"same_band": True},
    )
    converge.set_info_element(info_element)
    converge.set_QSVI(
        [source],
        [overlap],
        [torch.tensor([1.0], dtype=torch.float64)],
    )
    converge.set_projected_pi_objective(objective)
    return converge


class DeltaSTResponseCompressionTest(unittest.TestCase):
    def test_atomic_occupied_anchor_preserves_initial_span_and_exact_occupancy(self):
        primitive, fixed, _, _ = _fixture()
        inverse_sqrt_two = 2.0 ** -0.5
        eigenvector = torch.tensor(
            [
                [inverse_sqrt_two, inverse_sqrt_two, 0.0],
                [inverse_sqrt_two, -inverse_sqrt_two, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.complex128,
        )
        fixed = SternheimerFixedAOData(
            format_version=1,
            representation="fixed_lcao_gamma",
            energy_unit="Ha",
            channels=fixed.channels,
            eigenvalue_ha=torch.tensor([[-0.5, 0.2, 0.8]], dtype=torch.float64),
            occupation=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
            overlap=torch.eye(3, dtype=torch.complex128),
            hamiltonian_ha=(
                eigenvector
                @ torch.diag(
                    torch.tensor([-0.5, 0.2, 0.8], dtype=torch.complex128)
                )
                @ eigenvector.mH
            ).reshape(1, 3, 3),
            perturbation_ha=torch.zeros((2, 3, 3), dtype=torch.complex128),
            frequency_ha=fixed.frequency_ha,
            frequency_weight_ha=fixed.frequency_weight_ha,
            provenance=fixed.provenance,
        )
        coefficients = {
            "H": [
                torch.eye(3, dtype=torch.float64, requires_grad=True)
            ]
        }

        anchored, metadata = anchor_atomic_occupied_radial(
            primitive,
            fixed,
            coefficients,
            element="H",
        )

        expected_occupied = eigenvector[:, 0].real
        torch.testing.assert_close(
            anchored["H"][0][:, 0], expected_occupied
        )
        anchored_s = anchored["H"][0]
        projector = (
            anchored_s
            @ torch.linalg.inv(anchored_s.mT @ anchored_s)
            @ anchored_s.mT
        )
        torch.testing.assert_close(projector, torch.eye(3, dtype=torch.float64))
        self.assertEqual(metadata.occupied_band_index, 0)
        self.assertEqual(metadata.omitted_original_s_zeta, 1)
        self.assertLess(metadata.maximum_off_s_coefficient, 1.0e-14)
        self.assertTrue(anchored["H"][0].requires_grad)

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

    def test_joint_optimizer_moves_only_variable_lcao_and_lowers_response_loss(self):
        primitive, fixed, reference, coulomb = _fixture()
        objective = FrozenOccupiedDeltaSTCompression(
            reference, primitive, fixed, coulomb
        )
        coefficients = _radial_coefficients()
        fixed_before = coefficients["H"][0][:, 0].detach().clone()
        variable_before = coefficients["H"][0][:, 1].detach().clone()
        initial_loss = float(objective.evaluate(coefficients).loss)

        result = _converge(objective).cal_converge(
            coefficients,
            [io.StringIO(), io.StringIO()],
        )

        self.assertTrue(torch.equal(result["C"]["H"][0][:, 0], fixed_before))
        self.assertFalse(torch.equal(result["C"]["H"][0][:, 1], variable_before))
        self.assertLess(result["loss_components"]["projected_pi"], initial_loss)


if __name__ == "__main__":
    unittest.main()
