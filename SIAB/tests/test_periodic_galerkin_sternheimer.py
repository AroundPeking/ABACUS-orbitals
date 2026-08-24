import unittest
from dataclasses import replace

import torch

import common  # noqa: F401 - configures the optimizer import path
from periodic_galerkin_data import (
    PeriodicGalerkinDataset,
    PeriodicGalerkinKPoint,
    PeriodicGalerkinPrimitiveBlock,
)
from periodic_galerkin_sternheimer import evaluate_periodic_galerkin_response


class PeriodicGalerkinSternheimerTest(unittest.TestCase):
    def complete_two_level_dataset(self):
        omega = 0.4
        delta = -0.3 / (1.2 + 1.0j * omega)
        half = 2.0 * 0.3 * delta
        response = half + half.conjugate()
        record = PeriodicGalerkinKPoint(
            source_ik=1,
            target_ik=1,
            source_kpoint=(0.0, 0.0, 0.0),
            target_kpoint=(0.0, 0.0, 0.0),
            reciprocal_shift=(0, 0, 0),
            k_weight=2.0,
            occupation=torch.tensor([1.0], dtype=torch.float64),
            source_eigenvalue_ha=torch.tensor([-0.5], dtype=torch.float64),
            overlap=torch.eye(2, dtype=torch.complex128),
            hamiltonian_ha=torch.diag(
                torch.tensor([-0.5, 0.7], dtype=torch.float64)
            ).to(torch.complex128),
            occupied_projection=torch.tensor([[1.0, 0.0]], dtype=torch.complex128),
            source=torch.tensor([[[0.0, 0.3]]], dtype=torch.complex128),
            reference_projection=torch.tensor(
                [[[[0.0, delta.conjugate()]]]], dtype=torch.complex128
            ),
        )
        dataset = PeriodicGalerkinDataset(
            abacus_commit="1" * 40,
            executable_sha256="2" * 64,
            orbital_sha256="3" * 64,
            pseudopotential_sha256="4" * 64,
            auxiliary_basis_sha256="5" * 64,
            primitive_blocks_sha256="6" * 64,
            physics_hash="7" * 64,
            selected_iq=1,
            q_count=1,
            qpoint=(0.0, 0.0, 0.0),
            q_weight=1.0,
            primitive_count=2,
            raw_auxiliary_dimension=1,
            whitened_auxiliary_rank=1,
            frequency_ha=torch.tensor([omega], dtype=torch.float64),
            frequency_weights_ha=torch.tensor([1.0], dtype=torch.float64),
            coulomb_metric=torch.eye(1, dtype=torch.complex128),
            coulomb_whitening=torch.eye(1, dtype=torch.complex128),
            reference_response=torch.tensor([[[response]]], dtype=torch.complex128),
            primitive_blocks=(
                PeriodicGalerkinPrimitiveBlock("C", 0, 0, 0, 2, 0),
            ),
            kpoints=(record,),
        )
        return dataset, delta, response

    def test_complete_candidate_reproduces_exact_response_and_projection(self):
        dataset, delta, expected_response = self.complete_two_level_dataset()

        result = evaluate_periodic_galerkin_response(
            dataset, torch.eye(2, dtype=torch.complex128)
        )

        torch.testing.assert_close(
            result.response[0, 0, 0],
            torch.tensor(expected_response, dtype=torch.complex128),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            result.projected_response[0][0, 0, 0],
            torch.tensor([0.0, delta.conjugate()], dtype=torch.complex128),
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        self.assertLess(float(result.relative_response_error), 1.0e-14)
        self.assertLess(float(result.relative_projection_error), 1.0e-14)
        self.assertGreater(result.minimum_occupied_capture, 1.0 - 1.0e-14)

    def test_response_is_invariant_under_invertible_candidate_coordinates(self):
        dataset, _, _ = self.complete_two_level_dataset()
        transform = torch.tensor(
            [[1.0 + 0.0j, 0.2 - 0.1j], [0.1 + 0.05j, 1.1 + 0.0j]],
            dtype=torch.complex128,
        )

        reference = evaluate_periodic_galerkin_response(
            dataset, torch.eye(2, dtype=torch.complex128)
        )
        changed = evaluate_periodic_galerkin_response(dataset, transform)

        torch.testing.assert_close(
            changed.response, reference.response, rtol=1.0e-12, atol=1.0e-13
        )

    def test_rejects_candidate_that_does_not_capture_fixed_occupied_state(self):
        dataset, _, _ = self.complete_two_level_dataset()
        virtual_only = torch.tensor([[0.0], [1.0]], dtype=torch.complex128)

        with self.assertRaisesRegex(RuntimeError, "fixed occupied manifold"):
            evaluate_periodic_galerkin_response(dataset, virtual_only)

    def test_response_loss_has_finite_nonzero_basis_gradient(self):
        dataset, _, _ = self.complete_two_level_dataset()
        omega = float(dataset.frequency_ha[0])
        delta_1 = -0.3 / (1.2 + 1.0j * omega)
        delta_2 = -0.2 / (1.9 + 1.0j * omega)
        response = 2.0 * (0.3 * delta_1 + 0.2 * delta_2)
        response = response + response.conjugate()
        record = replace(
            dataset.kpoints[0],
            overlap=torch.eye(3, dtype=torch.complex128),
            hamiltonian_ha=torch.diag(
                torch.tensor([-0.5, 0.7, 1.4], dtype=torch.float64)
            ).to(torch.complex128),
            occupied_projection=torch.tensor(
                [[1.0, 0.0, 0.0]], dtype=torch.complex128
            ),
            source=torch.tensor([[[0.0, 0.3, 0.2]]], dtype=torch.complex128),
            reference_projection=torch.tensor(
                [[[[0.0, delta_1.conjugate(), delta_2.conjugate()]]]],
                dtype=torch.complex128,
            ),
        )
        dataset = replace(
            dataset,
            primitive_count=3,
            reference_response=torch.tensor(
                [[[response]]], dtype=torch.complex128
            ),
            kpoints=(record,),
        )
        scale = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
        virtual = torch.stack(
            (
                torch.tensor(0.0, dtype=torch.complex128),
                torch.tensor(1.0, dtype=torch.complex128),
                scale.to(torch.complex128),
            )
        )
        transform = torch.stack(
            (torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128), virtual),
            dim=1,
        )

        result = evaluate_periodic_galerkin_response(dataset, transform)
        loss = result.relative_response_error ** 2
        loss.backward()

        self.assertIsNotNone(scale.grad)
        self.assertTrue(bool(torch.isfinite(scale.grad)))
        self.assertGreater(abs(float(scale.grad)), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
