"""Independent finite-space SOS oracle; not a production LRI comparison."""

import math
import unittest
from dataclasses import replace

import torch

import common  # noqa: F401 - configures the optimizer import path
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock
from periodic_galerkin_sternheimer import evaluate_periodic_galerkin_response
import test_periodic_galerkin_sternheimer as fixtures


def adjoint(value):
    return value.transpose(-2, -1).conj()


def spectral_sum(dataset, transform):
    """Sum every eigenstate of Q H Q, keeping the supplied occupied reference."""
    half = torch.zeros_like(dataset.reference_response)
    for record in dataset.kpoints:
        metric = adjoint(transform) @ record.overlap @ transform
        lower = torch.linalg.cholesky(metric)
        frame = transform @ torch.linalg.inv(adjoint(lower))
        occupied = record.occupied_projection @ frame
        _, _, vh = torch.linalg.svd(occupied, full_matrices=True)
        virtual = frame @ adjoint(vh)[:, record.occupation.numel():]
        energies, vectors = torch.linalg.eigh(
            adjoint(virtual) @ record.hamiltonian_ha @ virtual
        )
        virtual = virtual @ vectors
        for band, occupation in enumerate(record.occupation):
            transition = record.source[band] @ virtual
            for iw, omega in enumerate(dataset.frequency_ha):
                denominator = energies - record.source_eigenvalue_ha[band] + 1j * omega
                half[iw] -= record.k_weight * occupation * (
                    (transition / denominator[None, :]) @ adjoint(transition)
                )
    return half + adjoint(half)


def correlation_integral(dataset, response):
    eigenvalues = torch.linalg.eigvalsh(response)
    if bool(torch.any(eigenvalues >= 1.0)):
        raise ValueError("trace-log argument is not positive")
    raw = (torch.log1p(-eigenvalues) + eigenvalues).sum(dim=-1)
    return dataset.q_weight * torch.dot(dataset.frequency_weights_ha, raw) / (2 * math.pi)


class PeriodicGalerkinSosEquivalenceTest(unittest.TestCase):
    def fixture(self):
        dataset, _, _ = fixtures.PeriodicGalerkinSternheimerTest().complete_two_level_dataset()
        generator = torch.Generator().manual_seed(314159)

        def random_complex(shape):
            return torch.complex(
                torch.randn(shape, generator=generator, dtype=torch.float64),
                torch.randn(shape, generator=generator, dtype=torch.float64),
            )

        coordinates = torch.eye(5, dtype=torch.complex128) + 0.12 * random_complex((5, 5))
        frequencies = torch.logspace(-2, 1, 12, dtype=torch.float64)
        records = []
        for ik, weight in enumerate((0.7, 1.3)):
            matrix = 0.08 * random_complex((5, 5))
            hamiltonian = 0.5 * (matrix + adjoint(matrix)) + torch.diag(
                torch.tensor([-0.8, -0.4, 0.7, 1.2, 2.1], dtype=torch.complex128)
            )
            source = 0.15 * random_complex((2, 3, 5))
            records.append(replace(
                dataset.kpoints[0],
                source_ik=ik + 1,
                target_ik=ik + 1,
                k_weight=weight,
                occupation=torch.tensor([1.0, 0.8], dtype=torch.float64),
                source_eigenvalue_ha=torch.tensor([-0.8, -0.4], dtype=torch.float64),
                overlap=adjoint(coordinates) @ coordinates,
                hamiltonian_ha=adjoint(coordinates) @ hamiltonian @ coordinates,
                occupied_projection=coordinates[:2],
                source=source @ coordinates,
                reference_projection=torch.zeros((12, 2, 3, 5), dtype=torch.complex128),
            ))
        dataset = replace(
            dataset,
            q_count=64,
            q_weight=8.0 / 64,
            primitive_count=5,
            raw_auxiliary_dimension=3,
            whitened_auxiliary_rank=3,
            frequency_ha=frequencies,
            frequency_weights_ha=torch.linspace(0.01, 1.0, 12, dtype=torch.float64),
            coulomb_metric=torch.eye(3, dtype=torch.complex128),
            coulomb_whitening=torch.eye(3, dtype=torch.complex128),
            reference_response=-torch.eye(3, dtype=torch.complex128).repeat(12, 1, 1),
            primitive_blocks=(PeriodicGalerkinPrimitiveBlock("C", 0, 0, 0, 5, 0),),
            kpoints=tuple(records),
        )
        return dataset, coordinates

    def reduced_transform(self, coordinates, angle):
        identity = torch.eye(5, dtype=torch.complex128)
        virtual = torch.cos(angle) * identity[:, 2] + torch.sin(angle) * identity[:, 3]
        frame = torch.stack((identity[:, 0], identity[:, 1], virtual, identity[:, 4]), dim=1)
        return torch.linalg.solve(coordinates, frame)

    def test_full_and_reduced_complex_spaces_match_all_band_spectral_sum(self):
        dataset, coordinates = self.fixture()
        transforms = (
            torch.eye(5, dtype=torch.complex128),
            self.reduced_transform(coordinates, torch.tensor(0.3, dtype=torch.float64)),
        )
        for transform in transforms:
            with self.subTest(candidate_size=transform.shape[1]):
                dense = evaluate_periodic_galerkin_response(dataset, transform)
                expected = spectral_sum(dataset, transform)
                torch.testing.assert_close(dense.response, expected, rtol=2e-12, atol=2e-13)
                torch.testing.assert_close(
                    correlation_integral(dataset, dense.response),
                    correlation_integral(dataset, expected),
                    rtol=2e-12, atol=2e-13,
                )

    def test_nonorthogonal_candidate_coordinates_preserve_response(self):
        dataset, coordinates = self.fixture()
        transform = self.reduced_transform(coordinates, torch.tensor(0.3, dtype=torch.float64))
        mixing = torch.tensor([
            [1.0, 0.1j, 0.2, 0.0], [0.1, 1.3, 0.0, -0.1j],
            [0.0, 0.2, 0.8, 0.1j], [0.1j, 0.0, 0.2, 1.1],
        ], dtype=torch.complex128)
        expected = spectral_sum(dataset, transform)
        actual = evaluate_periodic_galerkin_response(dataset, transform @ mixing)
        torch.testing.assert_close(actual.response, expected, rtol=2e-12, atol=2e-13)

    def test_trace_log_gradient_at_repeated_overlap_eigenvalues(self):
        dataset, coordinates = self.fixture()
        angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        transform = self.reduced_transform(coordinates, angle)
        response = evaluate_periodic_galerkin_response(dataset, transform).response
        loss = correlation_integral(dataset, response)
        loss.backward()
        step = 1e-5
        values = []
        for offset in (-step, step):
            moved = self.reduced_transform(coordinates, angle.detach() + offset)
            values.append(correlation_integral(dataset, spectral_sum(dataset, moved)))
        finite_difference = (values[1] - values[0]) / (2 * step)
        self.assertGreater(abs(float(finite_difference)), 1e-8)
        torch.testing.assert_close(angle.grad, finite_difference, rtol=2e-6, atol=1e-10)

    def test_rediagonalizing_full_h_changes_the_fixed_occupied_problem(self):
        dataset, coordinates = self.fixture()
        transform = torch.eye(5, dtype=torch.complex128)
        fixed = spectral_sum(dataset, transform)
        records = []
        inverse = torch.linalg.inv(coordinates)
        for record in dataset.kpoints:
            energies, vectors = torch.linalg.eigh(
                adjoint(inverse) @ record.hamiltonian_ha @ inverse
            )
            occupied = inverse @ vectors[:, :2]
            records.append(replace(
                record,
                occupied_projection=adjoint(occupied) @ record.overlap,
                source_eigenvalue_ha=energies[:2],
            ))
        changed = spectral_sum(replace(dataset, kpoints=tuple(records)), transform)
        self.assertGreater(float(torch.linalg.norm(changed - fixed)), 1e-3)


if __name__ == "__main__":
    unittest.main()
