import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from sternheimer_data import PrimitiveBlock, SternheimerData
from sternheimer_spillage import (
    OrbitalColumn,
    SternheimerSpillage,
    assemble_orbital_coefficients,
)


def make_sternheimer_data(
    blocks,
    q,
    norm=1.0,
    overlap=None,
    occupation=None,
    frequency_weight=None,
):
    q = torch.as_tensor(q, dtype=torch.complex128)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    n_reference, n_primitive = q.shape
    if overlap is None:
        overlap = torch.eye(n_primitive, dtype=torch.complex128)
    else:
        overlap = torch.as_tensor(overlap, dtype=torch.complex128)
    norm = torch.as_tensor(norm, dtype=torch.float64)
    if norm.ndim == 0:
        norm = norm.repeat(n_reference)
    if occupation is None:
        occupation = torch.ones(n_reference, dtype=torch.float64)
    else:
        occupation = torch.as_tensor(occupation, dtype=torch.float64)
    if frequency_weight is None:
        frequency_weight = torch.ones(n_reference, dtype=torch.float64)
    else:
        frequency_weight = torch.as_tensor(
            frequency_weight, dtype=torch.float64
        )

    return SternheimerData(
        format_version=1,
        grid_volume_bohr3=1.0,
        blocks=tuple(blocks),
        occupied_state=torch.arange(n_reference, dtype=torch.int64),
        auxiliary_channel=torch.zeros(n_reference, dtype=torch.int64),
        frequency_ha=torch.zeros(n_reference, dtype=torch.float64),
        occupation=occupation,
        frequency_weight=frequency_weight,
        norm=norm,
        q=q,
        overlap=overlap,
        provenance={
            "abacus_commit": "synthetic",
            "auxiliary_basis_sha256": "synthetic",
            "cell_bohr": [1.0, 1.0, 1.0],
            "ecut_ry": 1.0,
            "kernel": "none",
            "orbital_sha256": "synthetic",
            "pseudopotential_sha256": "synthetic",
            "spin_convention": "unit_test",
        },
    )


def h_s_block(n_primitive):
    return PrimitiveBlock(
        element="H",
        atom_index=0,
        l=0,
        m=0,
        n_primitive=n_primitive,
        offset=0,
    )


class SternheimerSpillageTest(unittest.TestCase):
    fixed_h_1s = OrbitalColumn("H", 0, 0, 0, 1)

    def test_analytic_projected_result(self):
        data = make_sternheimer_data(
            [h_s_block(3)],
            [0.6, 0.64, 0.48],
        )
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        result = SternheimerSpillage(data, c, [self.fixed_h_1s]).evaluate(c)

        torch.testing.assert_close(
            result.weighted_norm,
            torch.tensor(0.64, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            result.weighted_residual,
            torch.tensor(0.2304, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            result.loss,
            torch.tensor(0.36, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_fixed_space_phase_invariance(self):
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }
        real_data = make_sternheimer_data(
            [h_s_block(3)], [0.6, 0.64, 0.48]
        )
        phase_data = make_sternheimer_data(
            [h_s_block(3)], [0.6j, 0.64, 0.48]
        )

        real_loss = SternheimerSpillage(
            real_data, c, [self.fixed_h_1s]
        ).evaluate(c).loss
        phase_loss = SternheimerSpillage(
            phase_data, c, [self.fixed_h_1s]
        ).evaluate(c).loss

        torch.testing.assert_close(real_loss, phase_loss, rtol=0.0, atol=1.0e-14)

    def test_autograd_matches_finite_difference(self):
        data = make_sternheimer_data(
            [h_s_block(3)],
            [0.3, 0.4, 0.5],
        )
        base = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.2]],
            dtype=torch.float64,
        )
        loss_function = SternheimerSpillage(
            data, {"H": [base]}, [self.fixed_h_1s]
        )
        variable = base.clone().requires_grad_(True)

        loss_function.evaluate({"H": [variable]}).loss.backward()
        derivative = variable.grad[2, 1].item()

        epsilon = 1.0e-6
        plus = base.clone()
        minus = base.clone()
        plus[2, 1] += epsilon
        minus[2, 1] -= epsilon
        finite_difference = (
            loss_function.evaluate({"H": [plus]}).loss.item()
            - loss_function.evaluate({"H": [minus]}).loss.item()
        ) / (2.0 * epsilon)

        self.assertNotEqual(derivative, 0.0)
        self.assertAlmostEqual(
            derivative,
            finite_difference,
            delta=1.0e-8 + 1.0e-6 * abs(finite_difference),
        )

    def test_dense_schur_result_matches_direct_combined_projector(self):
        generator = torch.Generator().manual_seed(314159)
        dense = torch.randn(
            (5, 5), generator=generator, dtype=torch.complex128
        )
        overlap = (
            dense.conj().transpose(0, 1) @ dense
            + 0.75 * torch.eye(5, dtype=torch.complex128)
        )
        q = torch.tensor(
            [
                [0.4 + 0.2j, -0.3 + 0.1j, 0.6 - 0.5j, 0.2 + 0.7j, -0.1j],
                [0.1 - 0.4j, 0.5 + 0.3j, -0.2 + 0.6j, 0.8 - 0.1j, 0.3j],
                [-0.2 + 0.5j, 0.7 - 0.2j, 0.1 + 0.4j, -0.6 + 0.3j, 0.9],
            ],
            dtype=torch.complex128,
        )
        coefficient = torch.tensor(
            [
                [1.0, 0.2, 0.4, 0.1],
                [0.1, 1.0, 0.3, -0.2],
                [0.3, 0.2, 1.0, 0.4],
                [0.0, 0.4, 0.2, 1.0],
                [0.2, -0.1, 0.3, 0.5],
            ],
            dtype=torch.float64,
        )
        a = coefficient.to(torch.complex128)
        a0 = a[:, :2]
        q0 = q @ a0
        qa = q @ a
        fixed_represented = torch.sum(
            q0
            * torch.linalg.solve(
                a0.conj().transpose(0, 1) @ overlap @ a0,
                q0.conj().transpose(0, 1),
            ).transpose(0, 1),
            dim=1,
        ).real
        combined_represented = torch.sum(
            qa
            * torch.linalg.solve(
                a.conj().transpose(0, 1) @ overlap @ a,
                qa.conj().transpose(0, 1),
            ).transpose(0, 1),
            dim=1,
        ).real
        norm = combined_represented + torch.tensor(
            [0.7, 1.1, 0.4], dtype=torch.float64
        )
        occupation = torch.tensor([1.0, 0.8, 1.5], dtype=torch.float64)
        frequency_weight = torch.tensor(
            [0.5, 2.0, 0.25], dtype=torch.float64
        )
        weight = occupation * frequency_weight
        expected_norm = torch.sum(weight * (norm - fixed_represented))
        expected_residual = torch.sum(weight * (norm - combined_represented))
        expected_loss = expected_residual / expected_norm
        data = make_sternheimer_data(
            [h_s_block(5)],
            q,
            norm=norm,
            overlap=overlap,
            occupation=occupation,
            frequency_weight=frequency_weight,
        )
        fixed = (
            OrbitalColumn("H", 0, 0, 0, 1),
            OrbitalColumn("H", 0, 0, 0, 2),
        )
        loss_function = SternheimerSpillage(data, {"H": [coefficient]}, fixed)

        s01 = a0.conj().transpose(0, 1) @ overlap @ a[:, 2:]
        self.assertGreater(torch.linalg.norm(s01).item(), 0.0)
        result = loss_function.evaluate({"H": [coefficient]})
        torch.testing.assert_close(
            result.weighted_norm, expected_norm, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            result.weighted_residual,
            expected_residual,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            result.loss, expected_loss, rtol=1.0e-12, atol=1.0e-12
        )

        variable = coefficient.clone().requires_grad_(True)
        loss_function.evaluate({"H": [variable]}).loss.backward()
        derivative = variable.grad[4, 2].item()
        epsilon = 1.0e-6
        plus = coefficient.clone()
        minus = coefficient.clone()
        plus[4, 2] += epsilon
        minus[4, 2] -= epsilon
        finite_difference = (
            loss_function.evaluate({"H": [plus]}).loss.item()
            - loss_function.evaluate({"H": [minus]}).loss.item()
        ) / (2.0 * epsilon)
        self.assertNotEqual(derivative, 0.0)
        self.assertAlmostEqual(
            derivative,
            finite_difference,
            delta=1.0e-8 + 1.0e-6 * abs(finite_difference),
        )

    def test_row_local_tolerance_rejects_negative_residual(self):
        data = make_sternheimer_data(
            [h_s_block(3)],
            [[0.0, 0.0, 0.0], [0.0, 2.0**0.5, 0.0]],
            norm=[1.0e12, 1.0],
        )
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        with self.assertRaisesRegex(
            RuntimeError, "materially negative projected residual"
        ):
            SternheimerSpillage(data, c, [self.fixed_h_1s]).evaluate(c)

    def test_singular_variable_overlap_names_a_variable_column(self):
        data = make_sternheimer_data([h_s_block(2)], [0.3, 0.4])
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
                    dtype=torch.float64,
                )
            ]
        }

        with self.assertRaisesRegex(
            RuntimeError,
            r"variable overlap is not positive definite.*H/0/0/0/zeta2",
        ):
            SternheimerSpillage(data, c, [self.fixed_h_1s]).evaluate(c)

    def test_repeated_m_assembly_reuses_radial_coefficients(self):
        blocks = tuple(
            PrimitiveBlock("H", 0, 1, m, 2, 2 * (m + 1))
            for m in (-1, 0, 1)
        )
        data = make_sternheimer_data(blocks, torch.zeros(6))
        radial = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64
        )

        assembled, labels = assemble_orbital_coefficients(
            data, {"H": {1: radial}}
        )

        expected = torch.zeros((6, 6), dtype=torch.complex128)
        for block_index in range(3):
            row = 2 * block_index
            column = 2 * block_index
            expected[row : row + 2, column : column + 2] = radial
        self.assertEqual(assembled.shape, (6, 6))
        torch.testing.assert_close(assembled, expected)
        self.assertEqual(
            labels,
            tuple(
                OrbitalColumn("H", 0, 1, m, zeta)
                for m in (-1, 0, 1)
                for zeta in (1, 2)
            ),
        )

    def test_rejects_absent_fixed_label(self):
        data = make_sternheimer_data([h_s_block(2)], [0.3, 0.4])
        c = {
            "H": [torch.tensor([[1.0], [0.0]], dtype=torch.float64)]
        }
        absent = OrbitalColumn("H", 0, 0, 0, 2)

        with self.assertRaisesRegex(ValueError, "fixed orbital.*not found"):
            SternheimerSpillage(data, c, [absent])

    def test_rejects_ill_conditioned_variable_overlap(self):
        data = make_sternheimer_data([h_s_block(3)], [0.2, 0.3, 0.4])
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0e-4]],
                    dtype=torch.float64,
                )
            ]
        }
        loss_function = SternheimerSpillage(
            data,
            c,
            [self.fixed_h_1s],
            condition_limit=1.0e6,
        )

        with self.assertRaisesRegex(RuntimeError, "condition number"):
            loss_function.evaluate(c)

    def test_rejects_condition_limit_below_one(self):
        data = make_sternheimer_data([h_s_block(2)], [0.2, 0.3])
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
                )
            ]
        }

        with self.assertRaisesRegex(ValueError, "condition_limit.*at least 1"):
            SternheimerSpillage(
                data, c, [self.fixed_h_1s], condition_limit=0.5
            )

    def test_rejects_non_finite_coefficient(self):
        data = make_sternheimer_data([h_s_block(2)], [0.2, 0.3])
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, float("nan")]],
                    dtype=torch.float64,
                )
            ]
        }

        with self.assertRaisesRegex(
            ValueError, r"C\['H'\]\[0\] must contain only finite values"
        ):
            assemble_orbital_coefficients(data, c)

    def test_rejects_duplicate_primitive_block_keys(self):
        blocks = (
            PrimitiveBlock("H", 0, 0, 0, 1, 0),
            PrimitiveBlock("H", 0, 0, 0, 1, 1),
        )
        data = make_sternheimer_data(blocks, [0.2, 0.3])
        c = {"H": [torch.tensor([[1.0]], dtype=torch.float64)]}

        with self.assertRaisesRegex(ValueError, "duplicate PrimitiveBlock key"):
            assemble_orbital_coefficients(data, c)


if __name__ == "__main__":
    unittest.main()
