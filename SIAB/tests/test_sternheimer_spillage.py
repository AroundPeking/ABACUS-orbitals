import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from sternheimer_data import PrimitiveBlock, SternheimerData
from sternheimer_spillage import (
    OrbitalColumn,
    SternheimerSpillage,
    assemble_orbital_coefficients,
    evaluate_spillage_for_columns,
    evaluate_spillage_for_columns_rank_revealing,
    radial_residual_spectrum,
    shell_count_for_capture,
)
from sternheimer_fixed_ao_data import AuxiliaryChannel
from sternheimer_primitive_galerkin_data import SternheimerPrimitiveGalerkinData


def make_sternheimer_data(
    blocks,
    q,
    norm=1.0,
    overlap=None,
    occupation=None,
    frequency_weight=None,
    frequency_ha=None,
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
    if frequency_ha is None:
        frequency_ha = torch.zeros(n_reference, dtype=torch.float64)
    else:
        frequency_ha = torch.as_tensor(frequency_ha, dtype=torch.float64)

    return SternheimerData(
        format_version=1,
        grid_volume_bohr3=1.0,
        blocks=tuple(blocks),
        occupied_state=torch.arange(n_reference, dtype=torch.int64),
        auxiliary_channel=torch.zeros(n_reference, dtype=torch.int64),
        frequency_ha=frequency_ha,
        occupation=occupation,
        frequency_weight=frequency_weight,
        norm=norm,
        q=q,
        overlap=overlap,
        provenance={
            "abacus_commit": "synthetic",
            "auxiliary_basis_sha256": "synthetic",
            "cell_bohr": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
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


def make_primitive_galerkin_assembly_data():
    blocks = tuple(
        PrimitiveBlock("H", 0, 1, m, 2, 2 * (m + 1))
        for m in (-1, 0, 1)
    )
    channel = AuxiliaryChannel(0, 0, 0, 0, 0, "H0_l0_n0_m0")
    overlap = torch.eye(6, dtype=torch.complex128)
    hamiltonian = torch.diag(
        torch.linspace(0.1, 0.6, 6, dtype=torch.float64).to(torch.complex128)
    ).reshape(1, 6, 6)
    perturbation = torch.zeros((1, 6, 6), dtype=torch.complex128)
    return SternheimerPrimitiveGalerkinData(
        format_version=1,
        representation="bessel_primitive_uniform_grid_gamma",
        energy_unit="Ha",
        blocks=blocks,
        channels=(channel,),
        occupation=torch.tensor([[1.0]], dtype=torch.float64),
        overlap=overlap,
        hamiltonian_ha=hamiltonian,
        perturbation_ha=perturbation,
        primitive_ao_overlap=torch.zeros((6, 1), dtype=torch.complex128),
        fixed_ao_grid_overlap=torch.eye(1, dtype=torch.complex128),
        fixed_ao_grid_hamiltonian_ha=torch.tensor(
            [[[-0.5]]], dtype=torch.complex128
        ),
        frequency_ha=torch.tensor([0.2], dtype=torch.float64),
        frequency_weight_ha=torch.tensor([1.0], dtype=torch.float64),
        provenance={
            "abacus_commit": "1" * 40,
            "auxiliary_basis_sha256": "a" * 64,
            "cell_bohr": [
                20.0,
                0.0,
                0.0,
                0.0,
                20.0,
                0.0,
                0.0,
                0.0,
                20.0,
            ],
            "ecut_ry": 100.0,
            "kernel": "full_coulomb",
            "orbital_sha256": "b" * 64,
            "pseudopotential_sha256": "c" * 64,
            "spin_convention": "occupation_in_metadata",
        },
        primitive_ao_hamiltonian_ha=torch.zeros(
            (1, 6, 1), dtype=torch.complex128
        ),
        primitive_ao_perturbation_ha=torch.zeros(
            (1, 6, 1), dtype=torch.complex128
        ),
    )


def make_s_and_d_spectrum_data(d_eigenvalues):
    d_eigenvalues = torch.as_tensor(d_eigenvalues, dtype=torch.float64)
    n_primitive = d_eigenvalues.numel()
    blocks = [h_s_block(1)]
    for magnetic_offset, m in enumerate(range(-2, 3)):
        blocks.append(
            PrimitiveBlock(
                "H",
                0,
                2,
                m,
                n_primitive,
                1 + magnetic_offset * n_primitive,
            )
        )

    q = torch.zeros(
        (5 * n_primitive, 1 + 5 * n_primitive),
        dtype=torch.complex128,
    )
    row = 0
    for block in blocks[1:]:
        for radial_index, eigenvalue in enumerate(d_eigenvalues):
            q[row, block.offset + radial_index] = torch.sqrt(
                eigenvalue / 5.0
            )
            row += 1
    return make_sternheimer_data(blocks, q, norm=torch.ones(q.shape[0]))


class SternheimerSpillageTest(unittest.TestCase):
    @staticmethod
    def _two_frequency_case():
        q = torch.tensor(
            [
                [0.2**0.5, 0.3**0.5, 0.0],
                [0.1**0.5, 0.4**0.5, 0.0],
                [1j * 0.4**0.5, 1j * 0.1**0.5, 0.0],
                [-1j * 0.2**0.5, 1j * 0.3**0.5, 0.0],
            ],
            dtype=torch.complex128,
        )
        data = make_sternheimer_data(
            [h_s_block(3)],
            q,
            norm=[1.0, 1.2, 1.0, 0.9],
            occupation=[1.0, 2.0, 3.0, 4.0],
            frequency_weight=[0.25, 0.75, 0.25, 0.75],
            frequency_ha=[0.1, 0.4, 0.1, 0.4],
        )
        coefficient = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        fixed = (OrbitalColumn("H", 0, 0, 0, 1),)
        return data, {"H": [coefficient]}, fixed

    def test_reports_frequency_local_losses(self):
        data, coefficient, fixed = self._two_frequency_case()

        result = SternheimerSpillage(data, coefficient, fixed).evaluate(
            coefficient
        )

        expected_frequency = torch.tensor([0.1, 0.4], dtype=torch.float64)
        expected_norm = torch.tensor([2.6, 5.0], dtype=torch.float64)
        expected_residual = torch.tensor([2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(result.frequency_ha, expected_frequency)
        torch.testing.assert_close(result.frequency_norm, expected_norm)
        torch.testing.assert_close(
            result.frequency_residual, expected_residual
        )
        torch.testing.assert_close(
            result.frequency_loss, expected_residual / expected_norm
        )
        torch.testing.assert_close(
            result.loss,
            torch.tensor(
                (0.25 * 2.0 + 0.75 * 3.0)
                / (0.25 * 2.6 + 0.75 * 5.0),
                dtype=torch.float64,
            ),
        )
        self.assertAlmostEqual(result.lowest_frequency_ha.item(), 0.1)
        self.assertAlmostEqual(
            result.lowest_frequency_loss.item(), 2.0 / 2.6
        )

    def test_frequency_local_losses_are_invariant_to_row_phase(self):
        data, coefficient, fixed = self._two_frequency_case()
        phase = torch.exp(torch.tensor(0.37j, dtype=torch.complex128))
        phased_q = data.q.clone()
        phased_q[data.frequency_ha == 0.1] *= phase
        phased = make_sternheimer_data(
            data.blocks,
            phased_q,
            norm=data.norm,
            overlap=data.overlap,
            occupation=data.occupation,
            frequency_weight=data.frequency_weight,
            frequency_ha=data.frequency_ha,
        )

        reference = SternheimerSpillage(data, coefficient, fixed).evaluate(
            coefficient
        )
        rotated = SternheimerSpillage(phased, coefficient, fixed).evaluate(
            coefficient
        )

        torch.testing.assert_close(
            rotated.frequency_residual,
            reference.frequency_residual,
            rtol=0.0,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            rotated.frequency_loss,
            reference.frequency_loss,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_lowest_frequency_loss_gradient_matches_finite_difference(self):
        data, coefficient, fixed = self._two_frequency_case()
        q = data.q.clone()
        q[:, 2] = torch.tensor(
            [0.1, 0.2, 0.15j, -0.1j], dtype=torch.complex128
        )
        data = make_sternheimer_data(
            data.blocks,
            q,
            norm=data.norm,
            overlap=data.overlap,
            occupation=data.occupation,
            frequency_weight=data.frequency_weight,
            frequency_ha=data.frequency_ha,
        )
        initial = coefficient["H"][0].clone()
        initial[2, 1] = 0.3
        variable = initial.clone().requires_grad_(True)
        evaluator = SternheimerSpillage(
            data, {"H": [initial]}, fixed
        )

        result = evaluator.evaluate({"H": [variable]})
        result.lowest_frequency_loss.backward()
        derivative = variable.grad[2, 1].item()

        epsilon = 1.0e-6
        plus = initial.clone()
        minus = initial.clone()
        plus[2, 1] += epsilon
        minus[2, 1] -= epsilon
        finite_difference = (
            evaluator.evaluate({"H": [plus]}).lowest_frequency_loss.item()
            - evaluator.evaluate({"H": [minus]}).lowest_frequency_loss.item()
        ) / (2.0 * epsilon)
        self.assertNotEqual(derivative, 0.0)
        self.assertAlmostEqual(
            derivative,
            finite_difference,
            delta=1.0e-8 + 1.0e-6 * abs(finite_difference),
        )

    def test_rank_revealing_projector_drops_duplicate_orbital_columns(self):
        data = make_sternheimer_data(
            (h_s_block(2),),
            torch.tensor([[1.0, 0.0]], dtype=torch.complex128),
            norm=torch.ones(1, dtype=torch.float64),
        )
        duplicated = {
            "H": [
                torch.tensor(
                    [[1.0, 1.0], [0.0, 0.0]], dtype=torch.float64
                )
            ]
        }

        with self.assertRaisesRegex(
            RuntimeError, "selected projector overlap is not positive definite"
        ):
            evaluate_spillage_for_columns(
                data, duplicated, include=lambda _: True
            )

        result = evaluate_spillage_for_columns_rank_revealing(
            data,
            duplicated,
            include=lambda _: True,
            condition_limit=1.0e12,
        )

        self.assertAlmostEqual(float(result.weighted_residual), 0.0)
        self.assertAlmostEqual(float(result.loss), 0.0)
        self.assertEqual(result.max_condition, 1.0)

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

    def test_rejects_missing_nonempty_coefficient_channel(self):
        data = make_sternheimer_data(
            [h_s_block(3)],
            [0.6, 0.64, 0.48],
        )
        c = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [[1.0], [0.0]],
                    dtype=torch.float64,
                ),
            ]
        }

        with self.assertRaisesRegex(
            ValueError,
            "Sternheimer data is missing primitive blocks for C channels: H/1",
        ):
            assemble_orbital_coefficients(data, c)

    def test_rejects_incomplete_p_m_group(self):
        data = make_sternheimer_data(
            [PrimitiveBlock("H", 0, 1, 0, 2, 0)],
            torch.zeros(2),
        )
        c = {
            "H": {
                1: torch.tensor(
                    [[1.0], [0.0]], dtype=torch.float64
                )
            }
        }

        with self.assertRaisesRegex(
            ValueError,
            r"incomplete PrimitiveBlock m group for H/atom0/l1: "
            r"expected \(-1, 0, 1\), got \(0,\)",
        ):
            assemble_orbital_coefficients(data, c)

    def test_rejects_missing_channel_group_for_actual_atom(self):
        blocks = (
            PrimitiveBlock("H", 0, 0, 0, 2, 0),
            PrimitiveBlock("H", 0, 1, -1, 2, 2),
            PrimitiveBlock("H", 0, 1, 0, 2, 4),
            PrimitiveBlock("H", 0, 1, 1, 2, 6),
            PrimitiveBlock("H", 1, 0, 0, 2, 8),
        )
        data = make_sternheimer_data(blocks, torch.zeros(10))
        c = {
            "H": [
                torch.tensor([[1.0], [0.0]], dtype=torch.float64),
                torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            ]
        }

        with self.assertRaisesRegex(
            ValueError,
            r"Sternheimer data is missing PrimitiveBlock group for "
            r"H/atom1/l1",
        ):
            assemble_orbital_coefficients(data, c)

    def test_complete_s_and_p_m_groups_assemble(self):
        blocks = (
            PrimitiveBlock("H", 0, 0, 0, 2, 0),
            PrimitiveBlock("H", 0, 1, -1, 2, 2),
            PrimitiveBlock("H", 0, 1, 0, 2, 4),
            PrimitiveBlock("H", 0, 1, 1, 2, 6),
        )
        data = make_sternheimer_data(blocks, torch.zeros(8))
        c = {
            "H": [
                torch.tensor([[1.0], [0.0]], dtype=torch.float64),
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
                ),
            ]
        }

        assembled, labels = assemble_orbital_coefficients(data, c)

        self.assertEqual(assembled.shape, (8, 7))
        self.assertEqual(len(labels), 7)

    def test_zero_zeta_channel_does_not_require_a_block_group(self):
        data = make_sternheimer_data([h_s_block(2)], torch.zeros(2))
        c = {
            "H": [
                torch.tensor([[1.0], [0.0]], dtype=torch.float64),
                torch.empty((2, 0), dtype=torch.float64),
            ]
        }

        assembled, labels = assemble_orbital_coefficients(data, c)

        self.assertEqual(assembled.shape, (2, 1))
        self.assertEqual(labels, (OrbitalColumn("H", 0, 0, 0, 1),))

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

    def test_primitive_galerkin_assembly_reuses_radial_coefficients(self):
        data = make_primitive_galerkin_assembly_data()
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

    def test_radial_residual_spectrum_recovers_shared_d_eigenvalues(self):
        data = make_s_and_d_spectrum_data([9.0, 4.0, 1.0])
        c = {
            "H": [
                torch.tensor([[1.0]], dtype=torch.float64),
                torch.empty((3, 0), dtype=torch.float64),
                torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64),
            ]
        }

        spectrum = radial_residual_spectrum(
            data,
            c,
            [self.fixed_h_1s],
            element="H",
            atom_index=0,
            l=2,
        )

        self.assertEqual(spectrum.magnetic_channels, (-2, -1, 0, 1, 2))
        self.assertEqual(spectrum.numerical_rank, 3)
        torch.testing.assert_close(
            spectrum.eigenvalues,
            torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-13,
        )
        torch.testing.assert_close(
            spectrum.cumulative_capture,
            torch.tensor(
                [9.0 / 14.0, 13.0 / 14.0, 1.0], dtype=torch.float64
            ),
            rtol=0.0,
            atol=1.0e-13,
        )
        self.assertEqual(shell_count_for_capture(spectrum, 0.90), 2)
        self.assertEqual(shell_count_for_capture(spectrum, 0.95), 3)
        torch.testing.assert_close(
            spectrum.coefficients.transpose(0, 1)
            @ spectrum.coefficients,
            torch.eye(3, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-13,
        )

    def test_radial_residual_spectrum_rejects_m_dependent_overlap(self):
        data = make_s_and_d_spectrum_data([2.0, 1.0])
        overlap = data.overlap.clone()
        first_d = data.blocks[1]
        first_slice = slice(
            first_d.offset, first_d.offset + first_d.n_primitive
        )
        overlap[first_slice, first_slice] *= 2.0
        data = make_sternheimer_data(
            data.blocks,
            data.q,
            norm=data.norm,
            overlap=overlap,
        )
        c = {
            "H": [
                torch.tensor([[1.0]], dtype=torch.float64),
                torch.empty((2, 0), dtype=torch.float64),
                torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            ]
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "magnetic-channel projected overlaps disagree",
        ):
            radial_residual_spectrum(
                data,
                c,
                [self.fixed_h_1s],
                element="H",
                atom_index=0,
                l=2,
                magnetic_overlap_tolerance=1.0e-12,
            )

    def test_radial_residual_spectrum_reports_accepted_grid_anisotropy(self):
        data = make_s_and_d_spectrum_data([2.0, 1.0])
        overlap = data.overlap.clone()
        first_d = data.blocks[1]
        first_slice = slice(
            first_d.offset, first_d.offset + first_d.n_primitive
        )
        overlap[first_slice, first_slice] *= 1.0 + 5.0e-5
        data = make_sternheimer_data(
            data.blocks,
            data.q,
            norm=data.norm,
            overlap=overlap,
        )
        c = {
            "H": [
                torch.tensor([[1.0]], dtype=torch.float64),
                torch.empty((2, 0), dtype=torch.float64),
                torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            ]
        }

        spectrum = radial_residual_spectrum(
            data,
            c,
            [self.fixed_h_1s],
            element="H",
            atom_index=0,
            l=2,
            magnetic_overlap_tolerance=1.0e-4,
        )

        self.assertGreater(spectrum.overlap_relative_deviation, 0.0)
        self.assertLess(spectrum.overlap_relative_deviation, 1.0e-4)
        self.assertEqual(shell_count_for_capture(spectrum, 0.95), 2)


if __name__ == "__main__":
    unittest.main()
