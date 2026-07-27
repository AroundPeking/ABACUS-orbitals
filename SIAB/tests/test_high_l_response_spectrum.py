import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from sternheimer_data import PrimitiveBlock
from sternheimer_spillage import radial_residual_spectrum_many
from test_sternheimer_spillage import make_sternheimer_data


def make_high_l_target(l, eigenvalues, atom_count=1, omitted_m=None):
    eigenvalues = torch.as_tensor(eigenvalues, dtype=torch.float64)
    n_radial = eigenvalues.numel()
    blocks = [PrimitiveBlock("H", atom, 0, 0, 1, atom) for atom in range(atom_count)]
    offset = atom_count
    for atom in range(atom_count):
        for m in range(-l, l + 1):
            if m == omitted_m:
                continue
            blocks.append(
                PrimitiveBlock("H", atom, l, m, n_radial, offset)
            )
            offset += n_radial

    n_reference = atom_count * (2 * l + 1) * n_radial
    q = torch.zeros((n_reference, offset), dtype=torch.complex128)
    row = 0
    for atom in range(atom_count):
        atom_blocks = [
            block
            for block in blocks
            if block.atom_index == atom and block.l == l
        ]
        for block in atom_blocks:
            for radial_index, eigenvalue in enumerate(eigenvalues):
                q[row, block.offset + radial_index] = torch.sqrt(
                    eigenvalue / (2 * l + 1)
                )
                row += 1
    return make_sternheimer_data(blocks, q, norm=torch.ones(n_reference))


def coefficients(l, n_radial):
    by_l = [torch.tensor([[1.0]], dtype=torch.float64)]
    for channel in range(1, l + 1):
        rows = n_radial if channel == l else 1
        by_l.append(torch.empty((rows, 0), dtype=torch.float64))
    return {"H": by_l}


def fixed_dzp_specs():
    return ({"element": "H", "l": 0, "zeta": 1},)


class HighAngularResponseSpectrumTest(unittest.TestCase):
    def test_many_target_f_spectrum_sums_weighted_covariances(self):
        first = make_high_l_target(3, [4.0, 1.0])
        second = make_high_l_target(3, [2.0, 3.0])

        spectrum = radial_residual_spectrum_many(
            [first, second],
            coefficients(3, 2),
            fixed_dzp_specs(),
            element="H",
            l=3,
        )

        self.assertEqual(spectrum.magnetic_channels, tuple(range(-3, 4)))
        torch.testing.assert_close(
            spectrum.eigenvalues,
            torch.tensor([6.0, 4.0], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_many_target_spectrum_rejects_distinct_projected_metrics(self):
        first = make_high_l_target(3, [4.0, 1.0])
        second = make_high_l_target(3, [2.0, 3.0])
        second_overlap = second.overlap.clone()
        for block in second.blocks:
            if block.l != 3:
                continue
            block_slice = slice(block.offset, block.offset + block.n_primitive)
            second_overlap[block_slice, block_slice] *= 2.0
        second = make_sternheimer_data(
            second.blocks,
            second.q,
            norm=second.norm,
            overlap=second_overlap,
        )

        with self.assertRaisesRegex(
            RuntimeError, "target/atom projected overlaps disagree"
        ):
            radial_residual_spectrum_many(
                [first, second],
                coefficients(3, 2),
                fixed_dzp_specs(),
                element="H",
                l=3,
            )

    def test_many_target_spectrum_accumulates_every_atom(self):
        target = make_high_l_target(3, [2.5], atom_count=2)

        spectrum = radial_residual_spectrum_many(
            [target],
            coefficients(3, 1),
            fixed_dzp_specs(),
            element="H",
            l=3,
        )

        torch.testing.assert_close(
            spectrum.eigenvalues,
            torch.tensor([5.0], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_rejects_incomplete_g_multiplet(self):
        data = make_high_l_target(4, [2.0], omitted_m=4)

        with self.assertRaisesRegex(ValueError, "expected .* got"):
            radial_residual_spectrum_many(
                [data],
                coefficients(4, 1),
                fixed_dzp_specs(),
                element="H",
                l=4,
            )

    def test_rejects_cross_target_radial_count_mismatch(self):
        first = make_high_l_target(3, [2.0, 1.0])
        second = make_high_l_target(3, [2.0, 1.0, 0.5])

        with self.assertRaisesRegex(ValueError, "radial row count"):
            radial_residual_spectrum_many(
                [first, second],
                coefficients(3, 2),
                fixed_dzp_specs(),
                element="H",
                l=3,
            )


if __name__ == "__main__":
    unittest.main()
