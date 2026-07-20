import math
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from response_selection import (
    ResponseTargetFamily,
    borrowing_gap,
    normalized_family_loss,
)
from sternheimer_data import PrimitiveBlock
from sternheimer_spillage import evaluate_spillage_for_columns
from test_sternheimer_spillage import make_sternheimer_data


def one_center_family_data():
    return make_sternheimer_data(
        [PrimitiveBlock("H", 0, 0, 0, 3, 0)],
        [math.sqrt(2.0), math.sqrt(6.0), 0.0],
        norm=10.0,
    )


def fragment_ghost_data():
    blocks = (
        PrimitiveBlock("H", 0, 0, 0, 2, 0),
        PrimitiveBlock("H", 1, 0, 0, 2, 2),
    )
    return make_sternheimer_data(
        blocks,
        [0.0, math.sqrt(0.4), 0.0, math.sqrt(0.4)],
        norm=1.0,
    )


def fixed_dzp_coefficients():
    return {
        "H": [
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        ]
    }


def expanded_coefficients():
    return {"H": [torch.eye(2, dtype=torch.float64)]}


class ExplicitProjectorTest(unittest.TestCase):
    def test_selected_columns_use_the_full_overlap_metric(self):
        data = fragment_ghost_data()
        coefficients = expanded_coefficients()

        own = evaluate_spillage_for_columns(
            data,
            coefficients,
            include=lambda label: label.atom_index == 0,
        )
        all_centers = evaluate_spillage_for_columns(
            data,
            coefficients,
            include=lambda label: True,
        )

        torch.testing.assert_close(
            own.weighted_residual,
            torch.tensor(0.6, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-13,
        )
        torch.testing.assert_close(
            all_centers.weighted_residual,
            torch.tensor(0.2, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-13,
        )

    def test_rejects_an_empty_selected_projector(self):
        with self.assertRaisesRegex(
            ValueError, "selected projector contains no orbital columns"
        ):
            evaluate_spillage_for_columns(
                fragment_ghost_data(),
                fixed_dzp_coefficients(),
                include=lambda label: False,
            )


class ResponseTargetFamilyTest(unittest.TestCase):
    def test_family_loss_is_normalized_to_fixed_dzp(self):
        data = one_center_family_data()
        family = ResponseTargetFamily("atom", (data,), "physical")
        fixed_dzp = {
            "H": [
                torch.tensor(
                    [[1.0], [0.0], [0.0]], dtype=torch.float64
                )
            ]
        }
        current = {
            "H": [
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                )
            ]
        }

        self.assertAlmostEqual(
            normalized_family_loss(family, current, fixed_dzp),
            0.25,
            places=13,
        )

    def test_borrowing_gap_is_own_minus_all_with_one_dzp_scale(self):
        family = ResponseTargetFamily(
            "fragment_ghost",
            (fragment_ghost_data(),),
            "ghost",
            real_atom_index=0,
        )

        value = borrowing_gap(
            family,
            expanded_coefficients(),
            fixed_dzp_coefficients(),
        )

        self.assertAlmostEqual(value, 0.4, places=13)

    def test_family_contract_rejects_wrong_roles_and_missing_real_atom(self):
        data = fragment_ghost_data()
        with self.assertRaisesRegex(ValueError, "role"):
            ResponseTargetFamily("bad", (data,), "energy")
        with self.assertRaisesRegex(ValueError, "real_atom_index"):
            ResponseTargetFamily("ghost", (data,), "ghost")


if __name__ == "__main__":
    unittest.main()
