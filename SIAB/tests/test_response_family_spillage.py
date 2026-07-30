import math
import unittest

import torch

import common  # noqa: F401 - configures the optimizer import path
from response_family_spillage import NormalizedPhysicalFamilySpillage
from response_selection import ResponseTargetFamily
from sternheimer_data import PrimitiveBlock
from test_sternheimer_spillage import make_sternheimer_data


def target_data():
    return make_sternheimer_data(
        [PrimitiveBlock("H", 0, 0, 0, 3, 0)],
        [math.sqrt(2.0), math.sqrt(6.0), 0.0],
        norm=10.0,
    )


def two_frequency_target_data():
    return make_sternheimer_data(
        [PrimitiveBlock("H", 0, 0, 0, 3, 0)],
        [
            [math.sqrt(2.0), math.sqrt(6.0), 0.0],
            [1.0, math.sqrt(3.0), 0.0],
        ],
        norm=[10.0, 8.0],
        frequency_ha=[0.1, 0.4],
        frequency_weight=[0.2, 0.8],
    )


def fixed_dzp():
    return {
        "H": [
            torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64),
        ]
    }


def current_basis():
    return {
        "H": [
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                dtype=torch.float64,
            ),
        ]
    }


class NormalizedPhysicalFamilySpillageTest(unittest.TestCase):
    fixed_specs = ({"element": "H", "l": 0, "zeta": 1},)

    def test_sums_each_family_after_its_own_fixed_dzp_normalization(self):
        data = target_data()
        families = (
            ResponseTargetFamily("atom", (data,), "physical"),
            ResponseTargetFamily("multicenter", (data,), "physical"),
        )
        evaluator = NormalizedPhysicalFamilySpillage(
            families,
            current_basis(),
            fixed_dzp(),
            self.fixed_specs,
        )

        result = evaluator.evaluate(current_basis())

        torch.testing.assert_close(
            result.loss,
            torch.tensor(0.5, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-13,
        )
        self.assertTrue(math.isfinite(result.max_condition))

    def test_rejects_ghost_family_from_optimizer_loss(self):
        family = ResponseTargetFamily(
            "fragment_ghost",
            (target_data(),),
            "ghost",
            real_atom_index=0,
        )

        with self.assertRaisesRegex(ValueError, "physical target families"):
            NormalizedPhysicalFamilySpillage(
                (family,),
                current_basis(),
                fixed_dzp(),
                self.fixed_specs,
            )

    def test_averages_frequency_local_loss_across_physical_families(self):
        data = two_frequency_target_data()
        families = (
            ResponseTargetFamily("atom", (data,), "physical"),
            ResponseTargetFamily("multicenter", (data,), "physical"),
        )
        evaluator = NormalizedPhysicalFamilySpillage(
            families,
            current_basis(),
            fixed_dzp(),
            self.fixed_specs,
        )

        result = evaluator.evaluate(current_basis())

        torch.testing.assert_close(
            result.frequency_ha,
            torch.tensor([0.1, 0.4], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.frequency_loss,
            torch.tensor([0.25, 4.0 / 7.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.frequency_residual, result.frequency_loss
        )
        torch.testing.assert_close(
            result.frequency_norm, torch.ones(2, dtype=torch.float64)
        )


if __name__ == "__main__":
    unittest.main()
