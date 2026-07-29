import math
import unittest
from types import SimpleNamespace

import torch

import common  # noqa: F401 - configures the optimizer import path
from radial_locality import RadialSubspaceLocality


def locality_evaluator(condition_limit=1.0e10):
    info_element = {"H": SimpleNamespace(Nl=1, Ne=3)}
    radial = {
        "Rcut": {"H": 8.0},
        "dr": {"H": 0.02},
        "smearing_sigma": {"H": 0.1},
    }
    eigenvalues = {
        "H": torch.tensor(
            [[math.pi / 8.0, 2.0 * math.pi / 8.0, 3.0 * math.pi / 8.0]],
            dtype=torch.float64,
        )
    }
    return RadialSubspaceLocality(
        info_element,
        radial,
        eigenvalues,
        ({"element": "H", "l": 0, "zeta": 1},),
        local_radius=4.0,
        condition_limit=condition_limit,
    )


def independent_coefficients():
    return {
        "H": [
            torch.eye(3, dtype=torch.float64, requires_grad=True),
        ]
    }


class RadialSubspaceLocalityTest(unittest.TestCase):
    def test_tail_fraction_is_finite_bounded_and_differentiable(self):
        evaluator = locality_evaluator()
        coefficients = independent_coefficients()

        result = evaluator.evaluate(coefficients)

        self.assertGreaterEqual(result.loss.item(), 0.0)
        self.assertLessEqual(result.loss.item(), 1.0)
        channel = result.by_channel[("H", 0)]
        self.assertEqual(channel.variable_columns, 2)
        self.assertGreaterEqual(channel.tail_fraction.item(), 0.0)
        self.assertLessEqual(channel.tail_fraction.item(), 1.0)
        self.assertTrue(math.isfinite(result.max_condition))

        result.loss.backward()
        gradient = coefficients["H"][0].grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.all(torch.isfinite(gradient)))
        self.assertGreater(torch.linalg.norm(gradient[:, 1:]).item(), 0.0)

    def test_variable_subspace_metric_is_rotation_and_scale_invariant(self):
        evaluator = locality_evaluator()
        coefficients = independent_coefficients()
        transformed = {
            "H": [coefficients["H"][0].detach().clone()]
        }
        rotation = torch.tensor(
            [[2.0, 0.5], [-0.25, 1.5]], dtype=torch.float64
        )
        transformed["H"][0][:, 1:] = (
            transformed["H"][0][:, 1:] @ rotation
        )
        transformed["H"][0].requires_grad_(True)

        reference = evaluator.evaluate(coefficients)
        changed = evaluator.evaluate(transformed)

        torch.testing.assert_close(
            changed.loss,
            reference.loss,
            rtol=1.0e-11,
            atol=1.0e-12,
        )

    def test_fixed_only_basis_has_zero_locality_loss(self):
        evaluator = locality_evaluator()
        coefficients = {
            "H": [
                torch.tensor(
                    [[1.0], [0.0], [0.0]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
            ]
        }

        result = evaluator.evaluate(coefficients)

        self.assertEqual(result.loss.item(), 0.0)
        self.assertEqual(result.by_channel[("H", 0)].variable_columns, 0)

    def test_dependent_projected_variable_space_is_rejected(self):
        evaluator = locality_evaluator()
        coefficients = independent_coefficients()
        with torch.no_grad():
            coefficients["H"][0][:, 2].copy_(coefficients["H"][0][:, 1])

        with self.assertRaisesRegex(
            RuntimeError, "projected variable radial overlap"
        ):
            evaluator.evaluate(coefficients)

    def test_invalid_radius_and_fixed_spec_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "local_radius"):
            RadialSubspaceLocality(
                {"H": SimpleNamespace(Nl=1, Ne=3)},
                {
                    "Rcut": {"H": 8.0},
                    "dr": {"H": 0.02},
                    "smearing_sigma": {"H": 0.1},
                },
                {"H": torch.ones((1, 3), dtype=torch.float64)},
                ({"element": "H", "l": 0, "zeta": 1},),
                local_radius=8.0,
            )

        with self.assertRaisesRegex(ValueError, "fixed radial spec"):
            RadialSubspaceLocality(
                {"H": SimpleNamespace(Nl=1, Ne=3)},
                {
                    "Rcut": {"H": 8.0},
                    "dr": {"H": 0.02},
                    "smearing_sigma": {"H": 0.1},
                },
                {"H": torch.ones((1, 3), dtype=torch.float64)},
                ({"element": "H", "l": 0},),
                local_radius=4.0,
            )


if __name__ == "__main__":
    unittest.main()
