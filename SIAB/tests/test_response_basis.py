import unittest

import torch

from response_basis import canonicalize_columns, replace_channel_coefficients


class ResponseBasisTest(unittest.TestCase):
    def test_canonicalize_columns_makes_largest_entry_positive(self):
        coefficients = torch.tensor(
            [
                [-1.0, 0.1, 0.0],
                [0.2, -0.5, 0.0],
                [0.1, 0.2, -0.8],
            ],
            dtype=torch.float64,
        )

        actual = canonicalize_columns(coefficients)

        torch.testing.assert_close(
            actual,
            torch.tensor(
                [
                    [1.0, -0.1, 0.0],
                    [-0.2, 0.5, 0.0],
                    [-0.1, -0.2, 0.8],
                ],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            actual.transpose(0, 1) @ actual,
            coefficients.transpose(0, 1) @ coefficients,
        )

    def test_replace_channel_coefficients_preserves_other_channels(self):
        coefficients = {
            "H": [
                torch.arange(8, dtype=torch.float64).reshape(4, 2),
                torch.arange(4, dtype=torch.float64).reshape(4, 1),
                torch.zeros((4, 2), dtype=torch.float64),
            ]
        }
        before_s = coefficients["H"][0].clone()
        before_p = coefficients["H"][1].clone()
        d = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.2, 0.3], [0.4, 0.5]],
            dtype=torch.float64,
        )

        replace_channel_coefficients(coefficients, "H", 2, d)

        self.assertTrue(torch.equal(coefficients["H"][0], before_s))
        self.assertTrue(torch.equal(coefficients["H"][1], before_p))
        self.assertTrue(torch.equal(coefficients["H"][2], d))
        self.assertTrue(coefficients["H"][2].requires_grad)

    def test_replace_channel_coefficients_rejects_shape_mismatch(self):
        coefficients = {
            "H": [torch.zeros((4, 1), dtype=torch.float64)]
        }
        with self.assertRaisesRegex(ValueError, "shape"):
            replace_channel_coefficients(
                coefficients,
                "H",
                0,
                torch.zeros((4, 2), dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
