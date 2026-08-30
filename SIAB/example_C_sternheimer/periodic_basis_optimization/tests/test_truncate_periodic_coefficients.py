import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "truncate_periodic_coefficients.py"
SPEC = importlib.util.spec_from_file_location("truncate_periodic_coefficients", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TruncatePeriodicCoefficientsTest(unittest.TestCase):
    def test_removes_complete_high_angular_channels(self):
        channels = [
            torch.full((3, count), float(l + 1), dtype=torch.float64)
            for l, count in enumerate((3, 3, 2, 1, 1))
        ]

        truncated = MODULE.truncate_angular_channels(
            {"C": channels},
            target_lmax=2,
            element="C",
        )

        self.assertEqual(len(truncated["C"]), 3)
        for l in range(3):
            self.assertTrue(torch.equal(truncated["C"][l], channels[l]))
            self.assertNotEqual(truncated["C"][l].data_ptr(), channels[l].data_ptr())

    def test_rejects_non_reducing_target_lmax(self):
        coefficients = {"C": [torch.ones((3, 1), dtype=torch.float64)]}

        with self.assertRaisesRegex(ValueError, "smaller"):
            MODULE.truncate_angular_channels(
                coefficients,
                target_lmax=0,
                element="C",
            )

    def test_can_preserve_empty_high_l_layout_for_primitive_blocks(self):
        channels = [
            torch.full((3, count), float(l + 1), dtype=torch.float64)
            for l, count in enumerate((3, 3, 2, 1, 1))
        ]

        truncated = MODULE.truncate_angular_channels(
            {"C": channels},
            target_lmax=2,
            element="C",
            preserve_channel_layout=True,
        )

        self.assertEqual([channel.shape[1] for channel in truncated["C"]], [3, 3, 2, 0, 0])
        self.assertEqual(truncated["C"][3].shape, (3, 0))
        self.assertEqual(truncated["C"][4].shape, (3, 0))


if __name__ == "__main__":
    unittest.main()
