import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "select_periodic_coefficients.py"
SPEC = importlib.util.spec_from_file_location("select_periodic_coefficients", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelectPeriodicCoefficientsTest(unittest.TestCase):
    def test_selects_one_radial_column_without_changing_other_channels(self):
        s_channel = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        )
        g_channel = torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]],
            dtype=torch.float64,
        )
        coefficients = {"C": [s_channel, g_channel]}

        first = MODULE.select_radial_columns(
            coefficients,
            angular_channel=1,
            zeta_indices=(1,),
            element="C",
        )
        second = MODULE.select_radial_columns(
            coefficients,
            angular_channel=1,
            zeta_indices=(2,),
            element="C",
        )

        self.assertTrue(torch.equal(first["C"][0], s_channel))
        self.assertTrue(torch.equal(second["C"][0], s_channel))
        torch.testing.assert_close(
            first["C"][1],
            g_channel[:, :1],
            check_stride=False,
        )
        torch.testing.assert_close(
            second["C"][1],
            g_channel[:, 1:2],
            check_stride=False,
        )
        self.assertNotEqual(first["C"][0].data_ptr(), s_channel.data_ptr())

    def test_rejects_duplicate_or_out_of_range_zeta_indices(self):
        coefficients = {
            "C": [torch.eye(3, 2, dtype=torch.float64)],
        }

        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.select_radial_columns(
                coefficients,
                angular_channel=0,
                zeta_indices=(1, 1),
                element="C",
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.select_radial_columns(
                coefficients,
                angular_channel=0,
                zeta_indices=(3,),
                element="C",
            )


if __name__ == "__main__":
    unittest.main()
