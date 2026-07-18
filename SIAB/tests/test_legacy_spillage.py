import unittest
import torch

from common import info
from opt_orbital_spillage import Opt_Orbital_Spillage


class LegacySpillageTest(unittest.TestCase):
    def test_origin_only_loss_remains_point_four(self):
        info_stru = [info(Na={"H": 1}, Nb_true=1, weight=torch.tensor([1.0]))]
        info_element = {"H": info(Nl=1, Ne=2, Nu=[1])}
        q = {"H": [torch.tensor([[0.8, 0.6]], dtype=torch.complex128)]}
        s = {
            ("H", "H"): [[torch.eye(2, dtype=torch.complex128).reshape(1, 1, 2, 1, 1, 2)]]
        }
        c = {"H": [torch.tensor([[1.0], [0.0]], dtype=torch.float64, requires_grad=True)]}
        target = [torch.tensor([1.0], dtype=torch.float64)]
        loss = Opt_Orbital_Spillage(
            info_stru, info_element, {"same_band": True}, "one", {"origin": ["synthetic"]}
        )
        loss.set_QSVI([q], [s], target)
        self.assertAlmostEqual(loss.cal_Spillage(c).item(), 0.4, places=14)


if __name__ == "__main__":
    unittest.main()
