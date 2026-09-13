import unittest

import numpy as np
import torch

import common  # noqa: F401
import response_fit_iteration as iteration
from response_radial_fit import ResponseFitSector, shared_radial_fit_loss
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock


class ResponseFitIterationTest(unittest.TestCase):
    def case(self):
        blocks = (PeriodicGalerkinPrimitiveBlock('C', 0, 0, 0, 4, 0),)
        sector = ResponseFitSector('toy', blocks, np.eye(4), np.diag([1., 2., 5., 9.]), occupied_rank=0)
        initial = {'C': (torch.tensor([[1., 0.], [.2, 1.], [.3, .4], [.5, .3]], dtype=torch.float64),)}
        return initial, [sector]

    def test_all_radials_free_monotone_bounded_steps(self):
        initial, sectors = self.case()
        before = initial['C'][0].clone()
        records = []
        fitted, report = iteration.fit_shared_response_radials(initial, sectors,
            max_steps=3, max_evaluations=12, radius=.1, progress=records.append)
        self.assertEqual(report['accepted_steps'], 3)
        self.assertLess(report['final_loss'], report['initial_loss'])
        self.assertLessEqual(report['evaluations'], 12)
        self.assertEqual(report['stopping_reason'], 'max_steps_not_convergence')
        self.assertEqual(report['physical_release_gate'], 'hold')
        self.assertEqual(report['free_radial_count'], 2)
        torch.testing.assert_close(initial['C'][0], before)
        losses = [r['loss'] for r in records if r['accepted']]
        self.assertTrue(all(a > b for a, b in zip(losses, losses[1:])))
        self.assertGreater(np.linalg.norm(fitted['C'][0].numpy()@fitted['C'][0].numpy().T
                                        -np.linalg.qr(before.numpy())[0]@np.linalg.qr(before.numpy())[0].T), .01)
        self.assertAlmostEqual(shared_radial_fit_loss(fitted, sectors)[0].item(), report['final_loss'], places=12)

    def test_evaluation_budget_and_empty_channel_preserved(self):
        initial, sectors = self.case()
        initial['C'] += (torch.empty((4, 0), dtype=torch.float64),)
        fitted, report = iteration.fit_shared_response_radials(initial, sectors,
            max_steps=20, max_evaluations=1, radius=.1)
        self.assertEqual(report['evaluations'], 1)
        self.assertEqual(report['accepted_steps'], 0)
        self.assertEqual(report['stopping_reason'], 'evaluation_budget')
        self.assertEqual(fitted['C'][1].shape, (4, 0))

    def test_invalid_controls_and_dependent_radials_rejected(self):
        initial, sectors = self.case()
        for controls in ({'max_steps': 0}, {'max_evaluations': 0}, {'radius': 0}, {'radius': float('nan')}):
            with self.assertRaises(ValueError):
                iteration.fit_shared_response_radials(initial, sectors, **controls)
        initial['C'][0][:, 1] = initial['C'][0][:, 0]
        with self.assertRaisesRegex(ValueError, 'radial rank'):
            iteration.fit_shared_response_radials(initial, sectors)


if __name__ == '__main__':
    unittest.main()
