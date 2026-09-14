import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'opt_orb_pytorch_dpsi'))
try:
    from response_radial_fit import ResponseFitSector, shared_radial_fit_loss
except ImportError:
    ResponseFitSector = shared_radial_fit_loss = None
from periodic_galerkin_basis import build_primitive_to_candidate
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock


class RadialFitTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ResponseFitSector, 'shared radial fitting is not implemented')
        self.rng = np.random.default_rng(93)
        self.blocks = tuple(PeriodicGalerkinPrimitiveBlock('C', atom, 1, m, 3,
            (atom*3+m+1)*3) for atom in range(2) for m in (-1, 0, 1))

    def matrix(self, rows, cols):
        return self.rng.normal(size=(rows, cols))+1j*self.rng.normal(size=(rows, cols))

    def case(self):
        c = torch.tensor([[1.], [.3], [-.2]], dtype=torch.float64, requires_grad=True)
        coefficients = {'C': (torch.empty((3, 0), dtype=torch.float64), c)}
        data, sectors = [], []
        for index, weight in enumerate((.3, .7)):
            embedding = self.matrix(14, 18)
            snapshots = self.matrix(14, 9)
            weights = np.linspace(.1, .9, 9)*weight
            cov = (snapshots*weights)@snapshots.conj().T
            data.append((embedding, snapshots, weights))
            sectors.append(ResponseFitSector(str(index), self.blocks, embedding, cov,
                                              occupied_rank=4))
        return coefficients, sectors, data

    def test_covariance_loss_matches_all_snapshots_with_shared_radials(self):
        coefficients, sectors, data = self.case()
        value, diagnostics = shared_radial_fit_loss(coefficients, sectors)
        c = build_primitive_to_candidate(self.blocks, 18, coefficients).transform.detach().numpy()
        expected = norm = 0.
        for embedding, snapshots, weights in data:
            y = embedding@c
            residual = snapshots-y@np.linalg.pinv(y)@snapshots
            expected += float(np.sum(abs(residual)**2*weights))
            norm += float(np.sum(abs(snapshots)**2*weights))
        self.assertAlmostEqual(value.item(), expected/norm, places=12)
        self.assertEqual(diagnostics['radial_orbital_count'], 1)
        self.assertEqual(diagnostics['nominal_ao_per_element'], {'C': 3})
        self.assertEqual(diagnostics['virtual_rank_signature'], [6, 6])
        self.assertEqual(diagnostics['augmented_total_rank_by_sector'], [10, 10])
        self.assertEqual(diagnostics['physical_release_gate'], 'hold')

    def test_gradient_matches_finite_difference(self):
        coefficients, sectors, _ = self.case()
        c = coefficients['C'][1]
        value, _ = shared_radial_fit_loss(coefficients, sectors)
        value.backward()
        analytical = c.grad.detach().numpy().copy()
        self.assertTrue(np.isfinite(analytical).all())
        self.assertGreater(np.linalg.norm(analytical), 1e-5)
        for i in range(3):
            values = []
            for sign in (-1, 1):
                perturbed = c.detach().clone()
                perturbed[i, 0] += sign*1e-6
                values.append(shared_radial_fit_loss({'C': (coefficients['C'][0], perturbed)}, sectors)[0].item())
            self.assertAlmostEqual(analytical[i, 0], (values[1]-values[0])/2e-6, places=7)

    def test_projected_null_columns_have_finite_fixed_rank_gradient(self):
        block = (PeriodicGalerkinPrimitiveBlock('C', 0, 0, 0, 4, 0),)
        c = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]],
                         dtype=torch.float64, requires_grad=True)
        embedding = np.eye(4)[1:]
        sector = ResponseFitSector('null', block, embedding, np.diag([1., 2., 3.]), occupied_rank=1)
        loss, diagnostics = shared_radial_fit_loss({'C': (c,)}, [sector])
        loss.backward()
        self.assertTrue(torch.isfinite(c.grad).all().item())
        self.assertEqual(diagnostics['virtual_rank_signature'], [1])
        self.assertAlmostEqual(loss.item(), 5/6)

    def test_degenerate_augmented_metric_has_finite_occupied_gradient(self):
        block = (PeriodicGalerkinPrimitiveBlock('C', 0, 0, 0, 4, 0),)
        c = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]],
                         dtype=torch.float64, requires_grad=True)
        sector = ResponseFitSector(
            'degenerate-occupied', block, np.eye(4)[1:], np.diag([1., 2., 3.]),
            occupied_rank=1, occupied_embedding=np.eye(4)[:1])
        loss, diagnostics = shared_radial_fit_loss(
            {'C': (c,)}, [sector], occupied_weight=1.)
        loss.backward()
        self.assertTrue(torch.isfinite(c.grad).all().item())
        self.assertAlmostEqual(diagnostics['minimum_occupied_capture'], 1., places=12)
        self.assertAlmostEqual(diagnostics['occupied_residual'], 0., places=12)

    def test_unitary_coordinates_and_radial_rescaling_leave_loss_invariant(self):
        coefficients, sectors, data = self.case()
        value, _ = shared_radial_fit_loss(coefficients, sectors)
        transformed = []
        for index, (embedding, snapshots, weights) in enumerate(data):
            unitary, _ = np.linalg.qr(self.matrix(14, 14))
            x = unitary@snapshots
            transformed.append(ResponseFitSector(str(index), self.blocks, unitary@embedding,
                (x*weights)@x.conj().T, occupied_rank=4))
        scaled = {'C': (coefficients['C'][0], 7*coefficients['C'][1])}
        changed, _ = shared_radial_fit_loss(scaled, transformed)
        self.assertAlmostEqual(value.item(), changed.item(), places=12)

    def test_invalid_covariance_empty_targets_and_duplicate_sector_rejected(self):
        with self.assertRaises(ValueError):
            ResponseFitSector('bad', self.blocks, np.eye(18), np.diag([-1.]+[1.]*17), occupied_rank=0)
        coefficients, sectors, _ = self.case()
        for selected in ([], sectors+[sectors[0]]):
            with self.assertRaises(ValueError):
                shared_radial_fit_loss(coefficients, selected)
        with self.assertRaises(ValueError):
            shared_radial_fit_loss(coefficients, sectors, relative_singular_tolerance=0)

    def test_missing_angular_operator_blocks_cannot_be_filled_with_new_coefficients(self):
        coefficients, sectors, _ = self.case()
        unsupported = {'C': (*coefficients['C'], torch.ones((3, 1), dtype=torch.float64))}
        with self.assertRaisesRegex(ValueError, 'missing primitive blocks'):
            shared_radial_fit_loss(unsupported, sectors)

    def test_occupied_embedding_penalizes_lost_fixed_manifold(self):
        block = (PeriodicGalerkinPrimitiveBlock('C', 0, 0, 0, 3, 0),)
        occupied = np.array([[1., 0., 0.]])
        virtual = np.array([[0., 1., 0.], [0., 0., 1.]])
        sector = ResponseFitSector('occupied', block, virtual, np.eye(2),
                                   occupied_embedding=occupied, occupied_rank=1)
        good = {'C': (torch.tensor([[1., 0.], [0., 1.], [0., 0.]],
                                   dtype=torch.float64, requires_grad=True),)}
        bad = {'C': (torch.tensor([[0., 0.], [1., 0.], [0., 1.]],
                                  dtype=torch.float64, requires_grad=True),)}
        good_loss, good_report = shared_radial_fit_loss(good, [sector], occupied_weight=2.)
        bad_loss, bad_report = shared_radial_fit_loss(bad, [sector], occupied_weight=2.)
        self.assertAlmostEqual(good_report['minimum_occupied_capture'], 1., places=12)
        self.assertAlmostEqual(good_report['occupied_residual'], 0., places=12)
        self.assertAlmostEqual(bad_report['minimum_occupied_capture'], 0., places=12)
        self.assertAlmostEqual(bad_report['occupied_residual'], 1., places=12)
        self.assertGreater(bad_loss.item(), good_loss.item())

    def test_occupied_embedding_contract_is_explicit(self):
        block = (PeriodicGalerkinPrimitiveBlock('C', 0, 0, 0, 3, 0),)
        with self.assertRaisesRegex(ValueError, 'occupied embedding'):
            ResponseFitSector('bad-occ', block, np.eye(2, 3), np.eye(2),
                              occupied_embedding=np.eye(2, 3), occupied_rank=1)


if __name__ == '__main__':
    unittest.main()
