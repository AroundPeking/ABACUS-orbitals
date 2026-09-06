import importlib
import tempfile
import unittest
from dataclasses import replace

import torch

import common  # noqa: F401
from periodic_galerkin_basis import contract_periodic_candidate_operators
from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock, read_periodic_galerkin_dataset
from periodic_galerkin_fit import _minimum_occupied_capture, optimize_periodic_galerkin_basis
from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
from periodic_galerkin_rpa import periodic_rpa_objective
from periodic_galerkin_sternheimer import (
    evaluate_periodic_galerkin_mother_response,
    evaluate_periodic_galerkin_response,
    prepare_periodic_occupied_reference,
)
import test_periodic_galerkin_fit as fixtures
import test_periodic_galerkin_data as data_fixtures


class ActivePrimitiveReductionTest(unittest.TestCase):
    def reduce(self, dataset, coefficients):
        name = 'periodic_galerkin_reduction'
        self.assertIsNotNone(importlib.util.find_spec(name), 'missing exact active reduction')
        return importlib.import_module(name).reduce_periodic_active_primitives(dataset, coefficients)

    def fixture(self):
        d = fixtures.PeriodicGalerkinFitTest().three_level_dataset()
        overlap = torch.eye(9, dtype=torch.complex128)
        overlap[1, 7] = 0.07j
        overlap[7, 1] = -0.07j
        h = torch.diag(torch.tensor([-.5, .7, 1.4, 2., 3., 4., .9, 1.1, 1.8])).to(torch.complex128)
        h[1, 6], h[6, 1] = .03j, -.03j
        record = replace(
            d.kpoints[0], overlap=overlap, hamiltonian_ha=h,
            occupied_projection=torch.tensor([[1., 0., 0., .02j, 0., 0., 0., 0., 0.]], dtype=torch.complex128),
            source=torch.tensor([[[0., .3, .2j, .4, .1j, .2, .15j, .11, .06]]], dtype=torch.complex128),
            reference_projection=None,
        )
        blocks = tuple(PeriodicGalerkinPrimitiveBlock('C', 0, l, 0, 3, 3*l) for l in range(3))
        d = replace(d, primitive_count=9, primitive_blocks=blocks, kpoints=(record,),
                    q_count=8, q_weight=.25)
        c = {'C': [torch.tensor([[1., 0.], [0., 1.], [0., .2]], dtype=torch.float64),
                   torch.empty((3, 0), dtype=torch.float64),
                   torch.tensor([[1.], [.2], [.1]], dtype=torch.float64)]}
        return d, c

    def test_noncontiguous_mapping_owns_smaller_storage_and_preserves_normalization(self):
        d, c = self.fixture()
        reduced = self.reduce(d, c)
        self.assertEqual(reduced.primitive_count, 6)
        self.assertEqual([b.offset for b in reduced.primitive_blocks], [0, 3])
        meta = reduced.active_primitive_reduction
        self.assertEqual(meta.source_indices, (0, 1, 2, 6, 7, 8))
        self.assertEqual(meta.original_primitive_count, 9)
        self.assertEqual(meta.original_primitive_blocks_sha256, d.primitive_blocks_sha256)
        self.assertEqual(len(meta.mapping_sha256), 64)
        self.assertEqual(meta, self.reduce(d, c).active_primitive_reduction)
        self.assertEqual(reduced.physics_hash, d.physics_hash)
        self.assertIs(reduced.reference_response, d.reference_response)
        self.assertIsNone(d.kpoints[0].occupied_projection_normalization)
        full = prepare_periodic_occupied_reference(d)
        a, b = full.kpoints[0], reduced.kpoints[0]
        torch.testing.assert_close(a.occupied_projection_normalization, b.occupied_projection_normalization, rtol=0, atol=0)
        for field in ('overlap', 'hamiltonian_ha', 'source', 'occupied_projection'):
            x, y = getattr(a, field), getattr(b, field)
            self.assertNotEqual(x.storage().data_ptr(), y.storage().data_ptr())
            self.assertEqual(y.storage().size(), y.numel())
            self.assertLess(y.numel(), x.numel())
        self.assertIsNone(b.block_contraction_cache)
        self.assertIs(self.reduce(reduced, c), reduced)
        self.assertAlmostEqual(_minimum_occupied_capture((full,), c),
                               _minimum_occupied_capture((reduced,), c), places=13)

    def test_accepts_real_reader_omitted_projection_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            data_fixtures.PeriodicGalerkinDataTest().write_fixture(directory)
            data = read_periodic_galerkin_dataset(directory, include_reference_projection=False)
            self.assertEqual(data.kpoints[0].reference_projection.numel(), 0)
            coefficients = {'C': [torch.eye(2, dtype=torch.float64)]}
            reduced = self.reduce(data, coefficients)
            self.assertEqual(reduced.kpoints[0].reference_projection.numel(), 0)
            torch.testing.assert_close(reduced.kpoints[0].source, data.kpoints[0].source, rtol=0, atol=0)

    def test_candidate_operators_pi_rpa_components_and_gradients_are_equivalent(self):
        d, c = self.fixture()
        full = prepare_periodic_occupied_reference(d)
        reduced = self.reduce(d, c)
        outputs = []
        for data in (full, reduced):
            coeff = {'C': [x.clone().requires_grad_(True) for x in c['C']]}
            operators = contract_periodic_candidate_operators(data.kpoints[0], data.primitive_blocks, coeff)
            response = evaluate_periodic_galerkin_coefficient_response(data, coeff, contraction_backend='block', occupied_capture_tolerance=.01)
            obj = periodic_rpa_objective((data,), (response.response,))
            obj.loss.backward()
            outputs.append((operators, response, obj, [x.grad for x in coeff['C']]))
        a, b = outputs
        for field in ('overlap', 'hamiltonian_ha', 'source', 'occupied_projection'):
            torch.testing.assert_close(getattr(a[0], field), getattr(b[0], field), rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(a[1].response, b[1].response, rtol=1e-12, atol=1e-13)
        self.assertAlmostEqual(a[1].minimum_occupied_capture, b[1].minimum_occupied_capture, places=13)
        for field in ('loss', 'pi_relative_squared_error', 'trace_log_relative_squared_error', 'energy_relative_squared_error', 'candidate_energy_ha', 'reference_energy_ha'):
            torch.testing.assert_close(getattr(a[2], field), getattr(b[2], field), rtol=1e-12, atol=1e-13)
        for l in (0, 2):
            torch.testing.assert_close(a[3][l], b[3][l], rtol=1e-11, atol=1e-12)

    def test_two_step_fit_and_fixed_prefix_match_without_changing_default(self):
        d, c = self.fixture()
        kwargs = dict(fixed_nu={'C': (1, 0, 0)}, objective='rpa', max_steps=2,
                      minimum_steps=0, learning_rate=.001)
        full = optimize_periodic_galerkin_basis((d,), c, **kwargs)
        reduced = optimize_periodic_galerkin_basis((self.reduce(d, c),), c, **kwargs)
        self.assertAlmostEqual(full.best_loss, reduced.best_loss, places=12)
        for a, b in zip(full.coefficients['C'], reduced.coefficients['C']):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(reduced.coefficients['C'][0][:, :1], c['C'][0][:, :1], rtol=0, atol=0)
        self.assertIsNone(d.active_primitive_reduction)

    def test_rejects_projection_diagnostics_and_new_channels(self):
        d, c = self.fixture()
        with self.assertRaisesRegex(ValueError, 'reference_projection'):
            self.reduce(replace(d, kpoints=(replace(d.kpoints[0], reference_projection=torch.zeros((1, 1, 1, 9))),)), c)
        reduced = self.reduce(d, c)
        other = {'C': list(c['C'])}
        other['C'][1] = torch.ones((3, 1), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'profile'):
            self.reduce(reduced, other)
        with self.assertRaisesRegex(ValueError, 'profile'):
            evaluate_periodic_galerkin_coefficient_response(reduced, other)
        with self.assertRaisesRegex(ValueError, 'unreduced'):
            evaluate_periodic_galerkin_mother_response(reduced)
        with self.assertRaisesRegex(ValueError, 'unreduced'):
            evaluate_periodic_galerkin_response(reduced, torch.eye(6, dtype=torch.complex128))


if __name__ == '__main__':
    unittest.main()
