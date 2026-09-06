import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref
from dataclasses import replace

import torch

import common  # noqa: F401
import periodic_galerkin_data as reader
from periodic_galerkin_reduction import reduce_periodic_active_primitives
from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
from periodic_galerkin_rpa import periodic_rpa_objective
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference
import test_periodic_galerkin_data as fixtures


def write_multik_fixture(directory):
    root = Path(directory)
    fixtures.PeriodicGalerkinDataTest().write_fixture(directory)
    scalars, _, _, _, _ = reader._read_manifest(directory)
    n = 27
    lines = ['ABACUS_STERNHEIMER_BASIS_OPT_PRIMITIVES_V1']
    offset = 0
    for l in range(3):
        for m in range(-l, l + 1):
            lines.append('C 0 {} {} 3 {}'.format(l, m, offset))
            offset += 3
    (root/'primitive_blocks.dat').write_text('\n'.join(lines)+'\n')
    values = {key: ' '.join(value) for key, value in scalars.items()}
    values.update(k_count='2', frequency_count='2', primitive_count=str(n),
                  primitive_blocks_sha256=reader._sha256(str(root/'primitive_blocks.dat')))
    entries = []

    def chunk(kind, ik, iw, matrix):
        matrix = matrix.to(torch.complex128)
        name = '{}_{}_{}.bin'.format(kind, ik, iw)
        rows, cols = matrix.shape
        payload = reader._HEADER.pack(reader._CHUNK_MAGIC, 1, kind, 1, ik, iw, rows, cols)
        payload += matrix.numpy().astype('<c16').tobytes()
        (root/name).write_bytes(payload)
        frequency = (.25, .8)[iw] if iw >= 0 else -1.
        weight = 1. if ik == 0 else (.75, 1.25)[ik-1]
        entries.append('entry\t{}\t1\t{}\t{}\t{}\t{}\t1\t{}\t{}\t{}\t{}'.format(
            kind, ik, iw, rows, cols, weight, frequency, name, hashlib.sha256(payload).hexdigest()))

    chunk(4, 0, -1, torch.eye(1))
    chunk(5, 0, -1, torch.eye(1))
    for iw in range(2):
        chunk(8, 0, iw, torch.tensor([[-.4/(iw+1)]]))
    for ik in range(1, 3):
        s = torch.eye(n, dtype=torch.complex128)
        s[1, 13], s[13, 1] = .03j*ik, -.03j*ik
        h = torch.diag(torch.arange(n, dtype=torch.float64)/10+.7).to(torch.complex128)
        h[0, 0] = -.5
        h[1, 13], h[13, 1] = .02j, -.02j
        occupied = torch.zeros((1, n), dtype=torch.complex128)
        occupied[0, 0], occupied[0, 5] = 1., .02j
        source = torch.arange(n, dtype=torch.float64).reshape(1, n)/100
        source = source.to(torch.complex128)*(1+.2j*ik)
        for kind, matrix in ((1, s), (6, 2*h), (7, occupied), (2, source)):
            chunk(kind, ik, -1, matrix)
        for iw in range(2):
            chunk(3, ik, iw, source/(iw+1))
    values['entry_count'] = str(len(entries))
    manifest = [reader._MANIFEST_MAGIC]+[key+' '+value for key, value in values.items()]
    manifest += ['frequency 0 .25 .7', 'frequency 1 .8 1.3']
    for ik in range(1, 3):
        manifest += ['kpoint {} {} 0 0 0 0 0 0 0 0 0 {} 1 1'.format(ik, ik, (.75, 1.25)[ik-1]),
                     'eigenvalues_ry {} 1 -1'.format(ik)]
    (root/'manifest.dat').write_text('\n'.join(manifest+entries)+'\n')
    return {'C': [torch.tensor([[1., 0.], [0., 1.], [0., .2]], dtype=torch.float64),
                  torch.empty((3, 0), dtype=torch.float64),
                  torch.tensor([[1.], [.2], [.1]], dtype=torch.float64)]}


class StreamingReductionTest(unittest.TestCase):
    def read_compact(self, directory, coefficients, **kwargs):
        return reader.read_periodic_galerkin_dataset(
            directory, include_reference_projection=False,
            active_coefficients=coefficients, **kwargs)

    def test_all_k_noncontiguous_reduction_matches_in_memory_and_mult_q_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            c = write_multik_fixture(directory)
            full = prepare_periodic_occupied_reference(
                reader.read_periodic_galerkin_dataset(directory, include_reference_projection=False))
            expected = reduce_periodic_active_primitives(full, c)
            actual = self.read_compact(directory, c)
            self.assertEqual(len(actual.kpoints), 2)
            self.assertEqual(actual.primitive_count, 18)
            self.assertEqual(actual.active_primitive_reduction, expected.active_primitive_reduction)
            self.assertEqual(actual.active_primitive_reduction.source_indices, tuple(range(3))+tuple(range(12, 27)))
            for a, b in zip(actual.kpoints, expected.kpoints):
                for field in ('overlap', 'hamiltonian_ha', 'source', 'occupied_projection', 'occupied_projection_normalization'):
                    torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
                    self.assertEqual(getattr(a, field).storage().size(), getattr(a, field).numel())
                self.assertEqual(a.k_weight, b.k_weight)
            snapshots = []
            for data in (full, actual):
                coeff = {'C': [v.clone().requires_grad_(True) for v in c['C']]}
                family = (replace(data, q_count=2, selected_iq=1, q_weight=.25),
                          replace(data, q_count=2, selected_iq=2, q_weight=.75))
                responses = tuple(evaluate_periodic_galerkin_coefficient_response(
                    q, coeff, occupied_capture_tolerance=.01).response for q in family)
                obj = periodic_rpa_objective(family, responses)
                obj.loss.backward()
                snapshots.append((torch.stack(responses).detach(),
                                  torch.stack([getattr(obj, f).detach() for f in ('loss', 'pi_relative_squared_error',
                                      'trace_log_relative_squared_error', 'energy_relative_squared_error')]),
                                  [v.grad for v in coeff['C'] if v.numel()]))
            for a, b in zip(snapshots[0][:2], snapshots[1][:2]):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-13)
            for a, b in zip(snapshots[0][2], snapshots[1][2]):
                torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_previous_full_k_operators_released_before_loading_next_k(self):
        with tempfile.TemporaryDirectory() as directory:
            c = write_multik_fixture(directory)
            original = reader._read_chunk
            refs = []

            def tracked(root, entry):
                if entry.ik == 2:
                    self.assertTrue(all(ref() is None for ref in refs), 'previous full k remains resident')
                result = original(root, entry)
                if entry.ik == 1 and entry.kind in (1, 2, 6, 7):
                    refs.append(weakref.ref(result))
                return result

            with patch.object(reader, '_read_chunk', side_effect=tracked):
                actual = self.read_compact(directory, c)
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertEqual(len(actual.kpoints), 2)

    def test_unused_projection_corruption_and_unsafe_modes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            c = write_multik_fixture(directory)
            with self.assertRaisesRegex(ValueError, 'projection'):
                reader.read_periodic_galerkin_dataset(directory, active_coefficients=c)
            with self.assertRaisesRegex(ValueError, 'omitted|hash'):
                self.read_compact(directory, c, verify_omitted_chunks=False)
            path = Path(directory)/'3_2_1.bin'
            blob = bytearray(path.read_bytes())
            blob[-1] ^= 1
            path.write_bytes(blob)
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                self.read_compact(directory, c)

    def test_new_profile_and_full_mother_diagnostics_stay_rejected(self):
        from periodic_galerkin_sternheimer import evaluate_periodic_galerkin_mother_response
        with tempfile.TemporaryDirectory() as directory:
            c = write_multik_fixture(directory)
            actual = self.read_compact(directory, c)
            with self.assertRaisesRegex(ValueError, 'unreduced'):
                evaluate_periodic_galerkin_mother_response(actual)
            c['C'][1] = torch.ones((3, 1), dtype=torch.float64)
            with self.assertRaisesRegex(ValueError, 'profile'):
                evaluate_periodic_galerkin_coefficient_response(actual, c)


if __name__ == '__main__':
    unittest.main()
