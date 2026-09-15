import importlib.util
from pathlib import Path
import unittest
import tempfile
import numpy as np

PATH = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/audit_c_jy_operator_restart.py'
SPEC = importlib.util.spec_from_file_location('operator_audit', PATH)
MODULE = importlib.util.module_from_spec(SPEC)


class OperatorCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SPEC.loader.exec_module(MODULE)

    def test_bra_gauge_and_source_transformation(self):
        rng = np.random.default_rng(12)
        old = rng.normal(size=(2, 7)) + 1j*rng.normal(size=(2, 7))
        rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)))
        gauge, diagnostics = MODULE.occupied_gauge(old, rotation @ old)
        np.testing.assert_allclose(gauge, rotation, atol=1e-13)
        self.assertLess(diagnostics['relative'], 1e-13)
        self.assertLess(diagnostics['unitarity'], 1e-13)
        t, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3)))
        d = rng.normal(size=(2, 3, 7)) + 1j*rng.normal(size=(2, 3, 7))
        expected = np.empty_like(d)
        for n in range(2):
            expected[n] = sum(rotation[n, m] * (t.conj().T @ d[m]) for m in range(2))
        np.testing.assert_allclose(MODULE.transform_source(d, rotation, t), expected, atol=1e-13)

    def test_metric_signs_are_derived_without_response_fitting(self):
        v = np.array([[4., 1., .2], [1., 3., .4], [.2, .4, 2.]])
        sign = np.array([1., -1., 1.])
        actual = MODULE.metric_signs(v, sign[:, None]*v*sign[None, :])
        np.testing.assert_array_equal(actual, sign)

    def test_invalid_or_nonfinite_arrays_fail(self):
        with self.assertRaises(ValueError):
            MODULE.difference(np.array([float('nan')]), np.array([1.]))
        with self.assertRaises(ValueError):
            MODULE.difference(np.zeros((2, 1)), np.zeros((1, 2)))

    def test_native_chunk_header_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'overlap.bin'
            values = np.eye(2, dtype='<c16')
            path.write_bytes(MODULE.HEADER.pack(b'ABACUS_STBOPT_V1', 1, 1, 1, 1, -1, 2, 2)
                             + values.tobytes())
            reader = MODULE.OperatorFiles.__new__(MODULE.OperatorFiles)
            reader.directory = root.resolve()
            reader.hashes = {}
            reader.entries = {(1, 1, -1): dict(path=path.name, sha256=MODULE.sha(path),
                iq=1, rows=2, columns=2)}
            np.testing.assert_array_equal(reader.array(1, 1), values)

    def test_expanded_blocks_preserve_each_nested_radial_prefix(self):
        def block_text(rows):
            lines = ['ABACUS_STERNHEIMER_BASIS_OPT_PRIMITIVES_V1',
                     '# element atom_index l m n_primitive offset']
            offset = 0
            for atom in range(2):
                for l in range(2):
                    for m in range(-l, l + 1):
                        lines.append('C %d %d %d %d %d' %
                                     (atom, l, m, rows, offset))
                        offset += rows
            return '\n'.join(lines) + '\n'

        indices = MODULE.validate_nested_primitive_blocks(
            block_text(3), block_text(5), source_rows=3,
            target_rows=5, lmax=1, natom=2)
        self.assertEqual(indices, (0, 1, 2, 5, 6, 7, 10, 11, 12,
                                   15, 16, 17, 20, 21, 22, 25, 26, 27,
                                   30, 31, 32, 35, 36, 37))
        changed = block_text(5).replace('C 1 1 1 5 35', 'C 1 1 1 4 35')
        with self.assertRaisesRegex(ValueError, 'expanded primitive blocks'):
            MODULE.validate_nested_primitive_blocks(
                block_text(3), changed, source_rows=3,
                target_rows=5, lmax=1, natom=2)

    def test_expanded_prefix_selectors_preserve_old_shapes(self):
        indices = (0, 1, 3, 4)
        square = np.arange(36).reshape(6, 6)
        columns = np.arange(18).reshape(3, 6)
        np.testing.assert_array_equal(
            MODULE.nested_square_prefix(square, indices),
            square[np.ix_(indices, indices)])
        np.testing.assert_array_equal(
            MODULE.nested_column_prefix(columns, indices), columns[:, indices])

    def test_expanded_overlap_gate_covers_measured_roundoff_but_not_drift(self):
        measured = {'max_abs': 1.3164935808163136e-10,
                    'relative': 1.6093233430541642e-13}
        self.assertTrue(MODULE.expanded_overlap_compatible(measured))
        self.assertFalse(MODULE.expanded_overlap_compatible(
            dict(measured, max_abs=2.0000000001e-10)))
        self.assertFalse(MODULE.expanded_overlap_compatible(
            dict(measured, relative=1.0000000001e-11)))


if __name__ == '__main__':
    unittest.main()
