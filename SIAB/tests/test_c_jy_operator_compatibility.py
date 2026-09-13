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


if __name__ == '__main__':
    unittest.main()
