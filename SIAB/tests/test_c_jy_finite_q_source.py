import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

WF = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(WF))
PATH = WF/'audit_c_jy_finite_q_source.py'
MODULE = None
if PATH.exists():
    SPEC = importlib.util.spec_from_file_location('finite_source_audit', PATH)
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)


class FiniteQSourceTest(unittest.TestCase):
    def test_numeric_cache_label_and_internal_q_are_both_required(self):
        rows = [dict(label=label, selected_iq=iq) for label, iq in
                ((1, 1), (2, 22), (3, 43), (6, 6), (7, 27), (8, 23), (11, 11), (28, 55))]
        for index, row in enumerate(rows[1:], 1):
            self.assertEqual(MODULE.finite_q_dataset_position(rows, row['selected_iq']), index)
        rows[2]['selected_iq'] = 22
        with self.assertRaises(ValueError):
            MODULE.finite_q_dataset_position(rows, 43)
        with self.assertRaises(ValueError):
            MODULE.finite_q_dataset_position(rows + [rows[1]], 22)

    def test_source_k_gauge_not_target_k_gauge(self):
        self.assertIsNotNone(MODULE, 'finite-q source audit not implemented')
        from audit_c_jy_operator_restart import transform_source
        rng = np.random.default_rng(24)
        old = rng.normal(size=(2, 3, 5)) + 1j*rng.normal(size=(2, 3, 5))
        a = np.array([[0, 1j], [1, 0]], complex)
        target_a = np.diag([1j, -1])
        t = np.linalg.qr(rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3)))[0]
        new = transform_source(old, a, t)
        restored, result = MODULE.restore_and_compare_source(old, new, a, t)
        np.testing.assert_allclose(restored, old, atol=1e-13)
        self.assertTrue(result['pass_gate'])
        self.assertFalse(MODULE.restore_and_compare_source(old, new, target_a, t)[1]['pass_gate'])

    def test_physical_difference_and_nonfinite_are_rejected(self):
        self.assertIsNotNone(MODULE)
        old = np.ones((2, 3, 5), complex)
        a, t = np.eye(2), np.eye(3)
        self.assertFalse(MODULE.restore_and_compare_source(old, old*1.01, a, t)[1]['pass_gate'])
        with self.assertRaises(ValueError):
            MODULE.restore_and_compare_source(old, old*np.nan, a, t)

    def test_target_routing_is_inverted_before_source_gauge(self):
        self.assertIsNotNone(MODULE)
        mapping = MODULE.target_to_source({1: (2, None), 2: (3, None), 3: (1, None)}, 3)
        self.assertEqual(mapping, {2: 1, 3: 2, 1: 3})
        with self.assertRaises(ValueError):
            MODULE.target_to_source({1: (2, None), 2: (2, None), 3: (1, None)}, 3)

    def test_active_indices_are_remapped_blockwise_into_expanded_mother(self):
        self.assertEqual(
            MODULE.remap_nested_active_indices(
                (0, 2, 3, 5), source_rows=3, target_rows=5,
                source_primitive_count=6, target_primitive_count=10),
            (0, 2, 5, 7))
        with self.assertRaisesRegex(ValueError, 'nested active'):
            MODULE.remap_nested_active_indices(
                (0, 6), source_rows=3, target_rows=5,
                source_primitive_count=6, target_primitive_count=10)

    def test_expanded_finite_q_overlap_uses_the_nested_roundoff_gate(self):
        measured = {'max_abs': 1.3164935808163136e-10,
                    'relative': 1.6093233430541642e-13}
        self.assertFalse(MODULE.finite_q_overlap_compatible(measured, False))
        self.assertTrue(MODULE.finite_q_overlap_compatible(measured, True))
        self.assertFalse(MODULE.finite_q_overlap_compatible(
            dict(measured, max_abs=2.0000000001e-10), True))
        self.assertFalse(MODULE.finite_q_overlap_compatible(
            dict(measured, relative=1.0000000001e-11), True))


if __name__ == '__main__':
    unittest.main()
