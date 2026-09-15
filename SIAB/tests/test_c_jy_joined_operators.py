import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

WF = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(WF))
import importlib.util
spec = importlib.util.find_spec('c_jy_joined_operators')
module = importlib.import_module('c_jy_joined_operators') if spec else None


class JoinedOperatorTest(unittest.TestCase):
    def test_anchors_complete_old_mother_in_expanded_operator_record(self):
        self.assertIsNotNone(module, 'joined operator adapter missing')
        rng = np.random.default_rng(23)
        old_size, new_size = 5, 8
        prefix = (0, 1, 3, 4, 6)
        old = {
            'overlap': rng.normal(size=(old_size, old_size)),
            'hamiltonian_ha': rng.normal(size=(old_size, old_size)),
            'occupied_projection': rng.normal(size=(2, old_size)),
            'source': rng.normal(size=(2, 3, old_size)),
        }
        expanded = {
            'overlap': rng.normal(size=(new_size, new_size)),
            'hamiltonian_ha': rng.normal(size=(new_size, new_size)),
            'occupied_projection': rng.normal(size=(2, new_size)),
            'source': rng.normal(size=(2, 3, new_size)),
        }
        result, checks = module.anchor_nested_operator_record(
            expanded, old, prefix)
        np.testing.assert_array_equal(
            result['overlap'][np.ix_(prefix, prefix)], old['overlap'])
        np.testing.assert_array_equal(
            result['hamiltonian_ha'][np.ix_(prefix, prefix)],
            old['hamiltonian_ha'])
        np.testing.assert_array_equal(
            result['occupied_projection'][:, prefix],
            old['occupied_projection'])
        np.testing.assert_array_equal(result['source'][:, :, prefix],
                                      old['source'])
        self.assertTrue(checks['exact_old_mother_anchor'])
        self.assertEqual(checks['old_primitive_count'], old_size)
        self.assertEqual(checks['expanded_primitive_count'], new_size)
        self.assertNotEqual(result['overlap'][2, 2], old['overlap'][2, 2])

    def test_uses_target_original_operators_and_source_occupied_gauge(self):
        self.assertIsNotNone(module, 'joined operator adapter missing')
        from audit_c_jy_operator_restart import transform_source
        rng = np.random.default_rng(18)
        d = rng.normal(size=(2, 3, 5))+1j*rng.normal(size=(2, 3, 5))
        o = np.eye(2, 5, dtype=complex)
        a = np.array([[0, 1j], [1, 0]], complex)
        b = np.diag([1j, -1])
        t = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))[0]
        h = np.diag(np.arange(5, dtype=float)).astype(complex)
        h[1, 4], h[4, 1] = .3j, -.3j
        s = np.eye(5, dtype=complex)
        calls = []
        def original(kind, ik):
            calls.append((kind, ik))
            self.assertEqual(ik, 2)
            return {1:s, 6:2*h, 7:o}[kind]
        def new(kind, ik):
            self.assertEqual((kind, ik), (2, 1))
            return transform_source(d, a, t).reshape(6, 5)
        old = dict(source_ik=1, target_ik=2, source=d[:, :, :3],
                   overlap=s[:3, :3], hamiltonian_ha=h[:3, :3],
                   occupied_projection=b@o[:, :3])
        result, checks = module.join_operator_record(old, original, new,
            dict(occupied_at_k1=a, occupied_at_k2=np.eye(2), auxiliary_map=t),
            (0, 1, 2), 5)
        np.testing.assert_allclose(result['source'], d, atol=1e-13)
        np.testing.assert_allclose(result['occupied_projection'], b@o, atol=1e-13)
        np.testing.assert_allclose(result['hamiltonian_ha'], h)
        self.assertEqual(calls, [(1, 2), (6, 2), (7, 2)])
        self.assertTrue(checks['pass_gate'])
        self.assertEqual(checks['hamiltonian_subblock_gate'],
                         'required_and_pass')
        with self.assertRaisesRegex(ValueError, 'subblock'):
            module.join_operator_record(dict(old, source=old['source']*2), original, new,
                dict(occupied_at_k1=a, auxiliary_map=t), (0,1,2), 5)
        with self.assertRaisesRegex(ValueError, 'subblock'):
            module.join_operator_record(old,
                lambda kind, ik: (2*h + 2e-7*np.eye(5)) if kind == 6
                else original(kind, ik), new,
                dict(occupied_at_k1=a, auxiliary_map=t), (0,1,2), 5)

    def test_nested_join_reproduces_baseline_on_all_old_primitive_columns(self):
        self.assertIsNotNone(module, 'joined operator adapter missing')
        rng = np.random.default_rng(29)
        old_size, new_size = 5, 8
        prefix = (0, 1, 3, 4, 6)
        active = (0, 1, 2)
        expanded_active = tuple(prefix[index] for index in active)
        s = np.eye(old_size, dtype=complex)
        h = np.diag(np.arange(old_size, dtype=float)).astype(complex)
        o = np.eye(2, old_size, dtype=complex)
        d = rng.normal(size=(2, 3, old_size))
        sn = np.eye(new_size, dtype=complex)
        hn = np.diag(np.arange(new_size, dtype=float)).astype(complex)
        on = rng.normal(size=(2, new_size)).astype(complex)
        dn = rng.normal(size=(2, 3, new_size)).astype(complex)
        sn[np.ix_(prefix, prefix)] = s
        hn[np.ix_(prefix, prefix)] = h
        on[:, prefix] = o
        dn[:, :, prefix] = d

        def baseline_original(kind, ik):
            return {1: s, 6: 2*h, 7: o}[kind]

        def baseline_source(kind, ik):
            self.assertEqual(kind, 2)
            return d.reshape(6, old_size)

        def expanded_original(kind, ik):
            return {1: sn, 6: 2*hn, 7: on}[kind]

        def expanded_source(kind, ik):
            self.assertEqual(kind, 2)
            return dn.reshape(6, new_size)

        maps = dict(occupied_at_k1=np.eye(2), auxiliary_map=np.eye(3))
        cache = dict(source_ik=1, target_ik=2, source=d[:, :, active],
                     overlap=s[np.ix_(active, active)],
                     hamiltonian_ha=h[np.ix_(active, active)],
                     occupied_projection=o[:, active])
        result, checks = module.join_nested_operator_record(
            cache, baseline_original, baseline_source, maps,
            expanded_original, expanded_source, maps,
            baseline_indices=active, expanded_indices=expanded_active,
            prefix_indices=prefix, baseline_size=old_size,
            expanded_size=new_size)
        np.testing.assert_array_equal(
            result['overlap'][np.ix_(prefix, prefix)], s)
        np.testing.assert_array_equal(result['source'][:, :, prefix], d)
        self.assertTrue(checks['anchor']['exact_old_mother_anchor'])
        self.assertTrue(checks['baseline']['pass_gate'])
        self.assertTrue(checks['expanded']['pass_gate'])

    def test_nested_join_only_diagnoses_expanded_h_before_exact_anchor(self):
        self.assertIsNotNone(module, 'joined operator adapter missing')
        old_size, new_size = 5, 8
        prefix = (0, 1, 3, 4, 6)
        active = (0, 1, 2)
        expanded_active = tuple(prefix[index] for index in active)
        s = np.eye(old_size, dtype=complex)
        h = np.diag(np.arange(old_size, dtype=float)).astype(complex)
        o = np.eye(2, old_size, dtype=complex)
        d = np.arange(30, dtype=float).reshape(2, 3, old_size)
        sn = np.eye(new_size, dtype=complex)
        hn = np.diag(np.arange(new_size, dtype=float)).astype(complex)
        on = np.zeros((2, new_size), dtype=complex)
        dn = np.zeros((2, 3, new_size), dtype=complex)
        sn[np.ix_(prefix, prefix)] = s
        hn[np.ix_(prefix, prefix)] = h
        hn[prefix[1], prefix[1]] += 2e-7
        on[:, prefix] = o
        dn[:, :, prefix] = d

        def baseline_original(kind, ik):
            return {1: s, 6: 2*h, 7: o}[kind]

        def baseline_source(kind, ik):
            return d.reshape(6, old_size)

        def expanded_original(kind, ik):
            return {1: sn, 6: 2*hn, 7: on}[kind]

        def expanded_source(kind, ik):
            return dn.reshape(6, new_size)

        maps = dict(occupied_at_k1=np.eye(2), auxiliary_map=np.eye(3))
        cache = dict(source_ik=1, target_ik=2, source=d[:, :, active],
                     overlap=s[np.ix_(active, active)],
                     hamiltonian_ha=h[np.ix_(active, active)],
                     occupied_projection=o[:, active])
        result, checks = module.join_nested_operator_record(
            cache, baseline_original, baseline_source, maps,
            expanded_original, expanded_source, maps,
            baseline_indices=active, expanded_indices=expanded_active,
            prefix_indices=prefix, baseline_size=old_size,
            expanded_size=new_size)
        np.testing.assert_array_equal(
            result['hamiltonian_ha'][np.ix_(prefix, prefix)], h)
        self.assertEqual(checks['baseline']['hamiltonian_subblock_gate'],
                         'required_and_pass')
        self.assertEqual(checks['expanded']['hamiltonian_subblock_gate'],
                         'diagnostic_only_before_exact_anchor')
        self.assertGreater(checks['expanded']['hamiltonian_ha']['max_abs'],
                           1e-10)
        self.assertTrue(checks['expanded']['pass_gate'])

    def test_expanded_gamma_is_mapped_back_before_exact_anchor(self):
        self.assertIsNotNone(module, 'joined operator adapter missing')
        from audit_c_jy_operator_restart import transform_source
        rng = np.random.default_rng(31)
        old_size, new_size = 4, 7
        prefix = (0, 2, 3, 6)
        a = np.array([[0, 1j], [1, 0]], dtype=complex)
        t = np.linalg.qr(rng.normal(size=(3, 3))
                         + 1j*rng.normal(size=(3, 3)))[0]
        old = dict(
            overlap=np.eye(old_size, dtype=complex),
            hamiltonian_ha=np.diag(np.arange(old_size)).astype(complex),
            occupied_projection=rng.normal(size=(2, old_size)).astype(complex),
            source=(rng.normal(size=(2, 3, old_size))
                    + 1j*rng.normal(size=(2, 3, old_size))),
        )
        expanded_old_gauge = dict(
            overlap=np.eye(new_size, dtype=complex),
            hamiltonian_ha=np.diag(np.arange(new_size)).astype(complex),
            occupied_projection=(rng.normal(size=(2, new_size))
                                 + 1j*rng.normal(size=(2, new_size))),
            source=(rng.normal(size=(2, 3, new_size))
                    + 1j*rng.normal(size=(2, 3, new_size))),
        )
        expanded_old_gauge['overlap'][np.ix_(prefix, prefix)] = old['overlap']
        expanded_old_gauge['hamiltonian_ha'][np.ix_(prefix, prefix)] = old['hamiltonian_ha']
        expanded_old_gauge['occupied_projection'][:, prefix] = old['occupied_projection']
        expanded_old_gauge['source'][:, :, prefix] = old['source']
        native = dict(
            overlap=expanded_old_gauge['overlap'],
            hamiltonian_ha=expanded_old_gauge['hamiltonian_ha'],
            occupied_projection=a @ expanded_old_gauge['occupied_projection'],
            source=transform_source(expanded_old_gauge['source'], a, t),
        )
        result, checks = module.map_and_anchor_expanded_gamma_record(
            native, old, occupied_map=a, auxiliary_map=t,
            prefix_indices=prefix)
        for name in old:
            np.testing.assert_allclose(result[name], expanded_old_gauge[name],
                                       atol=1e-13)
        self.assertTrue(checks['exact_old_mother_anchor'])

    def test_requires_passed_same_q_audit_and_no_regenerated_h(self):
        self.assertIsNotNone(module)
        row = dict(status='success', finite_q_spd_source_compatibility='pass',
            failure_reasons=[], selected_iq=22, regenerated_hamiltonian_read=False,
            regenerated_hamiltonian_admitted=False, full_source_primitive_count=1550,
            compared_primitive_count=558, per_k=[dict(source_ik=k, target_ik=k,
                pass_gate=True) for k in range(1,65)])
        module.validate_source_audit(row, 22)
        expanded = dict(row, full_source_primitive_count=2400,
                        primitive_expansion=dict(
                            source_radial_rows=31,
                            expanded_radial_rows=48,
                            source_primitive_count=1550,
                            expanded_primitive_count=2400,
                            expanded_spdf_primitive_count=1536))
        module.validate_source_audit(
            expanded, 22, full_source_primitive_count=2400)
        for changes in (dict(selected_iq=43), dict(per_k=row['per_k'][:-1]),
                        dict(regenerated_hamiltonian_read=True),
                        dict(finite_q_spd_source_compatibility='hold')):
            with self.assertRaises(ValueError):
                module.validate_source_audit(dict(row, **changes), 22)
        with self.assertRaises(ValueError):
            module.validate_source_audit(
                expanded, 22, full_source_primitive_count=1550)

if __name__ == '__main__':
    unittest.main()
