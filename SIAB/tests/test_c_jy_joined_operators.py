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
        with self.assertRaisesRegex(ValueError, 'subblock'):
            module.join_operator_record(dict(old, source=old['source']*2), original, new,
                dict(occupied_at_k1=a, auxiliary_map=t), (0,1,2), 5)

    def test_requires_passed_same_q_audit_and_no_regenerated_h(self):
        self.assertIsNotNone(module)
        row = dict(status='success', finite_q_spd_source_compatibility='pass',
            failure_reasons=[], selected_iq=22, regenerated_hamiltonian_read=False,
            regenerated_hamiltonian_admitted=False, full_source_primitive_count=1550,
            compared_primitive_count=558, per_k=[dict(source_ik=k, target_ik=k,
                pass_gate=True) for k in range(1,65)])
        module.validate_source_audit(row, 22)
        for changes in (dict(selected_iq=43), dict(per_k=row['per_k'][:-1]),
                        dict(regenerated_hamiltonian_read=True),
                        dict(finite_q_spd_source_compatibility='hold')):
            with self.assertRaises(ValueError):
                module.validate_source_audit(dict(row, **changes), 22)

if __name__ == '__main__':
    unittest.main()
