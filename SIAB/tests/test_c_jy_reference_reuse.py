import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('reference_reuse', ROOT/'audit_c_jy_reference_reuse.py')
MODULE = importlib.util.module_from_spec(SPEC)


class ReferenceReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SPEC.loader.exec_module(MODULE)

    def fixture(self):
        rng = np.random.default_rng(4)
        o = rng.normal(size=(2, 5)).astype(complex)
        a = np.array([[0, 1j], [1, 0]], complex)
        old = dict(s=np.eye(5), h_ry=np.diag([1., 2, 3, 4, 5]), o=o,
                   source_eigen_ha=np.array([.2, .3]), target_eigen_ha=np.array([.4, .4]))
        new = dict(s=old['s'].copy(), h_ha=old['h_ry']*.5, o=a@o,
                   source_eigen_ha=old['source_eigen_ha'].copy(),
                   target_eigen_ha=old['target_eigen_ha'].copy())
        return old, new

    def test_units_and_occupied_bra_gauge(self):
        old, new = self.fixture()
        row = MODULE.compare_block(old, new)
        self.assertTrue(row['pass_gate'])
        self.assertEqual(row['hamiltonian_ha']['max_abs'], 0)
        self.assertLess(row['occupied']['relative'], 1e-13)

    def test_wrong_ry_ha_conversion_fails(self):
        old, new = self.fixture()
        new['h_ha'] = old['h_ry']
        self.assertFalse(MODULE.compare_block(old, new)['pass_gate'])

    def test_physical_occupied_change_and_nonfinite_rejected(self):
        old, new = self.fixture()
        new['o'][0, 3] += .01
        self.assertFalse(MODULE.compare_block(old, new)['pass_gate'])
        new['h_ha'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            MODULE.compare_block(old, new)

    def test_occupied_rotation_cannot_mix_distinct_energies(self):
        old, new = self.fixture()
        old['target_eigen_ha'] = np.array([.4, .6])
        new['target_eigen_ha'] = np.array([.4, .6])
        self.assertFalse(MODULE.compare_block(old, new)['pass_gate'])

    def test_routing_requires_bijection_and_exact_target_coordinate(self):
        MODULE.validate_routing([1, 2], [2, 1], [(.25, 0, 0), (0, 0, 0)],
                                {1: (0, 0, 0), 2: (.25, 0, 0)})
        with self.assertRaises(ValueError):
            MODULE.validate_routing([1, 2], [1, 1], [(0, 0, 0), (0, 0, 0)],
                                    {1: (0, 0, 0), 2: (.25, 0, 0)})
        with self.assertRaises(ValueError):
            MODULE.validate_routing([1, 2], [2, 1], [(0, .25, 0), (0, 0, 0)],
                                    {1: (0, 0, 0), 2: (.25, 0, 0)})

    def test_cache_reader_binds_original_tree_not_current_audit_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'reader.py').write_text('original')
            bindings = {'reader.py': hashlib.sha256(b'original').hexdigest()}
            MODULE.validate_reader_tree(root, bindings)
            (root/'reader.py').write_text('changed')
            with self.assertRaises(ValueError):
                MODULE.validate_reader_tree(root, bindings)
            with self.assertRaises(ValueError):
                MODULE.validate_reader_tree(root, {'../reader.py': bindings['reader.py']})


if __name__ == '__main__':
    unittest.main()
