import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


RUNNER = (Path(__file__).resolve().parents[1]/'example_C_sternheimer'
          /'periodic_basis_optimization/galerkin_binding_workflow'
          /'run_c_jy_all_channel_contraction.py')
spec = importlib.util.spec_from_file_location('c_jy_all_channel_runner', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CjYAllChannelContractionRunnerTest(unittest.TestCase):
    def test_rank_ladder_has_expected_compact_ao_counts(self):
        self.assertEqual(runner.PROFILES, ((3, 3, 2, 1, 0), (4, 4, 3, 2, 0),
                                           (5, 5, 4, 3, 0), (6, 6, 5, 4, 0)))
        self.assertEqual([runner.ao_per_element(profile) for profile in runner.PROFILES],
                         [29, 45, 61, 77])

    def test_collection_requires_all_occupied_augmented_sectors(self):
        valid = dict(status='success', target_completeness_gate='pass',
            occupied_constraints_complete=True, target_sector_count=512,
            primitive_count=992, lmax=3, q_array_slots=list(range(8)),
            k_record_count_per_q=64)
        runner.validate_collection(valid)
        for key, value in (('occupied_constraints_complete', False),
                           ('target_sector_count', 511), ('lmax', 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_collection(dict(valid, **{key: value}))

    def test_sector_loader_rejects_legacy_virtual_only_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'legacy.npz'
            np.savez(path, covariance=np.eye(2), embedding=np.eye(2, 3))
            with self.assertRaisesRegex(ValueError, 'occupied embedding'):
                runner.read_sector_arrays(path, dict(occupied_rank=1,
                    covariance_dimension=2, embedding_rows=2,
                    embedding_columns=3, occupied_embedding_rows=1))

    def test_runner_source_has_no_frozen_prefix_release(self):
        source = RUNNER.read_text(encoding='ascii')
        self.assertNotIn('frozen_prefix=', source)
        self.assertIn("stage='all_channel_occupied_safe_response_contraction'", source)
        self.assertIn("physical_release_gate='hold'", source)


if __name__ == '__main__':
    unittest.main()
