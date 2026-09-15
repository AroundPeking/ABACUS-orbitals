import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


RUNNER = (Path(__file__).resolve().parents[1]/'example_C_sternheimer'
          /'periodic_basis_optimization/galerkin_binding_workflow'
          /'run_c_jy_rank_capacity_bound.py')
runner = None
if RUNNER.exists():
    spec = importlib.util.spec_from_file_location('c_jy_rank_capacity_bound', RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


class CjYRankCapacityBoundRunnerTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, 'rank-capacity bound runner is not implemented')

    def test_aggregate_uses_cell_ao_minus_occupied_rank(self):
        sectors = [
            dict(q_slot=0, occupied_rank=1, covariance=np.diag([9., 4., 1.])),
            dict(q_slot=1, occupied_rank=1, covariance=np.diag([8., 2.])),
        ]
        result = runner.aggregate_rank_bounds(
            sectors, profiles=((1, 0, 0, 0, 0),), atoms_per_cell=2)
        row = result['profile_results'][0]
        self.assertEqual(row['ao_per_C'], 1)
        self.assertEqual(row['nominal_cell_ao'], 2)
        self.assertEqual(row['virtual_rank_by_sector'], [1, 1])
        self.assertAlmostEqual(row['target_norm2'], 24.)
        self.assertAlmostEqual(row['residual_norm2_lower_bound'], 7.)
        self.assertAlmostEqual(row['response_loss_lower_bound'], 7./24.)
        self.assertEqual(row['physical_release_gate'], 'hold')
        self.assertEqual(result['scope'], 'independent_sector_rank_only_lower_bound')

    def test_aggregate_rejects_missing_q_or_impossible_occupied_dimension(self):
        valid = [dict(q_slot=0, occupied_rank=1, covariance=np.eye(2))]
        with self.assertRaises(ValueError):
            runner.aggregate_rank_bounds([], profiles=((1, 0, 0, 0, 0),),
                                         atoms_per_cell=2)
        with self.assertRaises(ValueError):
            runner.aggregate_rank_bounds(valid, profiles=((1, 0, 0, 0, 0),),
                                         atoms_per_cell=0)
        bad = [dict(q_slot=0, occupied_rank=2, covariance=np.eye(2))]
        with self.assertRaises(ValueError):
            runner.aggregate_rank_bounds(bad, profiles=((1, 0, 0, 0, 0),),
                                         atoms_per_cell=1)

    def test_aggregate_diagonalizes_each_sector_only_once(self):
        sectors = [
            dict(q_slot=0, occupied_rank=1, covariance=np.diag([9., 4., 1.])),
            dict(q_slot=1, occupied_rank=1, covariance=np.diag([8., 2.])),
        ]
        profiles = ((1, 0, 0, 0, 0), (2, 0, 0, 0, 0))
        original = np.linalg.eigvalsh
        with mock.patch.object(np.linalg, 'eigvalsh', wraps=original) as eigvalsh:
            runner.aggregate_rank_bounds(
                sectors, profiles=profiles, atoms_per_cell=2)
        self.assertEqual(eigvalsh.call_count, len(sectors))

    def test_runner_and_wrapper_are_diagnostic_only(self):
        source = RUNNER.read_text(encoding='ascii')
        wrapper = RUNNER.with_suffix('.slurm').read_text(encoding='ascii')
        for text in (source, wrapper):
            self.assertNotIn('abacus_3p', text)
            self.assertNotIn('librpa.x', text.lower())
        self.assertIn("physical_release_gate='hold'", source)
        self.assertIn('no_physics_run=True', source)
        self.assertIn('mkdir execution-once.lock', wrapper)


if __name__ == '__main__':
    unittest.main()
