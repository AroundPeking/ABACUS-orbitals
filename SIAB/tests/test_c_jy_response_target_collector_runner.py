import json
from pathlib import Path
import sys
import unittest


WORKFLOW = (Path(__file__).resolve().parents[1] / 'example_C_sternheimer'
            / 'periodic_basis_optimization/galerkin_binding_workflow')
sys.path.insert(0, str(WORKFLOW))


class CjyResponseTargetCollectorRunnerTest(unittest.TestCase):
    def fixture(self):
        checked = dict(status='success', target_completeness_gate='pass',
                       occupied_constraints_complete=True,
                       target_sector_count=512,
                       per_q=[{'source_commit': 'a' * 40},
                              {'source_commit': 'b' * 40}])
        source = dict(status='success', lmax=3, primitive_count=992,
                      frequency_count=12, target_sector_count=512,
                      reference_energy_ha_per_cell=-.5144842779929139,
                      mother_candidate_energy_ha_per_cell=-.5091040675004793,
                      mother_error_ev_per_C=.0732014929)
        return checked, source

    def test_finalize_preserves_reference_and_marks_occupied_collection(self):
        from collect_c_jy_response_targets import finalize_collection

        checked, source = self.fixture()
        result = finalize_collection(checked, source,
            collector_source_commit='c' * 40,
            collector_source_manifest_sha256='d' * 64,
            source_summary_sha256='e' * 64,
            scheduler_job_id='123', scheduler_partition='48cp2',
            elapsed_seconds=2., max_rss_kb=10)

        self.assertEqual(result['stage'],
                         'occupied_augmented_complete_spdf_target_collection')
        self.assertEqual(result['target_source_commits'], ['a' * 40, 'b' * 40])
        self.assertEqual(result['reference_energy_ha_per_cell'],
                         source['reference_energy_ha_per_cell'])
        self.assertTrue(result['occupied_constraints_complete'])
        self.assertEqual(result['physical_release_gate'], 'hold')

    def test_finalize_rejects_incomplete_or_incompatible_source(self):
        from collect_c_jy_response_targets import finalize_collection

        checked, source = self.fixture()
        checked['occupied_constraints_complete'] = False
        with self.assertRaisesRegex(ValueError, 'occupied'):
            finalize_collection(checked, source,
                collector_source_commit='c' * 40,
                collector_source_manifest_sha256='d' * 64,
                source_summary_sha256='e' * 64,
                scheduler_job_id='123', scheduler_partition='48cp2',
                elapsed_seconds=2., max_rss_kb=10)

    def test_slurm_wrapper_is_cached_algebra_only_and_locked(self):
        text = (WORKFLOW / 'collect_c_jy_response_targets.slurm').read_text()
        self.assertIn('mkdir execution-once.lock', text)
        self.assertIn('collect_c_jy_response_targets.py', text)
        self.assertNotIn('abacus_3p', text)
        self.assertNotIn('librpa', text.lower())
        self.assertNotIn('delta-st', text.lower())


if __name__ == '__main__':
    unittest.main()
