import unittest
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / 'example_C_sternheimer'
          / 'periodic_basis_optimization/galerkin_binding_workflow'
          / 'run_c_jy_response_targets.slurm')


class CjyResponseTargetSlurmTest(unittest.TestCase):
    def test_wrapper_selects_only_cached_target_runners(self):
        text = SCRIPT.read_text()
        self.assertIn('run_c_jy_angular_ladder.py', text)
        self.assertIn('run_c_jy_joined_ladder.py', text)
        self.assertNotIn('abacus_3p', text)
        self.assertNotIn('librpa', text.lower())

    def test_wrapper_binds_runtime_job_id_and_protects_single_execution(self):
        text = SCRIPT.read_text()
        self.assertIn('mkdir execution-once.lock', text)
        self.assertIn('__RUNTIME_JOB_ID__', text)
        self.assertIn('$SLURM_JOB_ID', text)
        self.assertIn('RUNTIME_CONTRACT.json', text)


if __name__ == '__main__':
    unittest.main()
