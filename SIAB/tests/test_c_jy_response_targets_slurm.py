import unittest
from unittest import mock
from pathlib import Path

import numpy as np


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

    def test_finite_q_runner_loads_contract_from_its_immutable_source(self):
        runner = (SCRIPT.parent / 'run_c_jy_joined_ladder.py').read_text()
        self.assertIn('spec_from_file_location', runner)
        self.assertIn("repo/'SIAB/opt_orb_pytorch_dpsi/c_jy_response_targets.py'", runner)

    def test_finite_q_runner_adapts_legacy_cache_targets_to_occupied_targets(self):
        runner = (SCRIPT.parent / 'run_c_jy_joined_ladder.py').read_text()
        self.assertIn('def occupied_embedding_from_record(', runner)
        self.assertIn('def consume_legacy_target(', runner)
        self.assertIn('build_legacy_response_targets(', runner)
        self.assertIn('occupied_embedding=occupied_embedding', runner)

    def test_q_runners_support_direct_compressed_rank_rpa_evaluation(self):
        for name in ('run_c_jy_angular_ladder.py', 'run_c_jy_joined_ladder.py'):
            runner = (SCRIPT.parent / name).read_text()
            self.assertIn("compressed_shared_radial_full_q_body_RPA", runner)
            self.assertIn('evaluate_compressed_profiles(', runner)
            self.assertIn(
                'prepare_block_cache=prepare_periodic_block_contraction_record',
                runner,
            )
            self.assertIn("['lmax_values'] == [3]", runner)

    def test_finite_q_callback_accepts_current_five_argument_contract(self):
        import sys

        workflow = SCRIPT.parent
        optimizer = workflow.parents[2] / 'opt_orb_pytorch_dpsi'
        sys.path[:0] = [str(workflow), str(optimizer)]
        from run_c_jy_joined_ladder import consume_compatible_target

        covariance = np.eye(2)
        occupied = np.ones((1, 2))
        embedding = np.eye(2)
        target_pi = np.zeros((1, 1, 1))
        metadata = {'source_ik': 4, 'occupied_rank': 1}
        received = []

        consume_compatible_target(
            (covariance, occupied, embedding, target_pi, metadata),
            lambda *args: received.append(args), {}, 1e-10)

        self.assertEqual(len(received), 1)
        self.assertIs(received[0][1], occupied)
        self.assertEqual(received[0][4]['occupied_embedding_adapter'],
                         'provided_by_response_target')

    def test_finite_q_callback_rebuilds_only_legacy_four_argument_contract(self):
        import sys

        workflow = SCRIPT.parent
        optimizer = workflow.parents[2] / 'opt_orb_pytorch_dpsi'
        sys.path[:0] = [str(workflow), str(optimizer)]
        import run_c_jy_joined_ladder as runner

        occupied = np.ones((1, 2))
        metadata = {'source_ik': 4, 'occupied_rank': 1}
        received = []
        report = {'retained_rank': 2, 'occupied_rank': 1,
                  'minimum_capture': .999999}
        with mock.patch.object(runner, 'occupied_embedding_from_record',
                               return_value=(occupied, report)) as rebuild:
            runner.consume_compatible_target(
                (np.eye(2), np.eye(2), np.zeros((1, 1, 1)), metadata),
                lambda *args: received.append(args), {4: object()}, 1e-10)

        rebuild.assert_called_once()
        self.assertIs(received[0][1], occupied)
        self.assertEqual(received[0][4]['occupied_embedding_adapter'],
                         'rebuilt_from_frozen_reader_metric')


if __name__ == '__main__':
    unittest.main()
