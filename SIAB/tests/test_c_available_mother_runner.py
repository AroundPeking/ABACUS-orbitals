import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'example_C_sternheimer'
                       /'periodic_basis_optimization/galerkin_binding_workflow'))
import run_c_available_mother as runner


class MotherSummaryTest(unittest.TestCase):
    def test_cutoff_cli_default_and_explicit_control(self):
        flags = ['--bundle', '/bundle', '--output', '/output',
                 '--source-commit', 'a'*40, '--q-slot', '0']
        self.assertTrue(hasattr(runner, 'parse_args'))
        self.assertEqual(runner.parse_args(flags).relative_rank_tolerance, 1e-12)
        self.assertEqual(runner.parse_args(flags+['--relative-rank-tolerance', '1e-10'])
                         .relative_rank_tolerance, 1e-10)

    def test_invalid_cutoff_is_rejected_before_input_io(self):
        self.assertTrue(hasattr(runner, 'validate_rank_tolerance'))
        for value in (0., -1e-10, 1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'rank tolerance'):
                runner.validate_rank_tolerance(value)

    def test_cutoff_must_match_unique_registered_job(self):
        self.assertTrue(hasattr(runner, 'validate_job_contract'))
        job = dict(job_id='123', source_commit='a'*40, bundle_sha256=runner.BUNDLE_SHA,
                   relative_rank_tolerance=1e-10)
        runner.validate_job_contract(job, '123', 'a'*40, 1e-10)
        for changes in ({'relative_rank_tolerance': 1e-12}, {'job_id': '124'},
                        {'source_commit': 'b'*40}, {'bundle_sha256': 'b'*64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.validate_job_contract(dict(job, **changes), '123', 'a'*40, 1e-10)
        del job['relative_rank_tolerance']
        with self.assertRaises(ValueError):
            runner.validate_job_contract(job, '123', 'a'*40, 1e-10)

    def test_runner_preflight_has_no_dependency_on_optimizer_cli(self):
        with patch.dict('os.environ', {}, clear=True), patch.dict(
                sys.modules, {'optimize_c_all_radial_fast': None}):
            with self.assertRaisesRegex(ValueError, 'registered single-node DF'):
                runner.run(Path('/not-opened'), Path('/not-created'), 0, 'a'*40)

    def rows(self):
        return [dict(status='success', selected_iq=iq, q_weight=m/64.,
                     candidate_energy_ha=-.5*m/64.,
                     reference_energy_ha=runner.REFERENCE*m/64.,
                     frequency_ha=list(range(12)), frequency_weights_ha=[1.]*12,
                     source_commit='a'*40, bundle_sha256=runner.BUNDLE_SHA,
                     shared_protocol={'protocol': 'test'},
                     space='available_active_reduced_mother',
                     available_primitive_count=558, original_primitive_count=1550,
                     relative_rank_tolerance=1e-12, k_record_count=64,
                     physical_release_gate='hold')
                for iq, m in zip(runner.INDICES, runner.MULT)]

    def test_full_weights_and_reference_are_reconstructed_without_renormalizing(self):
        result = runner.summarize(self.rows())
        self.assertAlmostEqual(result['candidate_energy_ha'], -.5)
        self.assertAlmostEqual(result['reference_energy_ha'], runner.REFERENCE)
        self.assertEqual(result['physical_release_gate'], 'hold')
        self.assertFalse(result['exact_reference_snapshot_fit'])
        self.assertFalse(result['available_mother_energy_gate'])

    def test_missing_duplicate_wrong_protocol_or_frequency_fails(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            runner.summarize(rows[:-1])
        with self.assertRaises(ValueError):
            runner.summarize(rows[:-1]+[rows[0]])
        for field, value in [('frequency_ha', [1.]*12), ('q_weight', .9),
                             ('reference_energy_ha', -1.), ('status', 'failed'),
                             ('shared_protocol', {'other': True}),
                             ('k_record_count', 63), ('source_commit', 'b'*40),
                             ('relative_rank_tolerance', 1e-10),
                             ('candidate_energy_ha', float('nan'))]:
            changed = copy.deepcopy(rows)
            changed[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.summarize(changed)


if __name__ == '__main__':
    unittest.main()
