from dataclasses import dataclass
import types
import unittest

import numpy as np
import torch

import common  # noqa: F401
import c_jy_compressed_evaluation as evaluation


class CompressedEvaluationTest(unittest.TestCase):
    def specs(self):
        profiles = ((3, 3, 2, 1, 0), (4, 4, 3, 2, 0),
                    (5, 5, 4, 3, 0), (6, 6, 5, 4, 0))
        return [dict(profile=list(profile), coefficients_path='/tmp/%d' % index,
                     coefficients_sha256=str(index) * 64,
                     ao_per_C=sum((2*l+1)*count
                                  for l, count in enumerate(profile)))
                for index, profile in enumerate(profiles, 1)]

    def test_fixed_rank_ladder_is_validated(self):
        specs = self.specs()
        self.assertEqual(
            [row['ao_per_C'] for row in evaluation.validate_profile_specs(specs)],
            [29, 45, 61, 77])
        inputs = {row['coefficients_path']: row['coefficients_sha256'] for row in specs}
        evaluation.validate_profile_input_hashes(specs, inputs)
        inputs[specs[0]['coefficients_path']] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'input hash'):
            evaluation.validate_profile_input_hashes(specs, inputs)
        changed = self.specs()
        changed[2]['ao_per_C'] += 1
        with self.assertRaisesRegex(ValueError, 'AO count'):
            evaluation.validate_profile_specs(changed)

    def test_profiles_are_evaluated_with_fixed_physical_controls(self):
        calls = []
        checkpoints = []
        dataset = types.SimpleNamespace(frequency_ha=torch.arange(12))

        def read(path, *, element, radial_rows, expected_nu):
            self.assertEqual((element, radial_rows), ('C', 31))
            return dict(path=path, profile=expected_nu)

        def evaluate(current_dataset, coefficients, **controls):
            calls.append((current_dataset, coefficients, controls))
            index = len(calls)
            return types.SimpleNamespace(
                response=torch.full((1, 1, 1), float(index)),
                minimum_occupied_capture=.9999995,
                maximum_overlap_condition=10. + index,
                minimum_candidate_rank=50 + index)

        def summarize(current_dataset, response):
            self.assertIs(current_dataset, dataset)
            return dict(candidate_energy_ha=-.01 * float(response[0, 0, 0]),
                        reference_energy_ha=-.1, q_weight=.125)

        rows = evaluation.evaluate_compressed_profiles(
            dataset, self.specs(), read_coefficients=read,
            evaluate_response=evaluate, summarize_energy=summarize,
            relative_rank_tolerance=1e-10, occupied_capture_floor=.99999,
            profile_callback=checkpoints.append)
        self.assertEqual([row['ao_per_C'] for row in rows], [29, 45, 61, 77])
        self.assertEqual([row['candidate_energy_ha'] for row in rows],
                         [-.01, -.02, -.03, -.04])
        for _, _, controls in calls:
            self.assertEqual(controls['contraction_backend'], 'block')
            self.assertEqual(controls['relative_rank_tolerance'], 1e-10)
            self.assertEqual(controls['condition_limit'], 1e12)
            self.assertAlmostEqual(controls['occupied_capture_tolerance'], 1e-5)
            self.assertEqual(controls['frequency_batch_size'], 12)
        self.assertEqual(
            [row['ao_per_C'] for row in checkpoints],
            [29, 45, 61, 77],
        )
        self.assertTrue(all(row['physical_release_gate'] == 'hold' for row in rows))

    def test_block_cache_is_prepared_once_before_all_rank_evaluations(self):
        @dataclass(frozen=True)
        class Dataset:
            frequency_ha: torch.Tensor
            primitive_blocks: tuple
            kpoints: tuple

        dataset = Dataset(
            frequency_ha=torch.arange(12),
            primitive_blocks=('blocks',),
            kpoints=('k1', 'k2'),
        )

        def read(path, *, element, radial_rows, expected_nu):
            return dict(path=path, profile=expected_nu)

        prepared = []

        def prepare(record, blocks, coefficients):
            self.assertEqual(blocks, dataset.primitive_blocks)
            self.assertEqual(coefficients['profile'], (3, 3, 2, 1, 0))
            prepared.append(record)
            return 'prepared-' + record

        def evaluate(current_dataset, coefficients, **controls):
            self.assertEqual(
                current_dataset.kpoints,
                ('prepared-k1', 'prepared-k2'),
            )
            return types.SimpleNamespace(
                response=torch.ones((1, 1, 1)),
                minimum_occupied_capture=.9999995,
                maximum_overlap_condition=11.,
                minimum_candidate_rank=29,
            )

        rows = evaluation.evaluate_compressed_profiles(
            dataset,
            self.specs(),
            read_coefficients=read,
            evaluate_response=evaluate,
            summarize_energy=lambda current, response: dict(
                candidate_energy_ha=-.1,
                reference_energy_ha=-.2,
                q_weight=.125,
            ),
            prepare_block_cache=prepare,
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(prepared, ['k1', 'k2'])


if __name__ == '__main__':
    unittest.main()
