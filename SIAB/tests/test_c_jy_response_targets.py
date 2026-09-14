import unittest

from c_jy_response_targets import validate_target_manifest


def sector(q_slot, index=1):
    return dict(q_slot=q_slot, target_kind='response_covariance_embedding',
                assembled_pi=False, primitive_count=992,
                covariance_dimension=4, embedding_rows=4, embedding_columns=992,
                target_norm2=1.0, source_ik=index)


class CjyResponseTargetContractTest(unittest.TestCase):
    def manifest(self):
        sectors = [sector(q, index) for q in range(8)
                   for index in range(1, 65)]
        return dict(status='success', target_kind='response_covariance_embedding',
                    assembled_pi=False, lmax=3, frequency_count=12,
                    k_record_count=64, primitive_count=992, sectors=sectors)

    def test_complete_spdf_manifest_is_accepted(self):
        self.assertTrue(validate_target_manifest(self.manifest()))

    def test_single_q_manifest_is_accepted_for_array_output(self):
        manifest = self.manifest()
        manifest['sectors'] = [item for item in manifest['sectors'] if item['q_slot'] == 2]
        for item in manifest['sectors']:
            item['q_slot'] = 2
        self.assertTrue(validate_target_manifest(manifest, q_slots=(2,)))

    def test_pi_matrix_cannot_be_used_as_compression_target(self):
        with self.assertRaisesRegex(ValueError, 'PI matrix'):
            validate_target_manifest(dict(self.manifest(), assembled_pi=True))

    def test_missing_q_or_k_sector_is_rejected(self):
        bad = self.manifest()
        bad['sectors'] = bad['sectors'][:-1]
        with self.assertRaisesRegex(ValueError, 'complete q/k'):
            validate_target_manifest(bad)

    def test_target_must_be_uncontracted(self):
        with self.assertRaisesRegex(ValueError, 'uncontracted'):
            validate_target_manifest(dict(self.manifest(), primitive_count=558))


if __name__ == '__main__':
    unittest.main()
