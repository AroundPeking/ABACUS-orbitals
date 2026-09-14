import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'opt_orb_pytorch_dpsi'))
try:
    from c_jy_response_target_collection import (
        collect_target_collection,
        load_angular_primitive_blocks,
    )
except ImportError:
    collect_target_collection = None
    load_angular_primitive_blocks = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CjYResponseTargetCollectionTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(collect_target_collection,
                             'the jY response target collector is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.blocks = [
            dict(element='C', atom_index=0, l=0, m=0, n_primitive=3, offset=0),
        ]
        for slot, (label, selected_iq, multiplicity) in enumerate(((1, 1, 1), (2, 22, 8))):
            qroot = self.root/('q%02d' % slot)
            target = qroot/'targets'
            result = qroot/'result'
            (result/'lmax3').mkdir(parents=True)
            target.mkdir(parents=True)
            covariance = np.diag([1., 2.]).astype(np.complex128)
            embedding = np.array([[1., 0., .5], [0., 1., .25]], dtype=np.complex128)
            with (target/'k0001.npz').open('wb') as stream:
                np.savez(stream, covariance=covariance, embedding=embedding)
            pi = np.arange(8, dtype=float).reshape(2, 2, 2)
            np.save(target/'PI_DIAGNOSTIC.npy', pi, allow_pickle=False)
            np.save(result/'lmax3/PI.npy', pi, allow_pickle=False)
            manifest = dict(status='success', target_kind='response_covariance_embedding',
                assembled_pi=False, lmax=3, q_slot=slot, selected_iq=selected_iq,
                q_weight=multiplicity/9, frequency_count=2,
                frequency_ha=[.1, 1.], frequency_weights_ha=[.2, .3],
                k_record_count=1, primitive_count=3, physical_release_gate='hold',
                sectors=[dict(q_slot=slot, target_kind='response_covariance_embedding',
                    assembled_pi=False, lmax=3, selected_iq=selected_iq,
                    source_ik=1, target_ik=1, frequency_count=2, primitive_count=3,
                    occupied_rank=1, covariance_dimension=2, embedding_rows=2,
                    embedding_columns=3, target_norm2=3., file='k0001.npz',
                    file_sha256=sha(target/'k0001.npz'))])
            (target/'TARGETS.json').write_text(json.dumps(manifest))
            result_payload = dict(status='success', q_slot=slot, selected_iq=selected_iq)
            if slot == 0:
                result_payload.pop('selected_iq')
            (result/'RESULT.json').write_text(json.dumps(result_payload))
            provenance = dict(status='success', source_commit='abc', job_id=str(100+slot),
                files={'RESULT.json': sha(result/'RESULT.json'),
                       'lmax3/PI.npy': sha(result/'lmax3/PI.npy')})
            (result/'PROVENANCE.json').write_text(json.dumps(provenance))
            runtime = dict(source_commit='abc', job_id=str(100+slot))
            (qroot/'CONTRACT.json').write_text(json.dumps(runtime))
            (qroot/'RUNTIME_CONTRACT.json').write_text(json.dumps(runtime))

    def tearDown(self):
        self.temp.cleanup()

    def collect(self):
        return collect_target_collection(self.root, self.blocks,
            q_slots=(0, 1), q_labels=(1, 2), selected_iq=(1, 22),
            multiplicities=(1, 8), k_record_count=1, frequency_count=2,
            primitive_count=3, lmax=3, star_weight_denominator=9)

    def test_collects_hash_bound_local_sector_targets(self):
        result = self.collect()
        self.assertEqual(result['target_completeness_gate'], 'pass')
        self.assertEqual(result['target_sector_count'], 2)
        self.assertEqual(result['primitive_blocks'], self.blocks)
        self.assertEqual(result['q_labels'], [1, 2])
        self.assertEqual(result['selected_iq'], [1, 22])
        self.assertEqual(result['multiplicities'], [1, 8])
        self.assertEqual([row['pi_reconstruction_max_abs'] for row in result['per_q']], [0., 0.])
        self.assertEqual(len(result['target_files']), 2)
        self.assertFalse(result['assembled_pi'])
        self.assertEqual(result['physical_release_gate'], 'hold')

    def test_rejects_changed_sector_file(self):
        path = self.root/'q01/targets/k0001.npz'
        path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError, 'sector file hash mismatch'):
            self.collect()

    def test_filters_and_reindexes_angular_primitive_blocks(self):
        self.assertIsNotNone(load_angular_primitive_blocks)
        path = self.root/'primitive_blocks.dat'
        path.write_text('\n'.join((
            'ABACUS_STERNHEIMER_BASIS_OPT_PRIMITIVES_V1',
            '# element atom_index l m n_primitive offset',
            'C 0 0 0 3 0',
            'C 0 1 -1 3 3',
            'C 0 1 0 3 6',
            'C 0 1 1 3 9',
            'C 0 2 -2 3 12',
        ))+'\n')
        blocks = load_angular_primitive_blocks(path, sha(path),
            source_primitive_count=15, lmax=1)
        self.assertEqual(len(blocks), 4)
        self.assertEqual([block['offset'] for block in blocks], [0, 3, 6, 9])
        self.assertEqual(sum(block['n_primitive'] for block in blocks), 12)


if __name__ == '__main__':
    unittest.main()
