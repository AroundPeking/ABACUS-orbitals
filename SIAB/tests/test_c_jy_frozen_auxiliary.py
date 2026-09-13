import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/prepare_c_jy_operator_restart.py'
SPEC = importlib.util.spec_from_file_location('frozen_aux_restart', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrozenAuxiliaryTest(unittest.TestCase):
    def fixture(self, root):
        cache = root / 'cache'
        cache.mkdir()
        arrays = []
        for i, value in enumerate((np.eye(3, dtype=complex), np.eye(3, dtype=complex)[:, :2])):
            path = cache / ('array_%06d.npy' % i)
            np.save(path, value, allow_pickle=False)
            arrays.append(dict(file=path.name, sha256=MODULE.sha(path), dtype=value.dtype.str,
                               shape=list(value.shape)))
        meta = dict(format_version=1, arrays=arrays,
            dataset=dict(fields=dict(coulomb_metric=dict(type='tensor', index=0),
                                     coulomb_whitening=dict(type='tensor', index=1))),
            source=dict(scalar=dict(selected_iq=['22'], raw_auxiliary_dimension=['3'],
                                    whitened_auxiliary_rank=['2'], kernel=['full_coulomb'])))
        (cache / 'dataset.json').write_text(json.dumps(meta))
        digest = MODULE.sha(cache / 'dataset.json')
        (cache / 'COMPLETE.json').write_text(json.dumps(dict(format_version=1, status='complete',
                                                           metadata_sha256=digest)))
        return cache, digest

    def test_writes_same_rank_and_values_without_rewhitening(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, digest = self.fixture(root)
            result = MODULE.freeze_auxiliary_cache(cache, root / 'frozen', 22, digest)
            self.assertEqual(result['whitened_auxiliary_rank'], 2)
            self.assertEqual(result['metadata_sha256'], digest)
            data = (root / 'frozen/coulomb_whitening.bin').read_bytes()
            h = struct.unpack_from('<16sIIiiiQQ', data)
            self.assertEqual(h, (b'ABACUS_STBOPT_V1', 1, 5, 22, 0, -1, 3, 2))
            np.testing.assert_array_equal(np.frombuffer(data[52:], dtype='<c16').reshape(3, 2),
                                          np.eye(3, dtype=complex)[:, :2])
            self.assertEqual(result['whitening_sha256'], MODULE.sha(root / 'frozen/coulomb_whitening.bin'))
            with self.assertRaises(FileExistsError):
                MODULE.freeze_auxiliary_cache(cache, root / 'frozen', 22, digest)

    def test_rejects_wrong_q_or_unlocked_metadata_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, digest = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                MODULE.freeze_auxiliary_cache(cache, root / 'frozen', 22, '0' * 64)
            with self.assertRaisesRegex(ValueError, 'q'):
                MODULE.freeze_auxiliary_cache(cache, root / 'frozen', 23, digest)
            self.assertFalse((root / 'frozen').exists())

    def test_rejects_modified_array_and_incomplete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, digest = self.fixture(root)
            (cache / 'array_000001.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'array'):
                MODULE.freeze_auxiliary_cache(cache, root / 'frozen', 22, digest)
            (cache / 'COMPLETE.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'complete'):
                MODULE.freeze_auxiliary_cache(cache, root / 'frozen', 22, digest)
            self.assertFalse((root / 'frozen').exists())

    def test_frozen_mode_cannot_bypass_source_extension_admission(self):
        for iq in (1, 22, 43):
            with self.assertRaisesRegex(ValueError, 'source-extension'):
                MODULE.prepare(Path('reference'), Path('build'), Path('output'), iq=iq,
                               frozen_auxiliary_cache=Path('cache'))


if __name__ == '__main__':
    unittest.main()
