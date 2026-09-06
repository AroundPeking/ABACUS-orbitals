import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

import common
import test_periodic_galerkin_data as fixtures

WORKFLOW = Path(__file__).resolve().parents[1] / 'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0, str(WORKFLOW))


class RecordDiagnosticTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('compare_active_primitive_record'))
        return importlib.import_module('compare_active_primitive_record')

    def fixture(self, directory):
        root = Path(directory)
        fixtures.PeriodicGalerkinDataTest().write_fixture(directory)
        acceptance = root / 'ACCEPTANCE.json'
        acceptance.write_text('{"status":"success"}')
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        item = dict(label=1, dataset=str(root), acceptance=str(acceptance),
                    acceptance_sha256=digest(acceptance), manifest_sha256=digest(root/'manifest.dat'),
                    status_sha256=digest(root/'status.dat'), physics_hash='6'*64)
        freeze = root / 'FREEZE.json'
        freeze.write_text(json.dumps(dict(status='success', datasets=[item])))
        return freeze, digest(freeze)

    def test_reads_one_record_and_checks_every_chunk_without_projection_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            freeze, digest = self.fixture(directory)
            dataset, evidence = self.module()._load_sample(freeze, digest, 1, 1)
            self.assertEqual(len(dataset.kpoints), 1)
            self.assertEqual(dataset.kpoints[0].k_weight, 2.)
            self.assertEqual(dataset.kpoints[0].reference_projection.numel(), 0)
            self.assertEqual(evidence['checked_chunks'], 8)
            self.assertEqual(evidence['retained_chunks'], 7)
            self.assertEqual(evidence['scope'], 'single_k_equivalence_only_not_physical_energy')
            result = self.module().compare_sample(dataset, {'C': [torch.eye(2, dtype=torch.float64)]})
            self.assertEqual(result['equivalence_gate'], 'pass')
            self.assertLessEqual(result['maximum_response_absolute_difference'], 1e-13)
            self.assertNotIn('candidate_energy_ha', result)

    def test_corrupt_unused_projection_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            freeze, digest = self.fixture(directory)
            path = Path(directory) / 'response_ik_1_ifreq_0.bin'
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                self.module()._load_sample(freeze, digest, 1, 1)

    def test_requires_parent_freeze_identity_and_existing_record(self):
        with tempfile.TemporaryDirectory() as directory:
            freeze, digest = self.fixture(directory)
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                self.module()._load_sample(freeze, '0'*64, 1, 1)
            with self.assertRaisesRegex(RuntimeError, 'source_ik'):
                self.module()._load_sample(freeze, digest, 1, 2)


if __name__ == '__main__':
    unittest.main()
