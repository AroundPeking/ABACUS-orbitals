import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (Path(__file__).resolve().parents[1]/'example_C_sternheimer'
          /'periodic_basis_optimization/galerkin_binding_workflow'
          /'prepare_c_jy_compressed_rank_ladder.py')
spec = importlib.util.spec_from_file_location('prepare_c_jy_compressed', SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PrepareCjYCompressedRankLadderTest(unittest.TestCase):
    def fixtures(self, root):
        templates = root/'templates'
        templates.mkdir()
        shared = root/'shared.dat'
        shared.write_text('shared\n', encoding='ascii')
        for slot in range(8):
            stage = templates/('q%02d' % slot)
            stage.mkdir()
            contract = dict(job_id=str(100+slot), source_commit='a'*40,
                scope='old target generation', lmax_values=[2, 3, 4],
                relative_rank_tolerance=1e-10,
                inputs={str(shared): prepare.sha(shared)})
            if slot:
                contract['q_slot'] = slot
            (stage/'RUNTIME_CONTRACT.json').write_text(
                json.dumps(contract), encoding='ascii')
        candidates = []
        for profile in prepare.PROFILES:
            path = root/('coeff-' + ''.join(map(str, profile[:4])) + '.txt')
            path.write_text('candidate\n', encoding='ascii')
            candidates.append(dict(profile=list(profile),
                ao_per_C=prepare.ao_per_element(profile),
                coefficients_path=str(path)))
        return templates, candidates

    def test_writes_eight_hash_locked_compressed_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            templates, candidates = self.fixtures(root)
            output = root/'output'
            result = prepare.prepare_contracts(
                templates, output, candidates, source_commit='b'*40)
            self.assertEqual(result['q_slots'], list(range(8)))
            self.assertEqual(result['profiles'], [list(row) for row in prepare.PROFILES])
            for slot in range(8):
                path = output/('q%02d' % slot)/'CONTRACT.json'
                contract = json.loads(path.read_text(encoding='ascii'))
                self.assertEqual(contract['job_id'], '__RUNTIME_JOB_ID__')
                self.assertEqual(contract['q_slot'], slot)
                self.assertEqual(contract['source_commit'], 'b'*40)
                self.assertEqual(contract['lmax_values'], [3])
                self.assertEqual(contract['occupied_capture_floor'], .99999)
                self.assertEqual(contract['scope'],
                                 'compressed_shared_radial_full_q_body_RPA')
                self.assertEqual(len(contract['candidate_profiles']), 4)
                for candidate in contract['candidate_profiles']:
                    path = candidate['coefficients_path']
                    self.assertEqual(contract['inputs'][path],
                                     candidate['coefficients_sha256'])

    def test_rejects_incomplete_profile_order_and_mismatched_template_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            templates, candidates = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, 'profile'):
                prepare.prepare_contracts(
                    templates, root/'bad-profiles', candidates[::-1],
                    source_commit='b'*40)
            path = templates/'q03/RUNTIME_CONTRACT.json'
            contract = json.loads(path.read_text(encoding='ascii'))
            contract['q_slot'] = 4
            path.write_text(json.dumps(contract), encoding='ascii')
            with self.assertRaisesRegex(ValueError, 'q slot'):
                prepare.prepare_contracts(
                    templates, root/'bad-slot', candidates,
                    source_commit='b'*40)


if __name__ == '__main__':
    unittest.main()
