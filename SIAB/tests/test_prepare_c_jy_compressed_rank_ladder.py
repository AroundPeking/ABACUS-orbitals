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

    def test_writes_one_expanded_fixed_rank_contract_with_audited_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            templates, candidates = self.fixtures(root)
            candidate = [candidates[1]]
            for slot in range(1, 8):
                old_audit = templates/('q%02d' % slot)/'old-audit.json'
                self.write_baseline_audit(old_audit, selected_iq=slot)
                contract_path = templates/('q%02d' % slot)/'RUNTIME_CONTRACT.json'
                contract = json.loads(contract_path.read_text(encoding='ascii'))
                contract['audit_result'] = str(old_audit.resolve())
                contract_path.write_text(json.dumps(contract), encoding='ascii')
            gamma_run = self.accepted_run(root/'gamma-run')
            gamma_audit = self.accepted_audit(
                root/'gamma-audit', gamma_run, include_run=False)
            q_audits = {}
            for slot in range(1, 8):
                run = self.accepted_run(root/('q%d-run' % slot))
                q_audits[slot] = self.accepted_audit(
                    root/('q%d-audit' % slot), run)
            expansion = dict(source_radial_rows=31, expanded_radial_rows=48,
                source_primitive_count=1550, expanded_primitive_count=2400,
                expanded_spdf_primitive_count=1536,
                source_prefix_indices_sha256='c'*64)

            result = prepare.prepare_contracts(
                templates, root/'expanded', candidate, source_commit='b'*40,
                profiles=((4, 4, 3, 2, 0),), radial_rows=48,
                primitive_expansion=expansion,
                expanded_gamma_run=gamma_run,
                expanded_gamma_audit_result=gamma_audit,
                expanded_q_audit_results=q_audits)

            self.assertEqual(result['profiles'], [[4, 4, 3, 2, 0]])
            self.assertEqual(result['ao_per_C'], [45])
            self.assertEqual(result['radial_rows'], 48)
            for slot in range(8):
                contract = json.loads((root/'expanded'/('q%02d' % slot)/
                                       'CONTRACT.json').read_text(encoding='ascii'))
                self.assertEqual(contract['radial_rows'], 48)
                self.assertEqual(contract['primitive_expansion'], expansion)
                self.assertEqual(contract['expanded_gamma_run'], str(gamma_run.resolve()))
                self.assertEqual(contract['expanded_gamma_audit_result'],
                                 str(gamma_audit.resolve()))
                self.assertEqual(len(contract['candidate_profiles']), 1)
                if slot:
                    self.assertEqual(contract['baseline_audit_result'],
                                     str((templates/('q%02d' % slot)/
                                          'old-audit.json').resolve()))
                    self.assertEqual(contract['audit_result'],
                                     str(q_audits[slot].resolve()))
                for path, digest in contract['inputs'].items():
                    self.assertEqual(prepare.sha(path), digest)

    def test_expanded_fixed_rank_can_request_direct_q_energy_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            templates, candidates = self.fixtures(root)
            candidate = [candidates[1]]
            for slot in range(1, 8):
                old_audit = templates/('q%02d' % slot)/'old-audit.json'
                self.write_baseline_audit(old_audit, selected_iq=slot)
                contract_path = templates/('q%02d' % slot)/'RUNTIME_CONTRACT.json'
                contract = json.loads(contract_path.read_text(encoding='ascii'))
                contract['audit_result'] = str(old_audit.resolve())
                contract_path.write_text(json.dumps(contract), encoding='ascii')
            gamma_run = self.accepted_run(root/'gamma-run')
            gamma_audit = self.accepted_audit(
                root/'gamma-audit', gamma_run, include_run=False)
            q_audits = {}
            for slot in range(1, 8):
                run = self.accepted_run(root/('q%d-run' % slot))
                q_audits[slot] = self.accepted_audit(
                    root/('q%d-audit' % slot), run)
            expansion = dict(source_radial_rows=31, expanded_radial_rows=48,
                source_primitive_count=1550, expanded_primitive_count=2400,
                expanded_spdf_primitive_count=1536,
                source_prefix_indices_sha256='c'*64)

            result = prepare.prepare_contracts(
                templates, root/'gradient', candidate, source_commit='b'*40,
                profiles=((4, 4, 3, 2, 0),), radial_rows=48,
                primitive_expansion=expansion,
                expanded_gamma_run=gamma_run,
                expanded_gamma_audit_result=gamma_audit,
                expanded_q_audit_results=q_audits,
                scope='compressed_shared_radial_full_q_energy_gradient')

            self.assertEqual(
                result['scope'],
                'compressed_shared_radial_full_q_energy_gradient')
            for slot in range(8):
                contract = json.loads((root/'gradient'/('q%02d' % slot)/
                                       'CONTRACT.json').read_text())
                self.assertEqual(
                    contract['scope'],
                    'compressed_shared_radial_full_q_energy_gradient')

    def test_rejects_expanded_audit_as_finite_q_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            templates, candidates = self.fixtures(root)
            candidate = [candidates[1]]
            gamma_run = self.accepted_run(root/'gamma-run')
            gamma_audit = self.accepted_audit(
                root/'gamma-audit', gamma_run, include_run=False)
            q_audits = {}
            for slot in range(1, 8):
                run = self.accepted_run(root/('q%d-run' % slot))
                q_audits[slot] = self.accepted_audit(
                    root/('q%d-audit' % slot), run)
                contract_path = templates/('q%02d' % slot)/'RUNTIME_CONTRACT.json'
                contract = json.loads(contract_path.read_text(encoding='ascii'))
                contract['audit_result'] = str(q_audits[slot].resolve())
                contract_path.write_text(json.dumps(contract), encoding='ascii')
            expansion = dict(source_radial_rows=31, expanded_radial_rows=48,
                source_primitive_count=1550, expanded_primitive_count=2400,
                expanded_spdf_primitive_count=1536,
                source_prefix_indices_sha256='c'*64)

            with self.assertRaisesRegex(ValueError, 'baseline audit'):
                prepare.prepare_contracts(
                    templates, root/'gradient', candidate, source_commit='b'*40,
                    profiles=((4, 4, 3, 2, 0),), radial_rows=48,
                    primitive_expansion=expansion,
                    expanded_gamma_run=gamma_run,
                    expanded_gamma_audit_result=gamma_audit,
                    expanded_q_audit_results=q_audits,
                    scope='compressed_shared_radial_full_q_energy_gradient')

    @staticmethod
    def accepted_run(path):
        path.mkdir()
        for name in ('CONTRACT.json', 'STATUS', 'PROVENANCE'):
            (path/name).write_text(name+'\n', encoding='ascii')
        return path

    @staticmethod
    def accepted_audit(path, run, *, include_run=True):
        result = path/'result'
        result.mkdir(parents=True)
        audit = result/'RESULT.json'
        payload = dict(status='success')
        if include_run:
            payload['run'] = str(run.resolve())
        audit.write_text(json.dumps(payload), encoding='ascii')
        (result/'GAUGE_MAPS.npz').write_bytes(b'maps')
        (result/'STATUS').write_text('success\n', encoding='ascii')
        (path/'JOB_STATUS').write_text('success\n', encoding='ascii')
        (path/'PROVENANCE').write_text('success\n', encoding='ascii')
        return audit

    @staticmethod
    def write_baseline_audit(path, *, selected_iq):
        payload = dict(status='success',
            finite_q_spd_source_compatibility='pass', failure_reasons=[],
            selected_iq=selected_iq, regenerated_hamiltonian_read=False,
            regenerated_hamiltonian_admitted=False,
            full_source_primitive_count=1550, compared_primitive_count=558,
            per_k=[dict(pass_gate=True, source_ik=index, target_ik=index)
                   for index in range(1, 65)])
        path.write_text(json.dumps(payload), encoding='ascii')


if __name__ == '__main__':
    unittest.main()
