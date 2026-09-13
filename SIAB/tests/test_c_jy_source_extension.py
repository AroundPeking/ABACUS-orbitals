import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/prepare_c_jy_operator_restart.py'
SPEC = importlib.util.spec_from_file_location('source_extension', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def evidence():
    reuse = dict(status='success', original_gamma_operator_reuse_gate='pass',
        failure_reasons=[], compared_space='existing_spd_subblocks_only',
        regenerated_hamiltonian_admitted=False, missing_finite_q_sources_admitted=False,
        per_q=[dict(selected_iq=iq, pass_gate=True,
            per_k=[dict(source_ik=k, target_ik=k, pass_gate=True) for k in range(1, 65)])
            for iq in (1, 22, 43, 6, 27, 23, 11, 55)])
    row = dict(overlap=dict(max_abs=0.0), occupied=dict(relative=1e-14, unitarity=1e-14),
        occupied_energy_ry=dict(max_abs=1e-14), occupied_energy_commutator_ry=1e-14,
        source=dict(relative=1e-10), hamiltonian_ry=dict(max_abs=2.3e-6), pass_gate=False)
    gamma = dict(status='success', gamma_operator_compatibility_gate='hold',
        failure_reasons=['k1'], metric=dict(relative=1e-13), auxiliary_map_unitarity=1e-7,
        per_k=[dict(copy.deepcopy(row), ik=k) for k in range(1, 65)])
    return reuse, gamma


class SourceExtensionTest(unittest.TestCase):
    def test_export_is_only_one_pilot_with_locked_evidence(self):
        self.assertTrue(hasattr(MODULE, 'source_extension_contract'))
        with self.assertRaisesRegex(ValueError, 'pilot'):
            MODULE.source_extension_contract(43, Path('/unused'), None, None)
        with self.assertRaisesRegex(ValueError, 'locked'):
            MODULE.source_extension_contract(22, Path('/unused'), None, None)

    def validate(self, reuse, gamma):
        self.assertTrue(hasattr(MODULE, 'validate_source_extension_evidence'),
                        'missing explicit original-operator/new-source contract')
        return MODULE.validate_source_extension_evidence(reuse, gamma)

    def test_hamiltonian_failure_is_retained_not_waived_for_consumer(self):
        reuse, gamma = evidence()
        result = self.validate(reuse, gamma)
        self.assertEqual(result['hamiltonian_origin'], 'original_Gamma_at_target_k')
        self.assertFalse(result['regenerated_hamiltonian_admitted'])
        self.assertEqual(result['new_source_admission'], 'pending_finite_q_spd_check')
        self.assertEqual(gamma['gamma_operator_compatibility_gate'], 'hold')
        self.assertEqual(gamma['failure_reasons'], ['k1'])

    def test_all_q_and_k_bijections_required(self):
        for mutation in ('missing_q', 'duplicate_k', 'failed_k', 'failed_q'):
            reuse, gamma = evidence()
            if mutation == 'missing_q':
                reuse['per_q'].pop()
            elif mutation == 'duplicate_k':
                reuse['per_q'][1]['per_k'][2]['target_ik'] = 1
            elif mutation == 'failed_k':
                reuse['per_q'][1]['per_k'][2]['pass_gate'] = False
            else:
                reuse['per_q'][1]['pass_gate'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(reuse, gamma)

    def test_source_occupied_auxiliary_and_finite_gates_cannot_be_bypassed(self):
        for field in ('source', 'occupied', 'auxiliary', 'nan'):
            reuse, gamma = evidence()
            if field == 'source':
                gamma['per_k'][0]['source']['relative'] = 2e-6
            elif field == 'occupied':
                gamma['per_k'][0]['occupied']['unitarity'] = 2e-6
            elif field == 'auxiliary':
                gamma['auxiliary_map_unitarity'] = 1e-4
            else:
                gamma['per_k'][0]['source']['relative'] = float('nan')
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(reuse, gamma)


if __name__ == '__main__':
    unittest.main()
