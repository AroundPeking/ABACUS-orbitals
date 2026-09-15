import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/prepare_c_jy_operator_restart.py'
SPEC = importlib.util.spec_from_file_location('remaining_sources', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pilot():
    return dict(status='success', finite_q_spd_source_compatibility='pass',
        failure_reasons=[], selected_iq=22, compared_primitive_count=558,
        full_source_primitive_count=1550, regenerated_hamiltonian_read=False,
        regenerated_hamiltonian_admitted=False, metric=dict(relative=1e-13),
        auxiliary_map_unitarity=1e-8,
        per_k=[dict(source_ik=k, target_ik=k, pass_gate=True,
            source=dict(relative=3e-10), overlap=dict(max_abs=1e-13),
            occupied=dict(relative=1e-14, unitarity=1e-14),
            source_eigenvalue_ha=dict(max_abs=1e-14),
            occupied_energy_commutator_ha=1e-14) for k in range(1, 65)])


def expanded_pilot():
    value = pilot()
    value['full_source_primitive_count'] = 2400
    value['primitive_expansion'] = dict(
        source_radial_rows=31,
        expanded_radial_rows=48,
        source_primitive_count=1550,
        expanded_primitive_count=2400,
        expanded_spdf_primitive_count=1536,
    )
    for row in value['per_k']:
        row['overlap'].update(max_abs=1.4e-10, relative=5e-13)
    return value


class RemainingSourcesTest(unittest.TestCase):
    def test_pilot_admits_only_canonical_remaining_sources_not_physics(self):
        result = MODULE.validate_finite_q_pilot(pilot())
        self.assertFalse(result['full_q_admitted'])
        self.assertFalse(result['regenerated_hamiltonian_admitted'])
        self.assertEqual(set(MODULE.FINITE_Q_SPECS), {22, 43, 6, 27, 23, 11, 55})
        self.assertEqual(sum(v[1] for v in MODULE.FINITE_Q_SPECS.values()), 63)
        for iq, (label, multiplicity, digest) in MODULE.FINITE_Q_SPECS.items():
            self.assertEqual(len(digest), 64)
            self.assertGreater(multiplicity, 0)
            self.assertTrue(label.startswith('q'))

    def test_expanded_pilot_locks_the_48_row_mother_contract(self):
        result = MODULE.validate_finite_q_pilot(expanded_pilot())
        self.assertEqual(result['full_source_primitive_count'], 2400)
        self.assertEqual(
            MODULE.q2_source_audit_sha256(2400),
            '4feab1c2c36eee1e5ac8ab297c72b96851c53fe4d6c52be480c4c4095f2ae271')
        with self.assertRaisesRegex(ValueError, 'expanded'):
            value = expanded_pilot()
            value['primitive_expansion']['expanded_radial_rows'] = 47
            MODULE.validate_finite_q_pilot(value)

    def test_expanded_overlap_requires_both_absolute_and_relative_gates(self):
        for absolute, relative in ((2.1e-10, 5e-13), (1.4e-10, 1.1e-11)):
            value = expanded_pilot()
            value['per_k'][0]['overlap'].update(
                max_abs=absolute, relative=relative)
            with self.subTest(absolute=absolute, relative=relative), \
                    self.assertRaisesRegex(ValueError, 'numerical'):
                MODULE.validate_finite_q_pilot(value)

    def test_numerical_and_routing_failures_remain_rejected(self):
        for mutation in ('rank', 'q', 'missing', 'route', 'failure', 'new_H', 'source', 'nan'):
            value = pilot()
            if mutation == 'rank': value['compared_primitive_count'] = 557
            if mutation == 'q': value['selected_iq'] = 43
            if mutation == 'missing': value['per_k'].pop()
            if mutation == 'route': value['per_k'][0]['target_ik'] = 2
            if mutation == 'failure': value['failure_reasons'] = ['k1']
            if mutation == 'new_H': value['regenerated_hamiltonian_read'] = True
            if mutation == 'source': value['per_k'][0]['source']['relative'] = 1.01e-6
            if mutation == 'nan': value['auxiliary_map_unitarity'] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                MODULE.validate_finite_q_pilot(value)

    def test_remaining_requires_pinned_pilot_before_reading_reference(self):
        for iq in (43, 6, 27, 23, 11, 55):
            with self.assertRaisesRegex(ValueError, 'pilot'):
                MODULE.source_extension_contract(iq, Path('/unused'), None, None)

    def test_no_arbitrary_q_or_pilot_without_frozen_auxiliary(self):
        with self.assertRaisesRegex(ValueError, 'canonical'):
            MODULE.source_extension_contract(2, Path('/unused'), None, None)
        with self.assertRaisesRegex(ValueError, 'frozen auxiliary'):
            MODULE.prepare(Path('/unused'), Path('/unused'), Path('/unused'), iq=43,
                           source_extension_pilot_audit=Path('/unused'))


if __name__ == '__main__':
    unittest.main()
