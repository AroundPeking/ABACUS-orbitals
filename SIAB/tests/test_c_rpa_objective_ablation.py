import inspect
import unittest
from pathlib import Path
import sys

WORKFLOW=Path(__file__).resolve().parents[1]/'example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0,str(WORKFLOW))
import run_c_consecutive_radials as api
import check_c_accepted_combined_step as checker

STAGE='stage-c-pbe-inequality-b9494967'
RESULT='91a9a858e678999d368cad3094478b6761d59e52f2123d07f5942bd5e7c4279e'
VERIFY='6cfeddc7bb671f6853ed9f97967ea6392352a7167429268250ff67c3a2b3467c'


class CObjectiveAblationTest(unittest.TestCase):
    def configuration(self,mode):
        self.assertIn('ablation',inspect.signature(api.run_configuration).parameters)
        return api.run_configuration(dict(result_sha256=RESULT,verification_sha256=VERIFY,
            continuation_stage=str(api.ROOT/STAGE)), 'inequality', ablation=mode)

    def test_both_policies_pin_same_verified_parent_and_budget(self):
        a,b=[self.configuration(m) for m in ('energy_pbe','energy_free')]
        self.assertNotEqual(a['reservation_key'],b['reservation_key'])
        for c in (a,b):
            self.assertEqual(c['settings']['max_steps'],20)
            self.assertEqual(c['settings']['max_trials'],60)
            self.assertEqual(c['settings']['initial_radius'],.02)
            self.assertEqual(c['settings']['max_radius'],.10)
            self.assertFalse(c['settings']['require_nonincreasing_loss'])
        self.assertTrue(a['settings']['enforce_pbe'])
        self.assertFalse(b['settings']['enforce_pbe'])

    def test_ablation_requires_newest_parent_and_explicit_mode(self):
        self.configuration('energy_pbe')
        with self.assertRaises(ValueError):api.run_configuration(None,ablation='energy_free')
        with self.assertRaises(ValueError):self.configuration('unrecognized')
        wrong=dict(result_sha256=RESULT,verification_sha256=VERIFY,
                   continuation_stage=str(api.ROOT/STAGE))
        wrong['result_sha256']='0'*64
        with self.assertRaises(ValueError):api.run_configuration(wrong,'inequality',ablation='energy_pbe')

    def test_unmeasured_pbe_is_null_only_for_explicit_free_policy(self):
        self.assertIn('allow_unmeasured_pbe',inspect.signature(api.controller_record).parameters)
        r=dict(loss=.5,rpa=dict(candidate_energy_ha=-.4,reference_energy_ha=-.5))
        self.assertIsNone(api.controller_record(r,None,allow_unmeasured_pbe=True)['pbe'])
        with self.assertRaises(ValueError):api.controller_record(r,None)
        with self.assertRaises(ValueError):api.controller_record(r,float('nan'),allow_unmeasured_pbe=True)

    def test_adapter_routes_energy_only_no_per_step_scf_and_endpoint_measurement(self):
        self.assertIn('ablation',inspect.signature(api.run).parameters)
        source=inspect.getsource(api.run)
        self.assertIn('energy_only=not common_descent',source)
        self.assertIn('enforce_band_accuracy=not free_pbe',source)
        self.assertIn('return accuracy_guard(c,enforce_accuracy=False)',source)
        self.assertIn("pbe=None,actual_pbe=None",source)
        self.assertIn("measure_named(state['x'],'endpoint_pbe',force_scf=True)",source)
        self.assertIn("configuration['settings'].get('require_nonincreasing_loss',True)",source)
        self.assertIn("floor = 1e-12 if free_pbe",source)

    def test_free_band_screen_reports_accuracy_fail_without_disabling_stability(self):
        b=dict(gate=False,scf_pbe_gate='pending',scope='frozen_h_band_screen_not_scf_energy',
            k_weight_convention='ABACUS_spin_included_no_renormalization',k_weight_sum=2.,
            occupied_band_sum_limit_ev_per_atom=.01,target_band_change_limit_ev=.05,
            occupied_band_sum_change_ev_per_atom=.2,maximum_target_band_change_ev=1.,minimum_gap_ev=3.)
        checker._band_guard(b,enforce_accuracy=False)
        with self.assertRaises(ValueError):checker._band_guard(b)
        for key,value in (('minimum_gap_ev',0.),('minimum_gap_ev',float('nan')),
                          ('maximum_target_band_change_ev',float('inf')),
                          ('occupied_band_sum_change_ev_per_atom',float('nan'))):
            with self.assertRaises(ValueError):checker._band_guard(dict(b,**{key:value}),enforce_accuracy=False)

    def test_same_space_sos_is_the_existing_independent_spectral_oracle(self):
        oracle=(WORKFLOW.parents[2]/'tests/test_periodic_galerkin_sos_equivalence.py').read_text()
        self.assertIn('test_full_and_reduced_complex_spaces_match_all_band_spectral_sum',oracle)
        self.assertIn('test_rediagonalizing_full_h_changes_the_fixed_occupied_problem',oracle)


if __name__=='__main__':unittest.main()
