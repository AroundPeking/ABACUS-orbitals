import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import common  # noqa: F401


class ResponseShortCycleTest(unittest.TestCase):
    def api(self):
        import run_c_response_short_cycle as api
        return api

    def test_actual_pbe_projection_is_orthogonal_and_descends(self):
        api = self.api()
        rates = [1., 2., 3., 4., 5., 6., 7., 8.]
        slope = [2., -1., 1., 3., 1., -2., 0., 1.]
        energies = [[-300.+2*a*s*api.PROBE_RADIUS for s in (-1, 1)] for a in slope]
        fit = api.fit_radial_pbe(rates, -300., energies)
        d = fit['radial_weights']
        self.assertAlmostEqual(sum(a*b for a,b in zip(slope,d)), 0., places=9)
        self.assertAlmostEqual(sum(x*x for x in d), 1., places=12)
        self.assertLess(fit['ec_slope_ha_per_cell'], 0.)
        self.assertEqual(fit['maximum_band_normal'], 'not_constrained')
        self.assertEqual(fit['finite_step_safety'], 'requires_actual_PBE')

    def test_parallel_gradient_or_invalid_samples_do_not_release(self):
        api = self.api()
        with self.assertRaisesRegex(ValueError, 'resolved'):
            api.fit_radial_pbe([1.]*8, -300., [[-300.-.002,-300.+.002]]*8)
        for energies in ([[0.,math.nan]]*8, [[0.,1.]]*7):
            with self.assertRaises(ValueError): api.fit_radial_pbe([1.]*8, -300., energies)

    def test_zero_pbe_slope_retains_full_descent(self):
        fit = self.api().fit_radial_pbe([1.]*8, -300., [[-300.,-300.]]*8)
        self.assertAlmostEqual(fit['retained_descent_fraction'], 1.)

    def test_sequential_backoff_checks_pbe_before_rpa(self):
        api = self.api(); measurements=[]; forwards=[]
        def measure(name, weights):
            measurements.append(name)
            energy = -300.+2*weights[0]
            if name == 'trial_0': energy = -299.9
            if name.startswith('trial_') and name != 'trial_0': energy = -300.
            return dict(candidate_energy_ev=energy, pbe_gate='pass' if abs(energy+300.)/2<=.01 else 'fail')
        before = dict(loss=1., rpa=dict(candidate_energy_ha=-.4,reference_energy_ha=-.5))
        def evaluate(name):
            forwards.append(name)
            return dict(loss=.9,rpa=dict(candidate_energy_ha=-.41,reference_energy_ha=-.5))
        out=api.calibrate_search([1.]*8,-300.,before,measure,evaluate)
        self.assertEqual(len(measurements),18)
        self.assertEqual(forwards,['trial_1'])
        self.assertEqual(out['candidate_gate'],'improved_frozen_body')
        self.assertEqual(out['selected_trial'],'trial_1')

    def test_nonimproving_energy_does_not_update_center(self):
        api=self.api(); calls=[]
        def measure(name,w):
            calls.append(name)
            return dict(candidate_energy_ev=-300.+2*w[0],pbe_gate='pass')
        before=dict(loss=1.,rpa=dict(candidate_energy_ha=-.4,reference_energy_ha=-.5))
        out=api.calibrate_search([1.]*8,-300.,before,measure,lambda n:before)
        self.assertEqual(out['candidate_gate'],'rejected_no_accepted_trial')
        self.assertEqual(len(calls),19)
        self.assertIsNone(out['selected_trial'])

    def test_reservation_is_global_for_exact_center_and_policy(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            api.reserve(root,root/'stage-a',{'center_sha256':'a'*64,'policy':'two'})
            with self.assertRaises(FileExistsError):
                api.reserve(root,root/'stage-b',{'center_sha256':'a'*64,'policy':'two'})

    def test_no_normalization_of_complete_q_energy(self):
        api=self.api()
        old=dict(loss=1.,rpa=dict(candidate_energy_ha=-.4,reference_energy_ha=-.5))
        for bad in (dict(loss=.9,rpa=dict(candidate_energy_ha=math.nan,reference_energy_ha=-.5)),
                    dict(loss=.9,rpa=dict(candidate_energy_ha=-.41,reference_energy_ha=-.51))):
            with self.assertRaises(ValueError): api.improves(old,bad)

    def test_trial_cheap_rejection_backtracks_without_stopping_calibration_result(self):
        from periodic_galerkin_fit import CandidateGuardError
        api=self.api(); forwards=[]
        def measure(name,w):
            if name=='trial_0': raise CandidateGuardError('near-edge band guard')
            return dict(candidate_energy_ev=-300.+2*w[0],pbe_gate='pass')
        old=dict(loss=1.,rpa=dict(candidate_energy_ha=-.4,reference_energy_ha=-.5))
        def forward(name):
            forwards.append(name)
            return dict(loss=.9,rpa=dict(candidate_energy_ha=-.41,reference_energy_ha=-.5))
        r=api.calibrate_search([1.]*8,-300.,old,measure,forward)
        self.assertEqual(forwards,['trial_1'])
        self.assertEqual(r['trials'][0]['gate'],'rejected_cheap_guard')

    def test_scf_records_original_energy_and_rejects_changed_inputs(self):
        api=self.api()
        import test_c_optimized_pbe as fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); slot=root/'one'; log='OUT.C_Q1/running_scf.log'
            def launch(*args,**kwargs):
                target=kwargs['cwd']/log; target.parent.mkdir()
                target.write_text(fixture.log_text(fixture.BASELINE+.019))
            with mock.patch.object(api.subprocess,'run',side_effect=launch) as process:
                r=api.run_scf(slot,{'INPUT':b'frozen'},log,['frozen-abacus'])
            self.assertEqual(r['pbe_gate'],'pass')
            self.assertAlmostEqual(r['energy_delta_ev_per_c'],.0095)
            self.assertEqual(process.call_count,1)
            with self.assertRaises(FileExistsError): api.run_scf(slot,{},log,[])
            def mutate(*args,**kwargs):
                launch(*args,**kwargs)
                (kwargs['cwd']/'INPUT').write_text('changed')
            with mock.patch.object(api.subprocess,'run',side_effect=mutate),self.assertRaises(ValueError):
                api.run_scf(root/'two',{'INPUT':b'frozen'},log,['frozen-abacus'])

    def test_unsafe_scf_input_or_log_path_rejects_before_mkdir(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); slot=root/'new'
            for inputs,log in (({'../INPUT':b'data'},'OUT/running_scf.log'),({'INPUT':b'data'},'../escaped')):
                with self.assertRaises(ValueError): api.run_scf(slot,inputs,log,[])
                self.assertFalse(slot.exists())

    def test_scheduler_wrapper_is_single_node_bounded_and_preflights_before_lock(self):
        api=self.api()
        script=Path(api.__file__).with_suffix('.slurm').read_text()
        self.assertIn('#SBATCH --nodes=1',script)
        self.assertIn('#SBATCH --time=01:00:00',script)
        self.assertLess(script.index('--preflight-only'),script.index('mkdir "$stage/EXECUTION_LOCK"'))
        for forbidden in ('--nodelist','I_MPI_FABRICS','sbatch','scancel'):
            self.assertNotIn(forbidden,script)


if __name__=='__main__': unittest.main()
