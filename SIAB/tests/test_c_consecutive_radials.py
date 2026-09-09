import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
import hashlib
import json
import tempfile
import os
import subprocess
from contextlib import contextmanager

OLD_RESULT = "4dbed5d6f07e72487677565f06f384e53f461c68c2f098ee8283395951bae511"
OLD_VERIFY = "4d21d1cf673b35774ceb7b128d9b30cc2ddd4fe9423be2f9391d1e4edf1b37c4"
COMMON_STAGE = "stage-c-common-descent-8acbf7b9"
COMMON_RESULT = "dc3e2c5041cd5ba9ad53072c03ae4c29d4a90caa40fbdf82fbb98d9e6063bbc3"
COMMON_VERIFY = "45550e3f6d0ff62494fb54482d3a00afabb624c834fc1a028bc4721654ae16d3"

WORKFLOW = Path(__file__).resolve().parents[1] / "example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow"
sys.path.insert(0, str(WORKFLOW))


class ConsecutiveAdmissionTest(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("run_c_consecutive_radials"))
        import run_c_consecutive_radials as api
        return api

    def fixture(self):
        return dict(radius=.04, gate="diagnostic_measured",
                    actual_pbe=dict(pbe_gate="pass", baseline_energy_ev=-309.8590820440637,
                                    energy_delta_ev_per_c=.009745847520548523),
                    proposal=dict(nu=[3,3,2,0,0], fixed_nu=[0]*5, ao_per_C=22),
                    rpa=dict(candidate=dict(loss=.07, rpa=dict(candidate_energy_ha=-.43,
                                                                  reference_energy_ha=-.514))))

    def test_only_verified_feasible_same_size_endpoint(self):
        self.api().validate_parent_trial(self.fixture())
        for path, value in ((("actual_pbe","pbe_gate"),"fail"),
                            (("actual_pbe","energy_delta_ev_per_c"),.01001),
                            (("actual_pbe","baseline_energy_ev"),-309.85),
                            (("proposal","fixed_nu"),[2,2,1,0,0]),
                            (("proposal","nu"),[3,3,3,0,0])):
            trial = copy.deepcopy(self.fixture())
            trial[path[0]][path[1]] = value
            with self.assertRaises(ValueError):
                self.api().validate_parent_trial(trial)

    def test_reference_objective_not_unbounded_minimum(self):
        api = self.api()
        r = dict(loss=.1, rpa=dict(candidate_energy_ha=-.5, reference_energy_ha=-.4))
        self.assertAlmostEqual(api.controller_record(r,.009)["objective"], .1)
        with self.assertRaises(ValueError):
            api.controller_record(r,float("nan"))

    def test_only_explicit_candidate_nonconvergence_is_recoverable(self):
        api=self.api()
        error=ValueError("SCF log is not unambiguously converged")
        self.assertTrue(api.candidate_nonconvergence(error,b"#SCF IS NOT CONVERGED#"))
        for text in (b"",b"#SCF IS CONVERGED#",b"#SCF IS CONVERGED# SCF NOT CONVERGED"):
            self.assertFalse(api.candidate_nonconvergence(error,text))
        self.assertFalse(api.candidate_nonconvergence(ValueError("input hash mismatch"),
                                                       b"SCF NOT CONVERGED"))

    def test_continuation_preserves_layout_pbe_origin_and_finite_ambient_model(self):
        api=self.api()
        result=dict(status="success",nu=[3,3,2,0,0],fixed_nu=[0]*5,ao_per_C=22,
                    full_constrained_stationarity="not_established",physical_release_gate="hold",
                    optimization=dict(pbe_gradient=[.1]*248,record=dict(pbe=.0087)))
        actual=dict(pbe_gate="pass",baseline_energy_ev=-309.8590820440637,
                    energy_delta_ev_per_c=.0087)
        api.validate_continuation_payload(result,actual)
        for key,value in (("pbe_gradient",[.1]*247),("pbe_gradient",[float("nan")]*248),
                          ("record",dict(pbe=.011))):
            invalid=copy.deepcopy(result);invalid["optimization"][key]=value
            with self.assertRaises(ValueError):api.validate_continuation_payload(invalid,actual)
        for key,value in (("baseline_energy_ev",-309.86),("pbe_gate","fail"),
                          ("energy_delta_ev_per_c",.007)):
            invalid=dict(actual);invalid[key]=value
            with self.assertRaises(ValueError):api.validate_continuation_payload(result,invalid)
        invalid=copy.deepcopy(result);invalid["fixed_nu"]=[2,2,1,0,0]
        with self.assertRaises(ValueError):api.validate_continuation_payload(invalid,actual)

    def test_parent_uses_isolated_original_declared_source_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp).resolve()
            name='SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/refresh_c_combined_step_gradient.py'
            validator=stage/'source'/name;validator.parent.mkdir(parents=True)
            validator.write_bytes(b'# frozen validator fixture\n')
            manifest=dict(source_commit='a'*40,source_directory=str(stage/'source'),
                          files={name:hashlib.sha256(validator.read_bytes()).hexdigest()})
            raw=json.dumps(manifest).encode();(stage/'DEPLOYMENT.json').write_bytes(raw)
            digest=hashlib.sha256(raw).hexdigest()
            with mock.patch('subprocess.check_output',return_value=raw) as child:
                self.assertEqual(self.api().verify_parent_source(stage,digest,'a'*40),manifest)
            command=child.call_args[0][0]
            self.assertEqual(command[1:3],['-I','-B'])
            self.assertEqual(command[-3:],[str(stage),digest,'a'*40])
            self.assertEqual(child.call_args[1]['env']['PYTHONDONTWRITEBYTECODE'],'1')
            validator.write_bytes(b'# changed\n')
            with mock.patch('subprocess.check_output') as child:
                with self.assertRaises(ValueError):self.api().verify_parent_source(stage,digest,'a'*40)
                child.assert_not_called()

    def test_continuation_log_and_inputs_are_one_scf_record(self):
        prior=Path('/prior')
        actual=dict(candidate_log_path='/prior/result/trial_021/pbe/OUT.C/running_scf.log')
        self.assertEqual(self.api().continuation_scf_root(prior,actual,dict(actual)),
                         Path('/prior/result/trial_021/pbe'))
        mismatch=dict(candidate_log_path='/prior/result/trial_022/pbe/OUT.C/running_scf.log')
        with self.assertRaises(ValueError):self.api().continuation_scf_root(prior,actual,mismatch)
        outside=dict(candidate_log_path='/other/result/trial_021/pbe/OUT.C/running_scf.log')
        with self.assertRaises(ValueError):self.api().continuation_scf_root(prior,outside,outside)

    def test_explicit_parent_profiles_and_mode_are_pinned(self):
        api=self.api()
        self.assertTrue(callable(getattr(api,'continuation_parent',None)))
        prior,spec=api.continuation_parent(OLD_RESULT,OLD_VERIFY)
        self.assertEqual(prior,api.ROOT/'stage-c-consecutive-radials-7d8eef3c')
        self.assertEqual(spec['job_id'],'21913349')
        prior,spec=api.continuation_parent(COMMON_RESULT,COMMON_VERIFY,COMMON_STAGE,'inequality')
        self.assertEqual(prior,api.ROOT/COMMON_STAGE)
        self.assertEqual(spec['job_id'],'21914609')
        self.assertEqual(spec['scope'],'common_descent_eight_radial_actual_PBE_constrained_frozen_body')
        for result,verify,stage,mode in ((COMMON_RESULT,COMMON_VERIFY,None,'inequality'),
                                        (OLD_RESULT,OLD_VERIFY,'stage-c-consecutive-radials-7d8eef3c','inequality'),
                                        ('0'*64,COMMON_VERIFY,COMMON_STAGE,'inequality'),
                                        (COMMON_RESULT,'0'*64,COMMON_STAGE,'inequality'),
                                        (COMMON_RESULT,COMMON_VERIFY,'/tmp/'+COMMON_STAGE,'inequality'),
                                        (COMMON_RESULT,COMMON_VERIFY,'../'+COMMON_STAGE,'inequality'),
                                        (COMMON_RESULT,COMMON_VERIFY,COMMON_STAGE,'unknown')):
            with self.subTest(stage=stage,mode=mode), self.assertRaises(ValueError):
                api.continuation_parent(result,verify,stage,mode)

    def test_parent_stage_symlink_cannot_escape_root(self):
        api=self.api()
        self.assertTrue(callable(getattr(api,'continuation_parent',None)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';root.mkdir()
            outside=Path(tmp)/'outside';outside.mkdir()
            (root/COMMON_STAGE).symlink_to(outside,target_is_directory=True)
            with mock.patch.object(api,'ROOT',root), self.assertRaises(ValueError):
                api.continuation_parent(COMMON_RESULT,COMMON_VERIFY,COMMON_STAGE,'inequality')

    def test_inequality_budget_and_reservation_leave_defaults_unchanged(self):
        api=self.api()
        self.assertTrue(callable(getattr(api,'run_configuration',None)))
        base=api.run_configuration(None)
        self.assertEqual(base['settings'],dict(max_steps=20,max_trials=80,initial_radius=.02,max_radius=.04))
        old=dict(result_sha256=OLD_RESULT,verification_sha256=OLD_VERIFY,
                 continuation_stage=str(api.ROOT/'stage-c-consecutive-radials-7d8eef3c'))
        legacy=api.run_configuration(old)
        self.assertEqual(legacy['settings'],dict(max_steps=12,max_trials=40,initial_radius=.001,max_radius=.01))
        self.assertEqual(legacy['reservation_key'],OLD_RESULT[:16]+'-common-v1')
        new=dict(result_sha256=COMMON_RESULT,verification_sha256=COMMON_VERIFY,
                 continuation_stage=str(api.ROOT/COMMON_STAGE))
        config=api.run_configuration(new,'inequality')
        self.assertEqual(config['settings'],dict(max_steps=6,max_trials=24,initial_radius=.001,max_radius=.01,
                                                pbe_proposal='inequality'))
        self.assertEqual(config['scope'],'inequality_common_descent_eight_radial_actual_PBE_constrained_frozen_body')
        self.assertEqual(config['reservation_key'],COMMON_RESULT[:16]+'-inequality-v1')
        with self.assertRaises(ValueError):api.run_configuration(None,'inequality')
        with self.assertRaises(ValueError):api.run_configuration(old,'inequality')

    def test_checkpoint_final_best_and_terminal_gradient_share_center(self):
        api=self.api()
        self.assertTrue(callable(getattr(api,'validate_continuation_checkpoint',None)))
        record=dict(objective=.08,loss=.07,pbe=.008)
        state=dict(x=[.1]*248,record=record,best=dict(x=[.1]*248,record=copy.deepcopy(record)),
                   counts=dict(accepted_steps=1))
        result=dict(optimization=state,coefficient_sha256='a'*64,
                    gradients=[dict(coefficient_sha256='a'*64)])
        checkpoint=dict(x=[.1]*248,record=copy.deepcopy(record),counts=dict(accepted_steps=1))
        api.validate_continuation_checkpoint(result,checkpoint,1)
        mutations=(lambda r,c:r['optimization']['best']['x'].__setitem__(0,.2),
                   lambda r,c:c['x'].__setitem__(0,.2),
                   lambda r,c:c['record'].update(loss=.08),
                   lambda r,c:r['gradients'][-1].update(coefficient_sha256='b'*64),
                   lambda r,c:r['optimization']['counts'].update(accepted_steps=2))
        for mutate in mutations:
            r,c=copy.deepcopy((result,checkpoint));mutate(r,c)
            with self.assertRaises(ValueError):api.validate_continuation_checkpoint(r,c,1)

    def test_cli_routes_explicit_stage_and_mode_and_keeps_default(self):
        api=self.api()
        base=['driver','--stage','/new-stage','--source-commit','a'*40,'--deployment-sha256','b'*64]
        class StopAdmission(Exception):pass
        with mock.patch.object(sys,'argv',base), mock.patch.object(api,'admit_parent',side_effect=StopAdmission) as old:
            with self.assertRaises(StopAdmission):api.main()
            old.assert_called_once_with()
        flags=['--continuation-stage',str(api.ROOT/COMMON_STAGE),'--pbe-proposal','inequality',
               '--continuation-result-sha256',COMMON_RESULT,'--continuation-verification-sha256',COMMON_VERIFY]
        with mock.patch.object(sys,'argv',base+flags), \
                mock.patch.object(api,'admit_continuation',side_effect=StopAdmission) as continuation:
            with self.assertRaises(StopAdmission):api.main()
            continuation.assert_called_once_with(COMMON_RESULT,COMMON_VERIFY,
                continuation_stage=str(api.ROOT/COMMON_STAGE),pbe_proposal='inequality')

    def test_slurm_forwards_mode_stage_and_both_pins_without_executing_jobs(self):
        shell=(WORKFLOW/'run_c_consecutive_radials.slurm').read_text()
        block=shell[shell.index('args=('):shell.index('"$root/optimizer-env-torch1121cpu/bin/python"')]
        command='stage=$C_CONSECUTIVE_STAGE\n'+block+'\nprintf "%s\\n" "${args[@]}"\n'
        environment=dict(C_CONSECUTIVE_STAGE='/new stage',C_CONSECUTIVE_SOURCE_COMMIT='a'*40,
                         C_CONSECUTIVE_DEPLOYMENT_SHA256='b'*64)
        default=subprocess.run(['bash','-eu','-c',command],env=environment,capture_output=True,text=True,check=True)
        self.assertIn('--pbe-proposal\ntangent\n',default.stdout)
        environment.update(C_PBE_PROPOSAL='inequality',C_CONTINUATION_STAGE='/root/'+COMMON_STAGE,
                           C_CONTINUATION_RESULT_SHA256=COMMON_RESULT,C_CONTINUATION_VERIFICATION_SHA256=COMMON_VERIFY)
        explicit=subprocess.run(['bash','-eu','-c',command],env=environment,capture_output=True,text=True,check=True)
        self.assertIn('--continuation-stage\n/root/'+COMMON_STAGE+'\n',explicit.stdout)
        self.assertIn('--pbe-proposal\ninequality\n',explicit.stdout)
        self.assertIn('--continuation-result-sha256\n'+COMMON_RESULT+'\n',explicit.stdout)
        environment.pop('C_CONTINUATION_VERIFICATION_SHA256')
        refused=subprocess.run(['bash','-eu','-c',command],env=environment,capture_output=True,text=True)
        self.assertNotEqual(refused.returncode,0)

    @contextmanager
    def continuation_files(self, mutate=None):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();prior=root/COMMON_STAGE
            result_dir=prior/'result';checkpoint_dir=result_dir/'checkpoint_001'
            gradient_dir=result_dir/'gradient_001';pbe_dir=result_dir/'trial_000/pbe'
            def sha(data):return hashlib.sha256(data).hexdigest()
            def write(path,data):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data)
            def dump(path,data):write(path,json.dumps(data).encode())
            coefficient=b'synthetic coefficient bytes\n';orbital=b'synthetic orbital bytes\n'
            for slot in (result_dir/'FINAL',checkpoint_dir,gradient_dir):
                write(slot/'COEFFICIENTS.txt',coefficient)
                write(slot/'C_3s3p2d.orb',orbital)
            inputs={'INPUT':b'frozen pure PBE input\n'}
            for name,data in dict(inputs,**{'C.orb':orbital}).items():write(pbe_dir/name,data)
            log=pbe_dir/'OUT.C/running_scf.log';write(log,b'#SCF IS CONVERGED#\n')
            energy=api.ORIGINAL_ENERGY_EV+.016
            pbe=(energy-api.ORIGINAL_ENERGY_EV)/2
            actual=dict(pbe_gate='pass',baseline_energy_ev=api.ORIGINAL_ENERGY_EV,
                        energy_delta_ev_per_c=pbe,candidate_energy_ev=energy,
                        candidate_log_path=str(log),candidate_log_sha256=sha(log.read_bytes()),
                        input_sha256={'INPUT':sha(inputs['INPUT']),'C.orb':sha(orbital)})
            candidate=dict(loss=.07,rpa=dict(candidate_energy_ha=-.43,reference_energy_ha=-.514))
            record=api.controller_record(candidate,pbe)
            checkpoint=dict(x=[.1]*248,record=copy.deepcopy(record),counts=dict(accepted_steps=1),
                            actual_pbe=actual,measurement=dict(actual_pbe=actual),candidate=candidate,
                            coefficient_sha256=sha(coefficient),orbital_sha256=sha(orbital))
            gradient=dict(coefficient_sha256=sha(coefficient),loss_gradient=dict(horizontal_gradient_norm=.1))
            spec=dict(api.CONTINUATION_PARENTS[COMMON_STAGE])
            result=dict(status='success',scope=api.COMMON_SCOPE,source_commit=spec['source_commit'],
                        deployment_sha256='d'*64,common_descent=True,calibration_scf_count=0,cache_loads=1,
                        nu=[3,3,2,0,0],fixed_nu=[0]*5,ao_per_C=22,physical_release_gate='hold',
                        full_constrained_stationarity='not_established',freeze_sha256='f'*64,
                        active_cache_index_sha256='e'*64,coefficient_sha256=sha(coefficient),orbital_sha256=sha(orbital),
                        optimization=dict(x=[.1]*248,record=record,best=dict(x=[.1]*248,record=copy.deepcopy(record)),
                                          counts=dict(accepted_steps=1),pbe_gradient=[.1]*248),
                        final_candidate=candidate,gradients=[dict(index=1,coefficient_sha256=sha(coefficient))])
            audit=dict(status=spec['verification_status'],job_id=spec['job_id'],
                       scheduler=spec['job_id']+'|COMPLETED|0:0|\n')
            provenance=dict(status='success',job_id=spec['job_id'],source_commit=spec['source_commit'],
                            deployment_sha256=result['deployment_sha256'])
            if mutate:mutate(result,checkpoint,actual,audit,provenance,gradient)
            dump(gradient_dir/'GRADIENT.json',gradient)
            result['gradients'][-1]['gradient_sha256']=sha((gradient_dir/'GRADIENT.json').read_bytes())
            dump(result_dir/'RESULT.json',result)
            result_sha=sha((result_dir/'RESULT.json').read_bytes())
            audit['result_sha256']=provenance['result_sha256']=result_sha
            dump(prior/'VERIFICATION.json',audit);dump(prior/'PROVENANCE.json',provenance)
            dump(checkpoint_dir/'CHECKPOINT.json',checkpoint);write(prior/'STATUS',b'success\n')
            verify_sha=sha((prior/'VERIFICATION.json').read_bytes())
            spec.update(result_sha256=result_sha,verification_sha256=verify_sha)
            cycle=mock.Mock()
            def check(path,digest):
                if sha(Path(path).read_bytes())!=digest:raise ValueError('file hash mismatch')
            cycle.admission.common._check_file.side_effect=check
            cycle.endpoint._sha256.side_effect=sha
            cycle._strict_log.return_value=dict(energy_ev=energy)
            cycle.frozen_pbe_bundle.return_value=(inputs,'C.orb','running_scf.log')
            old=dict(result=dict(freeze_sha256='f'*64,active_cache_index_sha256='e'*64),
                     occupied_capture_floor=.9999,quarter=dict(training_weights={}))
            with mock.patch.object(api,'ROOT',root), mock.patch.dict(api.CONTINUATION_PARENTS,{COMMON_STAGE:spec}), \
                    mock.patch.object(api,'admit_parent',return_value=(cycle,old,dict(rpa=dict(candidate=candidate)))), \
                    mock.patch.object(api,'verify_parent_source') as source:
                yield api,dict(result_sha=result_sha,verification_sha=verify_sha,
                               continuation_stage=str(prior),pbe_proposal='inequality'),cycle,source,result

    def test_complete_common_parent_admission_reuses_final_secant_and_source_checks(self):
        with self.continuation_files() as (api,kwargs,cycle,source,result):
            _,_,parent,restart=api.admit_continuation(**kwargs)
            self.assertEqual(restart['pbe_gradient'],result['optimization']['pbe_gradient'])
            self.assertEqual(restart['continuation_stage'],kwargs['continuation_stage'])
            self.assertEqual(restart['coefficient_path'],kwargs['continuation_stage']+'/result/FINAL/COEFFICIENTS.txt')
            self.assertEqual(parent['proposal']['coefficient_sha256'],result['coefficient_sha256'])
            source.assert_called_once_with(Path(kwargs['continuation_stage']),result['deployment_sha256'],result['source_commit'])
            cycle.admission.refresh.accepted._record.assert_called_once()
            cycle.admission.refresh.accepted._same_grid_reference.assert_called_once()
            checked={str(c[0][0]) for c in cycle.admission.common._check_file.call_args_list}
            for suffix in ('/result/gradient_001/GRADIENT.json','/result/FINAL/COEFFICIENTS.txt',
                           '/result/checkpoint_001/COEFFICIENTS.txt','/result/trial_000/pbe/INPUT'):
                self.assertIn(kwargs['continuation_stage']+suffix,checked)

    def test_common_parent_rejects_wrong_identity_cache_inputs_and_missing_loss(self):
        mutations=(lambda r,c,a,v,p,g:r.update(scope='wrong_parent_scope'),
                   lambda r,c,a,v,p,g:r.update(common_descent=False),
                   lambda r,c,a,v,p,g:r.update(calibration_scf_count=16),
                   lambda r,c,a,v,p,g:r.update(cache_loads=2),
                   lambda r,c,a,v,p,g:r.update(freeze_sha256='a'*64),
                   lambda r,c,a,v,p,g:v.update(status='verified_consecutive_frozen_body_not_SOS_qavg_or_GW'),
                   lambda r,c,a,v,p,g:v.update(scheduler='21914609|RUNNING|0:0|\n'),
                   lambda r,c,a,v,p,g:p.update(source_commit='a'*40),
                   lambda r,c,a,v,p,g:a['input_sha256'].update(INPUT='b'*64),
                   lambda r,c,a,v,p,g:g.pop('loss_gradient'))
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i), self.continuation_files(mutate) as (api,kwargs,_,_,_):
                with self.assertRaises(ValueError):api.admit_continuation(**kwargs)

    def test_main_reserves_new_mode_and_rechecks_the_selected_parent(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();stage=root/'new-stage';stage.mkdir()
            restart=dict(result_sha256=COMMON_RESULT,verification_sha256=COMMON_VERIFY,
                         continuation_stage=str(root/COMMON_STAGE))
            runtime=dict(abacus_binary='/frozen/abacus',abacus_sha256='a'*64,mpi_library='/frozen/pmi',mpi_sha256='b'*64)
            cycle=mock.Mock();cycle.admission.refresh._WORKFLOW='wf/'
            cycle.admission.common.verify_source.return_value=dict(
                files={'wf/run_c_consecutive_radials.py':'a','wf/run_c_consecutive_radials.slurm':'b'})
            def write(path,data):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(data))
            cycle.endpoint._write_json.side_effect=write
            cycle.endpoint._sha256.side_effect=lambda raw:hashlib.sha256(raw).hexdigest()
            args=['driver','--stage',str(stage),'--source-commit','c'*40,'--deployment-sha256','d'*64,
                  '--continuation-stage',restart['continuation_stage'],'--pbe-proposal','inequality',
                  '--continuation-result-sha256',COMMON_RESULT,'--continuation-verification-sha256',COMMON_VERIFY]
            environment=dict(SLURM_JOB_NUM_NODES='1',SLURM_NTASKS='4',OMP_NUM_THREADS='7',
                             I_MPI_PMI_LIBRARY=runtime['mpi_library'],SLURM_JOB_ID='synthetic')
            with mock.patch.object(api,'ROOT',root), mock.patch.object(sys,'argv',args), \
                    mock.patch.dict(os.environ,environment), \
                    mock.patch.object(api,'admit_continuation',return_value=(cycle,dict(result=dict(actual_pbe=runtime)),{},restart)) as admit, \
                    mock.patch.object(api,'run',return_value=dict(status='success')) as run:
                api.main()
                self.assertEqual(admit.call_count,2)
                for call in admit.call_args_list:
                    self.assertEqual(call[1],dict(continuation_stage=restart['continuation_stage'],pbe_proposal='inequality'))
                self.assertEqual(run.call_args[1],dict(continuation=restart,pbe_proposal='inequality'))
                reservation=json.loads((root/('consecutive-radials-'+COMMON_RESULT[:16]+'-inequality-v1')/'RESERVATION.json').read_text())
                self.assertEqual(reservation['maximum_accepted_steps'],6)
                self.assertEqual(reservation['maximum_trials'],24)
                self.assertEqual(reservation['calibration_scf_count'],0)
                self.assertEqual(reservation['continuation_stage'],restart['continuation_stage'])
                self.assertEqual(reservation['pbe_proposal'],'inequality')
                with self.assertRaises(FileExistsError):api.main()
                self.assertEqual(run.call_count,1)


if __name__ == "__main__":
    unittest.main()
