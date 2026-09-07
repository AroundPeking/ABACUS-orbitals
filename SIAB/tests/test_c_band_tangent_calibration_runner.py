import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import test_c_optimized_pbe as f
sys.path.insert(0,str(f.MODULE.parent))


class BandCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('run_c_band_tangent_pbe_calibration'))
        self.h=importlib.import_module('run_c_band_tangent_pbe_calibration')
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup); self.root=Path(t.name)
        self.root.joinpath('stage').mkdir()
        self.call_order=[]; self.fail_at=None
        self.options=dict(stage=self.root/'stage',preparation_stage=self.root/'parent',
            preparation_result_sha256='a'*64,preparation_acceptance_sha256='b'*64,
            preparation_deployment_sha256='c'*64,preparation_source_commit='d'*40,
            source_commit='e'*40,launcher=['not-a-real-executable'])
        self.loaded=dict(center=dict(stage=self.root/'accepted-center',result=dict(actual_pbe=dict(candidate_energy_ev=-309.84))),
            result=dict(directions_sha256='f'*64,center_result_sha256='1'*64))
        for name,value in (('load_accepted_preparation',dict(return_value=self.loaded)),
                           ('prepare_probe_pbe',dict(side_effect=self.prepare)),
                           ('collect_probe_pbe',dict(side_effect=self.collect)),
                           ('analyze_axes',dict(return_value=dict(consistency_gate='pass',axes={})))):
            p=mock.patch.object(self.h,name,**value); p.start(); self.addCleanup(p.stop)
        p=mock.patch.object(self.h.subprocess,'run',side_effect=lambda *a,**kw:self.call_order.append(('run',kw['cwd'].name)))
        self.launch=p.start(); self.addCleanup(p.stop)

    def prepare(self,output,probe_index,**kwargs):
        output.mkdir(parents=True); (output/self.h.PREPARATION).write_text('{}')
        self.call_order.append(('prepare',probe_index))
        return dict(probe_index=probe_index)

    def collect(self,path,digest):
        i=int(path.name[-2:]); self.call_order.append(('collect',i))
        r=dict(status='collected',direction_name='T' if i<4 else 'N',signed_radius=(-.001,.001,-.002,.002)[i%4],
            pbe_gate='fail' if i==self.fail_at else 'pass',scf_log_gate='pass',candidate_energy_ev=-309.84)
        (path/self.h.COLLECTION).write_text(json.dumps(r)); return r

    def test_all_inputs_precede_exactly_eight_scf_and_no_replay(self):
        result=self.h.run_calibration(**self.options)
        self.assertEqual(self.call_order[:8],[('prepare',i) for i in range(8)])
        self.assertEqual(self.launch.call_count,8)
        self.assertEqual(result['actual_scf_count'],8)
        self.assertEqual(result['calibration_gate'],'pass')
        for k in ('rpa_evaluations','backward_passes','cache_loads','optimizer_steps','physical_candidate_count'):
            self.assertEqual(result[k],0)
        self.assertEqual(result['physical_release_gate'],'hold')
        with self.assertRaises(FileExistsError): self.h.run_calibration(**self.options)

    def test_failed_true_pbe_guard_stops_remaining_scf(self):
        self.fail_at=1
        result=self.h.run_calibration(**self.options)
        self.assertEqual(self.launch.call_count,2)
        self.assertEqual(result['actual_scf_count'],2)
        self.assertEqual(result['calibration_gate'],'rejected_actual_pbe_probe')
        self.assertEqual(result['directions_analysis'],'unmeasured_incomplete_calibration')

    def test_changed_destination_cannot_repeat_accepted_probe_campaign(self):
        self.h.run_calibration(**self.options)
        another=self.root/'another-stage'; another.mkdir()
        with self.assertRaises(FileExistsError):
            self.h.run_calibration(**dict(self.options,stage=another))
        self.assertFalse((another/'result').exists())
        self.assertEqual(self.launch.call_count,8)

    def test_protected_paths_are_unchanged_when_rejected(self):
        for parent in (self.root/'parent',self.root/'accepted-center'):
            stage=parent/'nested'; stage.mkdir(parents=True)
            with self.subTest(parent=parent),self.assertRaises(ValueError):
                self.h.run_calibration(**dict(self.options,stage=stage))
            self.assertEqual(list(stage.iterdir()),[])
        self.assertEqual(self.launch.call_count,0)

    def test_subprocess_failure_consumes_campaign_without_running_remaining(self):
        self.launch.side_effect=self.h.subprocess.CalledProcessError(1,['not-real'])
        with self.assertRaises(self.h.subprocess.CalledProcessError): self.h.run_calibration(**self.options)
        self.assertEqual(self.launch.call_count,1)
        another=self.root/'second-stage'; another.mkdir()
        with self.assertRaises(FileExistsError): self.h.run_calibration(**dict(self.options,stage=another))
        self.assertEqual(self.launch.call_count,1)

    def test_wrapper_requires_nonwriting_preflight_before_lock(self):
        text=(f.MODULE.parent/'run_c_band_tangent_pbe_calibration.slurm').read_text()
        self.assertIn('--preflight-only',text)
        self.assertLess(text.index('--preflight-only'),text.index('mkdir "$stage/EXECUTION_LOCK"'))

    def test_deployed_source_is_protected_but_owned_source_subdirectory_is_allowed(self):
        stage=self.root/'stage'
        for source in (stage, self.root, stage/'result'):
            with self.subTest(source=source),self.assertRaises(ValueError):
                self.h.validate_stage(stage,self.options,self.loaded,source_directory=source)
            self.assertEqual(list(stage.iterdir()),[])
        self.h.validate_stage(stage,self.options,self.loaded,source_directory=stage/'source')
        self.h.validate_stage(stage,self.options,self.loaded,source_directory=self.root/'external-source')

    def test_new_wrapper_is_required_in_source_manifest(self):
        self.assertTrue(hasattr(self.h,'require_source_files'))
        with self.assertRaises(ValueError): self.h.require_source_files({'files':{}})
        prefix='SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/'
        names=('check_c_band_tangent_preparation.py','check_c_band_tangent_probe_pbe.py',
               'run_c_band_tangent_pbe_calibration.py','run_c_band_tangent_pbe_calibration.slurm')
        self.h.require_source_files({'files':{prefix+n:'a'*64 for n in names}})
        with self.assertRaises(ValueError):
            self.h.require_source_files({'files':{prefix+n:'a'*64 for n in names[:-1]}})

    def test_default_wrapper_has_frozen_layout_and_no_provider_override(self):
        text=(f.MODULE.parent/'run_c_band_tangent_pbe_calibration.slurm').read_text()
        for value in ('--partition=long','--nodes=1','--ntasks=4','--cpus-per-task=7','--no-requeue','calibration.time'):
            self.assertIn(value,text)
        for value in ('sbatch','mpirun','I_MPI_FABRICS','FI_PROVIDER','--nodelist'):
            self.assertNotIn(value,text)


if __name__=='__main__': unittest.main()
