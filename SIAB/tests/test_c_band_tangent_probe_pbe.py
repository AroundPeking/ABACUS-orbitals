"""Own-scope current-center PBE staging; no electronic-structure execution."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import test_c_optimized_pbe as fixture

sys.path.insert(0, str(fixture.MODULE.parent))


class BandProbePbeTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('check_c_band_tangent_probe_pbe'),
                             'current-center probe PBE adapter is missing')
        self.h=importlib.import_module('check_c_band_tangent_probe_pbe')
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        self.source=self.root/'accepted-preparation'; self.source.mkdir()
        self.center=self.root/'accepted-center'; slot=self.center/'result/pbe'; slot.mkdir(parents=True)
        values=fixture.endpoint._pure_pbe(fixture.endpoint._input_values(fixture.INPUT.encode()))
        inputs={'INPUT':'INPUT_PARAMETERS\n'+'\n'.join(k+' '+v for k,v in values.items())+'\n',
                'STRU':fixture.STRU,'KPT':'K_POINTS\n0\nGamma\n4 4 4 0 0 0\n',
                'C.upf':'frozen PBE pseudo\n','C_original.orb':'accepted center orbital\n'}
        for n,v in inputs.items(): (slot/n).write_text(v)
        prep=dict(prepared_input_sha256={n:fixture.digest(slot/n) for n in inputs},
                  candidate_log_relative_path='OUT.C_Q1/running_scf.log')
        (slot/'EC_GRADIENT_STEP_PBE_PREPARED.json').write_text(json.dumps(prep))
        self.energy=fixture.BASELINE+.018
        log=slot/prep['candidate_log_relative_path']; log.parent.mkdir(); log.write_text(fixture.log_text(self.energy))
        self.binary=self.root/'abacus'; self.binary.write_bytes(b'frozen runtime')
        self.mpi=self.root/'libpmi'; self.mpi.write_bytes(b'frozen pmi')
        pbe=dict(preparation_sha256=fixture.digest(slot/'EC_GRADIENT_STEP_PBE_PREPARED.json'),
            baseline_energy_ev=fixture.BASELINE,candidate_energy_ev=self.energy,candidate_log_sha256=fixture.digest(log),
            baseline_log_sha256='a'*64,abacus_binary=str(self.binary),abacus_sha256=fixture.digest(self.binary),
            mpi_library=str(self.mpi),mpi_sha256=fixture.digest(self.mpi))
        cresult=dict(actual_pbe=pbe,orbital_sha256=fixture.digest(slot/'C_original.orb'))
        (self.center/'result/RESULT.json').write_text(json.dumps(cresult))
        self.cs=fixture.digest(self.center/'result/RESULT.json')
        self.probe=self.source/'result/probes/probe_00'; self.probe.mkdir(parents=True)
        files={'COEFFICIENTS.txt':b'probe coefficients\n','C_3s3p2d_probe.orb':b'probe orbital\n'}
        for n,b in files.items(): (self.probe/n).write_bytes(b)
        pm=dict(status='prepared',scope='direction_calibration_probe',direction_name='T',signed_radius=-.001,
            center_result_sha256=self.cs,center_coefficient_sha256='b'*64,center_orbital_sha256=cresult['orbital_sha256'],
            directions_sha256='c'*64,coefficient_filename='COEFFICIENTS.txt',orbital_filename='C_3s3p2d_probe.orb',
            coefficient_sha256=fixture.digest(self.probe/'COEFFICIENTS.txt'),orbital_sha256=fixture.digest(self.probe/'C_3s3p2d_probe.orb'),
            cheap_gate=dict(gate=True),galerkin_energy='unmeasured',physical_release_gate='hold')
        (self.probe/'PROBE.json').write_text(json.dumps(pm)); files['PROBE.json']=(self.probe/'PROBE.json').read_bytes()
        self.loaded=dict(center=dict(stage=self.center,result=cresult,result_sha256=self.cs),
            result=dict(center_result_sha256=self.cs,directions_sha256='c'*64),
            probes=[dict(path=self.probe/'PROBE.json',manifest=pm,files=files)])
        patch=mock.patch.object(self.h,'load_accepted_preparation',return_value=self.loaded)
        self.admit=patch.start(); self.addCleanup(patch.stop)
        self.output=self.root/'campaign/probe_00'
        self.args=dict(preparation_stage=self.source,preparation_result_sha256='d'*64,
            preparation_acceptance_sha256='e'*64,preparation_deployment_sha256='f'*64,
            preparation_source_commit='1'*40,probe_index=0,output=self.output)

    def prepare(self,**kwargs): return self.h.prepare_probe_pbe(**dict(self.args,**kwargs))

    def collect(self,energy=None,log=None):
        p=self.output/'OUT.C_Q1/running_scf.log'; p.parent.mkdir(exist_ok=True)
        p.write_text(log or fixture.log_text(self.energy if energy is None else energy))
        return self.h.collect_probe_pbe(self.output,fixture.digest(self.output/self.h.PREPARATION))

    def test_own_scope_and_only_orbital_replacement(self):
        r=self.prepare()
        self.assertEqual(r['scope'],'current_ec_center_direction_calibration_probe')
        self.assertEqual(r['baseline_energy_ev'],fixture.BASELINE)
        self.assertEqual(r['center_energy_ev'],self.energy)
        self.assertEqual(r['signed_radius'],-.001)
        for n in ('INPUT','STRU','KPT','C.upf'):
            self.assertEqual((self.output/n).read_bytes(),(self.center/'result/pbe'/n).read_bytes())
        self.assertEqual((self.output/'C_original.orb').read_bytes(),b'probe orbital\n')
        self.assertFalse((self.output/'OUT.C_Q1/running_scf.log').exists())
        self.assertEqual(self.collect(fixture.BASELINE+.019)['pbe_gate'],'pass')
        with self.assertRaises(FileExistsError): self.collect()

    def test_original_bound_not_current_center_and_strict_scf(self):
        self.prepare()
        for log in (fixture.log_text().replace('#SCF IS CONVERGED#','SCF NOT CONVERGED'),
                    fixture.log_text().replace('= 44','= 43'),fixture.log_text(float('nan'))):
            with self.assertRaises(ValueError): self.collect(log=log)
        r=self.collect(fixture.BASELINE+.0201)
        self.assertEqual(r['pbe_gate'],'fail')
        self.assertLess(abs(r['delta_from_center_ev_per_c']),.01)
        self.assertEqual(r['physical_release_gate'],'hold')

    def test_rehash_and_mutation_do_not_bypass_frozen_evidence(self):
        self.prepare()
        for p in (self.output/'INPUT',self.probe/'C_3s3p2d_probe.orb',self.binary,self.mpi):
            original=p.read_bytes(); p.write_bytes(original+b'change')
            with self.subTest(path=p),self.assertRaises(ValueError): self.collect()
            p.write_bytes(original)
        p=self.output/self.h.PREPARATION; data=json.loads(p.read_text())
        for change in (dict(scope='direction_calibration_probe'),dict(tolerance_ev_per_c=.1),
                       dict(baseline_energy_ev=self.energy),dict(probe_index=True),dict(physical_release_gate='pass')):
            p.write_text(json.dumps(dict(data,**change)))
            with self.subTest(change=change),self.assertRaises(ValueError): self.collect()
        p.write_text(json.dumps(data))

    def test_unique_bounded_slots_and_external_admission(self):
        for index in (True,-1,8):
            with self.subTest(index=index),self.assertRaises(ValueError): self.prepare(probe_index=index)
        with self.assertRaises(ValueError): self.prepare(output=self.source/'new')
        self.admit.side_effect=ValueError('parent no longer accepted')
        with self.assertRaisesRegex(ValueError,'parent no longer accepted'): self.prepare()
        self.admit.side_effect=None
        self.prepare()
        with self.assertRaises(FileExistsError): self.prepare()
        with self.assertRaises(ValueError): self.prepare(output=self.root/'campaign/another')


if __name__=='__main__': unittest.main()
