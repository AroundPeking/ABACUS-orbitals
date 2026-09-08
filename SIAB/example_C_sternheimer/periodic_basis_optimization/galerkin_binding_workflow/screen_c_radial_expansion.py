"""Three single-radial seed tests on the frozen C SPD cache; no SCF/SOS/GW."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

import run_c_response_short_cycle as cycle

ROOT=cycle.ROOT
PARENT=ROOT/'stage-c-response-short-cycle-18af5697'
RESULT_SHA='eec7d67694c61fe66eb60813c1119dc909b6ab7d0b9d657dfcf60aeffc4dd0e6'
VERIFY_SHA='6b9b2928271e91ad8b659a26ab6a3795b731e849fa01b38da5c21af37d871cb7'
SCOPE='three_single_radial_seed_frozen_body_screen'


def admit_failed_cache_attempt():
    failed=ROOT/'stage-c-radial-expansion-31623bc4'
    screen_error=failed/'slurm-21911787.err'
    cycle.admission.common._check_file(screen_error,
        'e5a37fe02141b1d3af3467eb4ccb363f9f48875c1d6d275e3e3fc183f0dd9c49')
    if ((failed/'STATUS').read_text()!='failed\n' or (failed/'result/RESULT.json').exists()
            or any((failed/'result').glob('*/SCREEN.json'))):
        raise ValueError('only the exact pre-candidate cache failure may recover')
    rows=subprocess.check_output(['sacct','-n','-P','-j','21911787','--format=JobID,State,ExitCode'],
        universal_newlines=True)
    states={r.split('|')[0]:r.split('|')[1:3] for r in rows.splitlines() if r.strip()}
    if states!={'21911787':['FAILED','1:0'],'21911787.batch':['FAILED','1:0'],
                '21911787.extern':['COMPLETED','0:0']}:
        raise ValueError('previous cache attempt must be terminal, never duplicate a live job')
    return dict(job_id='21911787',stage=str(failed),failure='reference_cache_dataset_identity',
        physical_candidates=0,scheduler=rows)


def admit_parent():
    check=cycle.admission.common._check_file
    check(PARENT/'result/RESULT.json',RESULT_SHA); check(PARENT/'VERIFICATION.json',VERIFY_SHA)
    result=json.loads((PARENT/'result/RESULT.json').read_text())
    audit=json.loads((PARENT/'VERIFICATION.json').read_text())
    if (audit['status']!='verified_frozen_body_cycle_not_SOS_or_GW' or audit['result_sha256']!=RESULT_SHA
            or result['status']!='success' or result['selected_trial']!='trial_0'
            or result['candidate_gate']!='improved_frozen_body' or (PARENT/'STATUS').read_text()!='success\n'):
        raise ValueError('independent accepted parent required')
    for row in audit['scheduler'].splitlines():
        if row.split('|')[1:3]!=['COMPLETED','0:0']: raise ValueError('parent scheduler gate failed')
    provenance=json.loads((PARENT/'PROVENANCE.json').read_text())
    if provenance['status']!='success' or provenance['result_sha256']!=RESULT_SHA:
        raise ValueError('parent provenance mismatch')
    trial=result['trials'][0]
    if trial['actual_pbe']['pbe_gate']!='pass': raise ValueError('accepted actual PBE required')
    slot=PARENT/'result/trial_0'
    for name,key in (('COEFFICIENTS.txt','coefficient_sha256'),('C_3s3p2d.orb','orbital_sha256'),
                     ('CANDIDATE.json','candidate_sha256')):
        check(slot/name,trial['actual_pbe'][key])
    old,_=cycle.admission.admit_gradient(cycle.GRADIENT_STAGE,cycle.GRADIENT_SHA,cycle.ACCEPTANCE_SHA)
    if any(result[k]!=old['result'][k] for k in ('freeze_sha256','active_cache_index_sha256')):
        raise ValueError('parent reference/cache changed')
    return result,old


def run(stage,source):
    import torch
    import numpy as np
    from periodic_galerkin_expansion import append_smooth_complement,prepare_expansion_evaluation
    from periodic_galerkin_reduction import reprofile_active_primitives
    from periodic_galerkin_basis import write_periodic_optimizer_coefficients
    from periodic_galerkin_pbe_guard import prepare_frozen_band_guard
    from periodic_galerkin_fit import CandidateGuardError,_global_rpa_loss
    from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference
    from run_c_combined_step import screen_candidate
    from export_periodic_orbitals import write_abacus_orbital,build_radial_orbitals
    parent,old=admit_parent(); rt=cycle.admission._runtime(); torch.set_num_threads(28)
    out=stage/'result'; out.mkdir()
    started=time.perf_counter()
    c=cycle.admission.common._read_coefficients(rt,PARENT/'result/trial_0/COEFFICIENTS.txt')
    original=cycle.admission.common._read_coefficients(rt,old['original_coefficient_path'])
    datasets,records,_=rt.load_frozen_c(old['freeze_path'],parent['freeze_sha256'],original,out,
        active_cache_index=old['active_cache_index_path'],active_cache_index_sha256=parent['active_cache_index_sha256'],
        band_guard_policy=cycle.RESPONSE)
    cycle.admission.common.validate_dataset_extent(datasets,old)
    load_seconds=time.perf_counter()-started
    datasets=tuple(prepare_periodic_occupied_reference(d) for d in datasets)
    guard=prepare_frozen_band_guard(datasets[0],original,extra_virtual_bands=2,allow_basis_expansion=True)
    floor=parent['occupied_capture_floor']; weights=old['quarter']['training_weights']
    checker=cycle.admission.refresh.accepted
    def evaluate(coefficients,views):
        beginning=time.perf_counter()
        views,ref=prepare_expansion_evaluation(views,coefficients)
        with torch.no_grad():
            band,capture=screen_candidate(views,coefficients,guard,floor)
            loss,capture,condition,_,rpa=_global_rpa_loss(views,coefficients,
                occupied_capture_tolerance=1-floor,weights=weights,
                frequency_batch_size=old['quarter']['configuration']['frequency_batch_size'],reference_cache=ref)
        record=dict(loss=float(loss),minimum_occupied_capture=capture,maximum_overlap_condition=condition,
            rpa=rpa,coefficient_guard=band,evaluation_seconds=time.perf_counter()-beginning)
        record['energy_quantity']='frozen_body_RPA_correlation_not_PBE_total'
        for prefix,key in (('rpa','candidate_energy_ha'),('reference_rpa','reference_energy_ha')):
            for suffix,divisor in (('cell',1),('c',2)):
                record[prefix+'_correlation_energy_ev_per_'+suffix]=rpa[key]*cycle.endpoint.HARTREE_TO_EV/divisor
        checker._record(record,floor,weights)
        return record
    center=evaluate(c,datasets)
    checker._same_grid_reference(center,parent['trials'][0]['rpa'],reproduce=True)
    cycle.endpoint._match_parent_initial(center,parent['trials'][0]['rpa'])
    rows=[]
    for l,name in enumerate(('4s3p2d','3s4p2d','3s3p3d')):
        slot=out/name; slot.mkdir()
        trial,seed=append_smooth_complement(c,'C',l,max_index=12)
        if not all(torch.equal(b[:,:a.shape[1]],a) for a,b in zip(c['C'],trial['C'])):
            raise ValueError('old columns changed')
        views=tuple(reprofile_active_primitives(d,trial) for d in datasets)
        write_periodic_optimizer_coefficients(slot/'COEFFICIENTS.txt',trial)
        orb=slot/('C_'+name+'.orb')
        options=dict(element='C',ecut_ry=100.,rcut_bohr=10.,dr_bohr=.01,smoothing_sigma_bohr=.1)
        write_abacus_orbital(orb,trial,**options)
        radius,radials=build_radial_orbitals(trial,**options)
        radial=radials[l][:,-1]; weights_r=np.ones(len(radius)); weights_r[1:-1:2]=4; weights_r[2:-1:2]=2
        weights_r*=.01/3
        deriv=np.gradient(radial,.01,edge_order=2)
        kinetic=float(np.sum(weights_r*(radius**2*deriv**2+l*(l+1)*radial**2)))
        tail=float(np.sum(weights_r*(radius>=9)*radius**2*radial**2))
        norm=float(np.sum(weights_r*radius**2*radial**2))
        if not all(math.isfinite(x) for x in (kinetic,tail,norm)) or abs(norm-1)>1e-10:
            raise ValueError('nonfinite or unnormalized exported radial')
        row=dict(name=name,added_l=l,nu=[x.shape[1] for x in trial['C']],added_ao_per_c=2*l+1,
            ao_per_c=22+2*l+1,seed=seed,radial_kinetic_ry=kinetic,radial_tail_probability_r_ge_9_bohr=tail,
            coefficient_sha256=cycle.endpoint._sha256((slot/'COEFFICIENTS.txt').read_bytes()),
            orbital_sha256=cycle.endpoint._sha256(orb.read_bytes()),
            mapping_sha256=[d.active_primitive_reduction.mapping_sha256 for d in views],
            actual_pbe='not_run',ordinary_sos='not_run',gw='not_run',physical_release_gate='hold')
        try:
            evaluated=evaluate(trial,views)
            checker._same_grid_reference(evaluated,center)
        except CandidateGuardError as error:
            row.update(gate='rejected_frozen_guard',reason=str(error),rpa='not_run')
        else:
            error=abs(evaluated['rpa']['candidate_energy_ha']-evaluated['rpa']['reference_energy_ha'])*27.211386245988/2
            old_error=abs(center['rpa']['candidate_energy_ha']-center['rpa']['reference_energy_ha'])*27.211386245988/2
            row.update(gate='improved_seed_requires_PBE' if error<old_error else 'not_improved_seed',
                rpa=evaluated,body_error_ev_per_c=error,gain_ev_per_c=old_error-error,
                gain_ev_per_added_ao=(old_error-error)/(2*l+1))
        cycle.endpoint._write_json(slot/'SCREEN.json',row); rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('rpa','mapping_sha256')}),flush=True)
    return dict(status='success',scope=SCOPE,parent_result_sha256=RESULT_SHA,parent_verification_sha256=VERIFY_SHA,
        source_commit=source['source_commit'],freeze_sha256=parent['freeze_sha256'],
        active_cache_index_sha256=parent['active_cache_index_sha256'],cache_loads=1,cache_load_seconds=load_seconds,
        load_records=records,center=center,candidates=rows,actual_scf_count=0,delta_st_runs=0,
        rpa_evaluations=1+sum(r['rpa']!='not_run' for r in rows),backward_passes=0,
        total_seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        physical_release_gate='hold',higher_l_scope='requires_existing_unreduced_mother_data_no_new_reference')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('stage','source-commit','deployment-sha256'): p.add_argument('--'+n,required=True)
    p.add_argument('--preflight-only',action='store_true'); args=p.parse_args()
    stage=Path(args.stage).resolve()
    if stage.parent!=ROOT or (stage/'result').exists(): raise ValueError('new sibling stage required')
    source=cycle.admission.common.verify_source(stage,args.deployment_sha256,args.source_commit)
    for n in ('screen_c_radial_expansion.py','run_c_radial_expansion.slurm'):
        if cycle.admission.refresh._WORKFLOW+n not in source['files']: raise ValueError('runner pin missing')
    admit_parent()
    recovery=admit_failed_cache_attempt()
    if args.preflight_only: return
    reservation=ROOT/('radial-expansion-'+RESULT_SHA[:16]+'-low-index-spd-cachefix-v2')
    reservation.mkdir()
    cycle.endpoint._write_json(reservation/'RESERVATION.json',dict(stage=str(stage),parent_result_sha256=RESULT_SHA,
        job_id=os.environ['SLURM_JOB_ID'],scope=SCOPE,physics_runs=0))
    result=run(stage,source)
    result['technical_recovery']=recovery
    cycle.admission.common.verify_source(stage,args.deployment_sha256,args.source_commit)
    admit_parent()
    result['deployment_sha256']=args.deployment_sha256
    cycle.endpoint._write_json(stage/'result/RESULT.json',result)
    cycle.endpoint._write_json(stage/'PROVENANCE.json',dict(status='success',source_commit=args.source_commit,
        deployment_sha256=args.deployment_sha256,result_sha256=cycle.endpoint._sha256((stage/'result/RESULT.json').read_bytes()),
        job_id=os.environ['SLURM_JOB_ID'],actual_scf_count=0,physical_release_gate='hold'))


if __name__=='__main__': main()
