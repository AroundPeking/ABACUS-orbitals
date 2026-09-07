"""Bounded actual-PBE calibration at the current accepted band-tangent center."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from check_c_band_tangent_probe_pbe import (
    load_accepted_preparation,prepare_probe_pbe,collect_probe_pbe,PREPARATION,COLLECTION,
    _ADMISSION,_root,_runtime_file,endpoint,
)


def analyze_axes(center,samples):
    from periodic_galerkin_direction_calibration import analyze_pbe_axes
    return analyze_pbe_axes(center,samples)


def require_source_files(manifest):
    prefix='SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/'
    required={prefix+n for n in ('check_c_band_tangent_preparation.py','check_c_band_tangent_probe_pbe.py',
        'run_c_band_tangent_pbe_calibration.py','run_c_band_tangent_pbe_calibration.slurm')}
    if not required.issubset(manifest['files']): raise ValueError('complete current calibration source required')


def validate_stage(stage,options,loaded,source_directory=None):
    stage=_root(stage)
    parent=_root(options['preparation_stage'])
    if stage.parent!=parent.parent: raise ValueError('calibration must use the same campaign root')
    for p in (parent,_root(loaded['center']['stage'])):
        if stage==p or stage in p.parents or p in stage.parents:
            raise ValueError('calibration stage must be separate from accepted evidence')
    if source_directory is not None:
        source=_root(source_directory)
        # The staging contract owns stage/source; all other overlapping layouts
        # could place run outputs inside the immutable source tree.
        if (source==stage or source in stage.parents
                or (stage in source.parents and source!=stage/'source')):
            raise ValueError('calibration must not write into deployed source')


def reserve_execution(stage,options,loaded):
    identity=dict(preparation_result_sha256=options['preparation_result_sha256'],
        center_result_sha256=loaded['result']['center_result_sha256'],directions_sha256=loaded['result']['directions_sha256'])
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    reservation=_root(options['preparation_stage']).parent/('band-tangent-pbe-execution-'+fingerprint)
    reservation.mkdir()
    endpoint._write_json(reservation/'RESERVATION.json',dict(identity,stage=str(stage),actual_scf_budget=8,
        status='reserved_no_automatic_replay',fingerprint=fingerprint))


def run_calibration(*,stage,preparation_stage,preparation_result_sha256,preparation_acceptance_sha256,
                    preparation_deployment_sha256,preparation_source_commit,source_commit,launcher):
    stage=_root(stage); output=stage/'result'
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    started=time.perf_counter()
    options=dict(preparation_stage=str(preparation_stage),preparation_result_sha256=preparation_result_sha256,
        preparation_acceptance_sha256=preparation_acceptance_sha256,
        preparation_deployment_sha256=preparation_deployment_sha256,preparation_source_commit=preparation_source_commit)
    loaded=load_accepted_preparation(*(options[k] for k in _ADMISSION))
    validate_stage(stage,options,loaded)
    reserve_execution(stage,options,loaded)
    output.mkdir(); (stage/'PBE_EXECUTION_LOCK').mkdir()
    staged=[]
    # Stage every prescribed probe before spending any SCF work.
    for index in range(8):
        destination=output/'pbe'/('probe_%02d'%index)
        prepare_probe_pbe(output=destination,probe_index=index,**options)
        digest=endpoint._sha256((destination/PREPARATION).read_bytes())
        staged.append((index,destination,digest))
    setup_seconds=time.perf_counter()-started
    samples,executions=[],[]
    for index,destination,digest in staged:
        t=time.perf_counter()
        with (destination/'abacus.out').open('xb') as stream:
            subprocess.run(launcher,cwd=destination,stdout=stream,stderr=subprocess.STDOUT,check=True)
        sample=collect_probe_pbe(destination,digest)
        samples.append(sample)
        record=dict(probe_index=index,preparation_sha256=digest,
            collection_path=str(destination/COLLECTION),collection_sha256=endpoint._sha256((destination/COLLECTION).read_bytes()),
            exit_code=0,seconds=time.perf_counter()-t)
        endpoint._write_json(destination/'EXECUTION.json',record); executions.append(record)
        print(json.dumps(dict(completed=len(samples),sample=sample),allow_nan=False),flush=True)
        if sample['pbe_gate']!='pass': break
    complete=len(samples)==8 and all(x['pbe_gate']=='pass' for x in samples)
    analysis=analyze_axes(loaded['center']['result']['actual_pbe']['candidate_energy_ev'],samples) if complete else None
    gate=('pass' if analysis['consistency_gate']=='pass' else 'rejected_derivative_consistency') if complete else 'rejected_actual_pbe_probe'
    load_accepted_preparation(*(options[k] for k in _ADMISSION))
    result=dict(status='success',scope='current_ec_center_bounded_actual_pbe_calibration',source_commit=source_commit,
        **options,samples=samples,executions=executions,center_result_sha256=loaded['result']['center_result_sha256'],
        directions_sha256=loaded['result']['directions_sha256'],actual_scf_budget=8,actual_scf_count=len(samples),
        calibration_gate=gate,directions_analysis=analysis if analysis else 'unmeasured_incomplete_calibration',
        setup_seconds=setup_seconds,total_seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cache_loads=0,rpa_evaluations=0,backward_passes=0,optimizer_steps=0,physical_candidate_count=0,
        coefficient_update='none',physical_release_gate='hold',scheduler_gate='pending_external_validation')
    endpoint._write_json(output/'BAND_TANGENT_PBE_CALIBRATION.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight-only',action='store_true')
    for name in ('stage','source-commit','deployment-sha256')+tuple(k.replace('_','-') for k in _ADMISSION):
        p.add_argument('--'+name,required=True)
    args=vars(p.parse_args()); preflight=args.pop('preflight_only')
    stage=_root(args['stage']); deployment=args.pop('deployment_sha256')
    from check_c_ec_constraint_screen import screen
    source=screen.verify_source(stage,deployment,args['source_commit'])
    require_source_files(source)
    loaded=load_accepted_preparation(*(args[k] for k in _ADMISSION))
    validate_stage(stage,args,loaded,source_directory=source['source_directory'])
    actual=loaded['center']['result']['actual_pbe']
    binary=_runtime_file(actual['abacus_binary'],actual['abacus_sha256'])
    pmi=_runtime_file(actual['mpi_library'],actual['mpi_sha256'])
    if (os.environ.get('SLURM_JOB_NUM_NODES')!='1' or os.environ.get('SLURM_NTASKS')!='4'
            or os.environ.get('OMP_NUM_THREADS')!='7' or os.environ.get('I_MPI_PMI_LIBRARY')!=pmi):
        raise ValueError('one node, four MPI ranks, seven OMP threads and pinned PMI2 required')
    if preflight: return
    endpoint._write_json(stage/'EXECUTION_PROVENANCE.json',dict(status='launching',source_commit=args['source_commit'],
        deployment_sha256=deployment,scheduler_job=os.environ['SLURM_JOB_ID'],source_archive_sha256=source['archive_sha256'],
        abacus_sha256=actual['abacus_sha256'],pmi_sha256=actual['mpi_sha256'],actual_scf_budget=8,
        mpi_ranks=4,omp_threads=7,rpa='not_run',delta_st='not_run',librpa='not_run',physical_release_gate='hold'))
    result=run_calibration(**args,launcher=['srun','--mpi=pmi2','--cpu-bind=none','-n','4',binary])
    screen.verify_source(stage,deployment,args['source_commit'])
    _runtime_file(binary,actual['abacus_sha256']); _runtime_file(pmi,actual['mpi_sha256'])
    endpoint._write_json(stage/'PROVENANCE.json',dict(status='success',scope=result['scope'],source_commit=args['source_commit'],
        deployment_sha256=deployment,result_sha256=endpoint._sha256((stage/'result/BAND_TANGENT_PBE_CALIBRATION.json').read_bytes()),
        calibration_gate=result['calibration_gate'],actual_scf_count=result['actual_scf_count'],physical_release_gate='hold'))


if __name__=='__main__': main()
