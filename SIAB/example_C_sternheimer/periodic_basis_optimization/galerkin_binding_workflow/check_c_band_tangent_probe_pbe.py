"""Own-scope pure-PBE probes at an accepted Ec-step center; no processes launched."""

from pathlib import Path

import check_c_optimized_pbe as endpoint
from check_c_accepted_combined_step import _small, _expect
from check_c_combined_step_pbe import _root, _number, _near
from check_c_direction_probe_pbe import _digest, _equal, _json, _strict_log, _IDENTITY
from check_c_ec_gradient_step_pbe import _runtime_file, ORIGINAL_ENERGY_EV

PREPARATION='BAND_TANGENT_PROBE_PBE_PREPARED.json'
COLLECTION='BAND_TANGENT_PROBE_PBE_COLLECTION.json'
SCOPE='current_ec_center_direction_calibration_probe'
_ADMISSION=('preparation_stage','preparation_result_sha256','preparation_acceptance_sha256',
            'preparation_deployment_sha256','preparation_source_commit')
_COMMON=dict(scope=SCOPE,atoms_per_cell=2,physical_release_gate='hold',galerkin_energy='unmeasured',
    energy_quantity='PBE_total_energy_not_RPA_E0',tolerance_ev_per_c=.01,
    scheduler_gate='pending_external_validation')


def load_accepted_preparation(*args):
    from check_c_band_tangent_preparation import load_accepted_preparation as load
    return load(*args)


def _snapshot(*,preparation_stage,preparation_result_sha256,preparation_acceptance_sha256,
              preparation_deployment_sha256,preparation_source_commit,probe_index):
    if type(probe_index) is not int or not 0<=probe_index<8:
        raise ValueError('exactly eight indexed probes are permitted')
    stage=_root(preparation_stage)
    loaded=load_accepted_preparation(stage,preparation_result_sha256,preparation_acceptance_sha256,
                                     preparation_deployment_sha256,preparation_source_commit)
    center=loaded['center']; root=_root(center['stage']); result=center['result']
    pbe=result['actual_pbe']; slot=root/'result/pbe'
    _near(pbe['baseline_energy_ev'],ORIGINAL_ENERGY_EV,'original PBE reference')
    energy=_number(pbe['candidate_energy_ev'],'accepted center PBE')
    if abs((energy-ORIGINAL_ENERGY_EV)/2)>.01+1e-12:
        raise ValueError('accepted center exceeds original PBE tolerance')
    runtime=dict(abacus_binary=_runtime_file(pbe['abacus_binary'],pbe['abacus_sha256']),
        abacus_sha256=pbe['abacus_sha256'],mpi_library=_runtime_file(pbe['mpi_library'],pbe['mpi_sha256']),
        mpi_sha256=pbe['mpi_sha256'])
    prep_bytes=_small(slot,'EC_GRADIENT_STEP_PBE_PREPARED.json',_digest(pbe['preparation_sha256']))
    prep=_json(prep_bytes)
    hashes=prep['prepared_input_sha256']
    if not isinstance(hashes,dict) or len(hashes)!=5: raise ValueError('five frozen PBE inputs required')
    inputs={n:_small(slot,n,_digest(h)) for n,h in hashes.items()}
    if not {'INPUT','STRU','KPT'}.issubset(inputs) or any(not b for b in inputs.values()):
        raise ValueError('complete nonempty PBE inputs required')
    pseudo,orbital=endpoint._c2_files(inputs['STRU'])
    _equal(set(inputs),{'INPUT','STRU','KPT',pseudo,orbital},'exact PBE files')
    if set(inputs)&{PREPARATION,COLLECTION,'.provenance'}: raise ValueError('input filename collision')
    values=endpoint._input_values(inputs['INPUT'])
    _equal(values,endpoint._pure_pbe(values),'original pure PBE settings')
    _equal(endpoint._sha256(inputs[orbital]),result['orbital_sha256'],'center orbital')
    log_name='OUT.'+endpoint._basename(values.get('suffix','ABACUS'))+'/running_scf.log'
    _equal(prep['candidate_log_relative_path'],log_name,'center log path')
    log=_small(slot,log_name,_digest(pbe['candidate_log_sha256']))
    _near(_strict_log(log)['energy_ev'],energy,'center actual SCF energy')
    item=loaded['probes'][probe_index]; probe=item['manifest']; probe_root=_root(item['path'].parent)
    expected_name='probe_%02d'%probe_index
    if item['path'].name!='PROBE.json' or probe_root.name!=expected_name:
        raise ValueError('canonical bounded probe path required')
    expected_radius=(-.001,.001,-.002,.002)[probe_index%4]
    _expect(probe,dict(status='prepared',scope='direction_calibration_probe',
        direction_name='T' if probe_index<4 else 'N',signed_radius=expected_radius,
        center_result_sha256=center['result_sha256'],center_orbital_sha256=result['orbital_sha256'],
        directions_sha256=loaded['result']['directions_sha256'],physical_release_gate='hold',
        galerkin_energy='unmeasured'),'accepted probe')
    if probe.get('cheap_gate',{}).get('gate') is not True: raise ValueError('cheap probe rejected')
    files=dict(item['files'])
    for name,content in files.items(): _equal(_small(probe_root,name),content,'unchanged probe '+name)
    for kind,name in (('coefficient','COEFFICIENTS.txt'),('orbital','C_3s3p2d_probe.orb')):
        _equal(probe[kind+'_filename'],name,'probe filename')
        _equal(endpoint._sha256(files[name]),_digest(probe[kind+'_sha256']),'probe file hash')
    inputs[orbital]=files['C_3s3p2d_probe.orb']
    evidence={'.provenance/'+n:b for n,b in files.items()}
    evidence.update({'.provenance/center_RESULT.json':_small(root,'result/RESULT.json',center['result_sha256']),
        '.provenance/center_PBE_PREPARED.json':prep_bytes,'.provenance/center_running_scf.log':log})
    manifest=dict(_COMMON,status='prepared',format_version=1,preparation_stage=str(stage),
        preparation_result_sha256=preparation_result_sha256,preparation_acceptance_sha256=preparation_acceptance_sha256,
        preparation_deployment_sha256=preparation_deployment_sha256,preparation_source_commit=preparation_source_commit,
        probe_index=probe_index,probe_sha256=endpoint._sha256(files['PROBE.json']),
        center_stage=str(root),center_preparation_sha256=pbe['preparation_sha256'],
        baseline_energy_ev=ORIGINAL_ENERGY_EV,center_energy_ev=energy,baseline_log_sha256=pbe['baseline_log_sha256'],
        center_log_sha256=pbe['candidate_log_sha256'],candidate_log_relative_path=log_name,**runtime)
    manifest.update({k:probe[k] for k in _IDENTITY})
    manifest.update(prepared_input_sha256={n:endpoint._sha256(b) for n,b in inputs.items()},
                    evidence_sha256={n:endpoint._sha256(b) for n,b in evidence.items()})
    return manifest,dict(inputs,**evidence)


def _prepared(root,digest):
    root=_root(root); manifest=_json(_small(root,PREPARATION,_digest(digest)))
    expected,files=_snapshot(**{k:manifest[k] for k in _ADMISSION+('probe_index',)})
    _equal(manifest,expected,'complete frozen current-center probe preparation')
    for name,b in files.items(): _equal(_small(root,name),b,'staged input/evidence '+name)
    return manifest


def prepare_probe_pbe(*,output,**admission):
    path=Path(output)
    if path.exists() or path.is_symlink(): raise FileExistsError(path)
    output=_root(path); manifest,files=_snapshot(**admission)
    if output.name!='probe_%02d'%manifest['probe_index']: raise ValueError('canonical probe output name required')
    for p in (manifest['preparation_stage'],manifest['center_stage']):
        source=_root(p)
        if output==source or output in source.parents or source in output.parents:
            raise ValueError('separate output required')
    output.parent.mkdir(parents=True,exist_ok=True)
    lock=output.parent/'.band_tangent_pbe_prepare.lock'; lock.mkdir()
    try:
        siblings=[p for p in output.parent.iterdir() if p!=lock]
        if len(siblings)>=8: raise ValueError('eight-probe budget exhausted')
        for p in siblings:
            other=_prepared(p,endpoint._sha256(_small(p,PREPARATION)))
            for k in _ADMISSION: _equal(other[k],manifest[k],'same preparation campaign')
            if other['probe_index']==manifest['probe_index']: raise ValueError('duplicate probe')
        output.mkdir()
        for n,b in files.items():
            target=output/n; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f: f.write(b)
        endpoint._write_json(output/PREPARATION,manifest)
    finally: lock.rmdir()
    return manifest


def collect_probe_pbe(output,preparation_sha256):
    root=_root(output)
    if (root/COLLECTION).exists() or (root/COLLECTION).is_symlink(): raise FileExistsError(root/COLLECTION)
    p=_prepared(root,preparation_sha256)
    content=_small(root,p['candidate_log_relative_path']); measured=_strict_log(content)
    delta=(measured['energy_ev']-p['baseline_energy_ev'])/2
    center_delta=(measured['energy_ev']-p['center_energy_ev'])/2
    _number(delta,'original PBE difference'); _number(center_delta,'center PBE difference')
    r=dict(_COMMON,status='collected',preparation_sha256=preparation_sha256,
        candidate_energy_ev=measured['energy_ev'],energy_delta_ev_per_c=delta,delta_from_center_ev_per_c=center_delta,
        pbe_gate='pass' if abs(delta)<=.01+1e-12 else 'fail',scf_log_gate='pass',band_count_check='pass',
        band_counts=measured['band_counts'],candidate_log_path=str(root/p['candidate_log_relative_path']),
        candidate_log_sha256=measured['log_sha256'])
    keys=_ADMISSION+_IDENTITY+('probe_index','probe_sha256','center_stage','center_preparation_sha256',
        'baseline_energy_ev','center_energy_ev','baseline_log_sha256','center_log_sha256',
        'abacus_binary','abacus_sha256','mpi_library','mpi_sha256')
    r.update({k:p[k] for k in keys})
    _small(root,p['candidate_log_relative_path'],measured['log_sha256'])
    endpoint._write_json(root/COLLECTION,r)
    return r
