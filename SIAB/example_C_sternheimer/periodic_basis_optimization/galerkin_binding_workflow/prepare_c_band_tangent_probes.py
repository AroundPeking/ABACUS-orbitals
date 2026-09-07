"""Prepare eight band-tangent probes, without SCF or response calculations."""

import argparse
import json
import math
from pathlib import Path
import resource
import time

import check_c_ec_constraint_screen as admission


def summarize_target_branches(center,probes):
    def read(r):
        if r.get('target_band_identity')!='sorted_eigenvalue_index_not_eigenvector_tracking':
            raise ValueError('explicit sorted-band identity required')
        result={}
        for row in r['target_band_details']:
            if len(row['band_indices'])!=len(row['signed_changes_ev']):
                raise ValueError('complete signed band detail required')
            for band,value in zip(row['band_indices'],row['signed_changes_ev']):
                key=(row['source_ik'],band)
                if key in result or type(value) not in (float,int) or not math.isfinite(value):
                    raise ValueError('unique finite band detail required')
                result[key]=value
        if not result or not math.isclose(max(map(abs,result.values())),r['maximum_target_band_change_ev'],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('detail does not reproduce maximum band change')
        return result
    original=read(center)
    maximum=max(map(abs,original.values()))
    tolerance=1e-6
    near={k for k,v in original.items() if maximum-abs(v)<=tolerance}
    active=[]
    for r in probes:
        values=read(r)
        if set(values)!=set(original): raise ValueError('same k/band grid required')
        top=max(map(abs,values.values()))
        active.append({k for k,v in values.items() if top-abs(v)<=tolerance})
    return dict(center_near_active_indices=[list(k) for k in sorted(near)],near_active_tolerance_ev=tolerance,
        probe_near_active_indices=[[list(k) for k in sorted(keys)] for keys in active],
        probe_maxima_within_center_near_active_set=all(keys.issubset(near) for keys in active),
        smoothness_gate='unproven_sorted_eigenvalue_branches',
        interpretation='finite sampled guards apply; no eigenvector tracking or global smoothness claim')


def prepare(*,stage,source_commit,deployment_sha256,screen_stage,screen_result_sha256,
            screen_acceptance_sha256,screen_deployment_sha256,screen_source_commit,
            gradient_result_sha256,gradient_acceptance_sha256):
    screen=admission.screen
    stage,previous=screen._root(stage),screen._root(screen_stage)
    if stage==previous or stage in previous.parents or previous in stage.parents:
        raise ValueError('separate preparation stage required')
    output=stage/'result'
    if output.exists() or output.is_symlink() or (stage/'PROVENANCE.json').exists(): raise FileExistsError(output)
    started=time.perf_counter()
    manifest=screen.verify_source(stage,deployment_sha256,source_commit)
    required={screen.refresh._WORKFLOW+n for n in ('prepare_c_band_tangent_probes.py','run_c_band_tangent_preparation.slurm','check_c_ec_constraint_screen.py')}
    if not required.issubset(manifest['files']): raise ValueError('preparation source pins required')
    args=(previous,screen_result_sha256,screen_acceptance_sha256,screen_deployment_sha256,
          screen_source_commit,gradient_result_sha256,gradient_acceptance_sha256)
    center,g,r,rows=admission.load_accepted_screen(*args)
    rt=screen._runtime()
    rt.torch.set_num_threads(28)
    from periodic_galerkin_band_tangent import build_band_tangent
    from periodic_galerkin_fit import _minimum_occupied_capture, CandidateGuardError
    from prepare_c_pbe_direction_calibration import export_probes
    c=screen.common._read_coefficients(rt,center['coefficient_path'])
    original=screen.common._read_coefficients(rt,center['original_coefficient_path'])
    artifact=build_band_tangent(c,g['energy_gradient'],rows)
    artifact.update(source_commit=source_commit,screen_result_sha256=screen_result_sha256,
        screen_acceptance_sha256=screen_acceptance_sha256,center_result_sha256=center['result_sha256'],
        center_coefficient_sha256=g['coefficient_sha256'],center_orbital_sha256=g['orbital_sha256'])
    output.mkdir()
    t=time.perf_counter()
    datasets,records,guard=rt.load_frozen_c(center['freeze_path'],center['result']['freeze_sha256'],original,
        output,active_cache_index=center['active_cache_index_path'],active_cache_index_sha256=center['result']['active_cache_index_sha256'])
    load_seconds=time.perf_counter()-t
    screen.common.validate_dataset_extent(datasets,center)
    detail=guard(c,diagnostics=True)
    screen.compare_center_guard({k:detail[k] for k in r['center_band_screen']},r['center_band_screen'])
    details=[]
    def cheap(trial):
        bands=guard(trial,diagnostics=True)
        capture=_minimum_occupied_capture(datasets,trial,relative_rank_tolerance=1e-12,condition_limit=1e12)
        if not math.isfinite(capture) or capture<center['occupied_capture_floor']:
            raise CandidateGuardError('original occupied capture floor failed')
        details.append(bands)
        return dict(gate=True,band_screen=bands,minimum_occupied_capture=capture,
            occupied_capture_floor=center['occupied_capture_floor'],overlap_rank_and_condition_gate='pass',
            overlap_relative_rank_tolerance=1e-12,overlap_condition_limit=1e12)
    probes=export_probes(c,artifact,center=g,center_result_sha256=center['result_sha256'],output=output/'probes',screen=cheap)
    all_prepared=all(p['status']=='prepared' for p in probes)
    branches=summarize_target_branches(detail,details)
    result=dict(status='success',scope='bounded_current_center_band_tangent_probe_preparation',
        source_commit=source_commit,deployment_sha256=deployment_sha256,screen_stage=str(previous),
        screen_result_sha256=screen_result_sha256,screen_acceptance_sha256=screen_acceptance_sha256,
        gradient_result_sha256=gradient_result_sha256,gradient_acceptance_sha256=gradient_acceptance_sha256,
        center_result_sha256=center['result_sha256'],center_coefficient_sha256=g['coefficient_sha256'],
        center_orbital_sha256=g['orbital_sha256'],center_actual_pbe=center['result']['actual_pbe'],
        directions_sha256=screen.refresh.endpoint._sha256((output/'probes/DIRECTIONS.json').read_bytes()),
        probes=probes,center_band_detail=detail,branch_diagnostic=branches,
        preparation_gate='pass' if all_prepared else 'rejected_cheap_probe',load_records=records,
        cache_load_seconds=load_seconds,total_seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cache_loads=1,probe_count=8,actual_scf_count=0,rpa_evaluations=0,backward_passes=0,
        optimizer_steps=0,physical_candidate_count=0,coefficient_update='none',physical_release_gate='hold')
    json.dumps(result,allow_nan=False)
    admission.load_accepted_screen(*args)
    screen.verify_source(stage,deployment_sha256,source_commit)
    screen.refresh.endpoint._write_json(output/'BAND_TANGENT_PREPARED.json',result)
    screen.refresh.endpoint._write_json(stage/'PROVENANCE.json',dict(status='success',scope=result['scope'],
        source_commit=source_commit,deployment_sha256=deployment_sha256,
        result_sha256=screen.refresh.endpoint._sha256((output/'BAND_TANGENT_PREPARED.json').read_bytes()),
        preparation_gate=result['preparation_gate'],physical_release_gate='hold'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','source-commit','deployment-sha256','screen-stage','screen-result-sha256',
            'screen-acceptance-sha256','screen-deployment-sha256','screen-source-commit',
            'gradient-result-sha256','gradient-acceptance-sha256'):
        p.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(p.parse_args())),allow_nan=False))
