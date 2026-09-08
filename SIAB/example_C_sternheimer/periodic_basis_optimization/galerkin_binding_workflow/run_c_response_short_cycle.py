"""One bounded actual-PBE-constrained full-q C Galerkin cycle, all radials free."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

import screen_c_ec_step_constraints as admission
from c_response_band_policy import RESPONSE
from check_c_direction_probe_pbe import _strict_log
from check_c_ec_gradient_step_pbe import ORIGINAL_ENERGY_EV

endpoint = admission.refresh.endpoint
ROOT = Path('/work1/ghj/c-solid-fd8-q13-standard-20260903')
GRADIENT_STAGE = ROOT/'stage-c-ec-step-gradient-0a1e0326'
GRADIENT_SHA = 'b57963c60aba962e6ffa9a4a1d7842a04deab32a8cc4b27f51c7697e2bd71eca'
ACCEPTANCE_SHA = 'fbae9949d4257954ea8d6fb52d59f6076a5e0e2659c2f42464da82d86de59633'
PROBE_RADIUS = .001
RADII = (.02, .01, .005)
SCOPE = 'eight_radial_actual_pbe_constrained_response_cycle'


def finite(x):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError('finite numeric value required')
    return float(x)


def fit_radial_pbe(rates, center_energy, energies):
    import torch
    if len(rates) != 8 or len(energies) != 8 or any(len(row) != 2 for row in energies):
        raise ValueError('eight signed radial pairs required')
    center_energy = finite(center_energy)
    g = torch.tensor([finite(v) for v in rates], dtype=torch.float64)
    values = [[finite(v) for v in row] for row in energies]
    if bool((g <= 0).any()):
        raise ValueError('resolved positive radial Ec descent rates required')
    # Energies are per two-atom cell; PBE derivatives are per C.
    a = torch.tensor([(p-m)/(4*PROBE_RADIUS) for m,p in values], dtype=torch.float64)
    curvature = [(p+m-2*center_energy)/(2*PROBE_RADIUS**2) for m,p in values]
    tangent = g-a*(a@g)/(a@a) if float(a.norm()) > 1e-10 else g.clone()
    norm = float(tangent.norm())
    if norm <= 1e-10*float(g.norm()):
        raise ValueError('no resolved actual-PBE-tangent descent in measured radial span')
    d = tangent/norm
    return dict(radial_weights=d.tolist(), actual_pbe_slopes_ev_per_c=a.tolist(),
        axial_curvatures_ev_per_c=curvature, ec_slope_ha_per_cell=-float(g@d),
        actual_pbe_linear_slope_ev_per_c=float(a@d), retained_descent_fraction=float(g@d)/float(g.norm()),
        coordinate_metric='Euclidean_coefficient_signed_QR_frame',
        maximum_band_normal='not_constrained', mixed_curvature='unmeasured',
        finite_step_safety='requires_actual_PBE', derivative_scope='central_difference_at_one_radius')


def improves(old, new):
    for r in (old, new):
        if finite(r['loss']) < 0:
            raise ValueError('nonnegative loss required')
        for key in ('candidate_energy_ha', 'reference_energy_ha'):
            finite(r['rpa'][key])
    ref = old['rpa']['reference_energy_ha']
    if not math.isclose(ref, new['rpa']['reference_energy_ha'], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('same frozen reference required')
    return (abs(new['rpa']['candidate_energy_ha']-ref) < abs(old['rpa']['candidate_energy_ha']-ref)
            and new['loss'] <= old['loss'])


def calibrate_search(rates, center_pbe, center_record, measure, evaluate):
    from periodic_galerkin_fit import CandidateGuardError
    samples, energies = [], []
    for i in range(8):
        pair = []
        for j, sign in enumerate((-1., 1.)):
            weights = [0.]*8; weights[i] = sign*PROBE_RADIUS
            sample = measure('probe_%02d'%(2*i+j), weights)
            pair.append(finite(sample['candidate_energy_ev']))
            samples.append(sample)
        energies.append(pair)
    fit = fit_radial_pbe(rates, center_pbe, energies)
    result = dict(calibration=samples, tangent=fit, trials=[], selected_trial=None,
        candidate_gate='rejected_no_accepted_trial', physical_release_gate='hold')
    for i, radius in enumerate(RADII):
        name = 'trial_'+str(i)
        try:
            sample = measure(name, [radius*w for w in fit['radial_weights']])
        except CandidateGuardError as error:
            result['trials'].append(dict(name=name,radius=radius,gate='rejected_cheap_guard',
                                         reason=str(error),actual_scf_count=0,rpa='not_run'))
            continue
        row = dict(name=name, radius=radius, actual_pbe=sample, rpa='not_run')
        result['trials'].append(row)
        if sample['pbe_gate'] != 'pass':
            continue
        measured = evaluate(name)
        row['rpa'] = measured
        if improves(center_record, measured):
            result.update(selected_trial=name, candidate_gate='improved_frozen_body')
            break
    return result


def reserve(root, stage, identity):
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
    path = Path(root)/('response-short-cycle-execution-'+fingerprint)
    path.mkdir()
    endpoint._write_json(path/'RESERVATION.json', dict(identity, stage=str(stage),
        fingerprint=fingerprint, status='reserved_no_automatic_replay', actual_scf_budget=19))


def frozen_pbe_bundle(center):
    small = admission.refresh.accepted._small
    pbe = center['result']['actual_pbe']
    slot = center['stage']/'result/pbe'
    prep = json.loads(small(slot, 'EC_GRADIENT_STEP_PBE_PREPARED.json', pbe['preparation_sha256']))
    inputs = {n:small(slot,n,h) for n,h in prep['prepared_input_sha256'].items()}
    pseudo, orbital = endpoint._c2_files(inputs['STRU'])
    if set(inputs) != {'INPUT','STRU','KPT',pseudo,orbital}:
        raise ValueError('exact five frozen PBE files required')
    values = endpoint._input_values(inputs['INPUT'])
    if values != endpoint._pure_pbe(values):
        raise ValueError('pure PBE INPUT required')
    if endpoint._sha256(inputs[orbital]) != center['result']['orbital_sha256']:
        raise ValueError('accepted center orbital mismatch')
    measured = _strict_log(small(slot,prep['candidate_log_relative_path'],pbe['candidate_log_sha256']))
    if not math.isclose(measured['energy_ev'],pbe['candidate_energy_ev'],abs_tol=1e-10,rel_tol=0):
        raise ValueError('accepted actual PBE energy mismatch')
    if pbe['baseline_energy_ev'] != ORIGINAL_ENERGY_EV:
        raise ValueError('original TZDP PBE baseline cannot reset')
    return inputs,orbital,prep['candidate_log_relative_path']


def run_scf(slot, inputs, log_name, launcher):
    slot = Path(slot)
    if (any(Path(n).name != n or n in ('.','..') for n in inputs)
            or Path(log_name).is_absolute() or '..' in Path(log_name).parts):
        raise ValueError('safe relative log and flat input names required')
    slot.mkdir()
    hashes = {n:endpoint._sha256(b) for n,b in inputs.items()}
    for n,b in inputs.items():
        with (slot/n).open('xb') as stream: stream.write(b)
    endpoint._write_json(slot/'INPUT_HASHES.json',hashes)
    started = time.perf_counter()
    with (slot/'abacus.out').open('xb') as stream:
        subprocess.run(launcher,cwd=slot,stdout=stream,stderr=subprocess.STDOUT,check=True)
    for n,h in hashes.items(): admission.common._check_file(slot/n,h)
    log = (slot/log_name).read_bytes()
    measured = _strict_log(log)
    delta = (measured['energy_ev']-ORIGINAL_ENERGY_EV)/2
    result = dict(candidate_energy_ev=measured['energy_ev'],baseline_energy_ev=ORIGINAL_ENERGY_EV,
        energy_delta_ev_per_c=delta,pbe_gate='pass' if abs(delta)<=.01+1e-12 else 'fail',
        scf_log_gate='pass',band_counts=measured['band_counts'],input_sha256=hashes,
        candidate_log_path=str(slot/log_name),candidate_log_sha256=endpoint._sha256(log),
        seconds=time.perf_counter()-started,exit_code=0,physical_release_gate='hold')
    endpoint._write_json(slot/'COLLECTION.json',result)
    return result


def validate_stage(stage, source, center):
    stage = Path(stage).resolve()
    if stage.parent != ROOT or Path(source['source_directory']).resolve() != stage/'source':
        raise ValueError('new sibling campaign stage with its own source required')
    if stage == Path(center['stage']).resolve() or stage == GRADIENT_STAGE:
        raise ValueError('accepted data cannot be a run destination')
    if (stage/'result').exists() or (stage/'PROVENANCE.json').exists():
        raise FileExistsError('stage already executed')


def run_cycle(stage, source, center, gradient, launcher):
    import torch
    from periodic_galerkin_radial_diagnostics import radial_descent_direction, retract_displacement
    from periodic_galerkin_basis import write_periodic_optimizer_coefficients
    from export_periodic_orbitals import write_abacus_orbital
    from run_c_combined_step import screen_candidate
    from backoff_c_optimized_direction import evaluate_pair

    started = time.perf_counter(); stage = Path(stage)
    reserve(ROOT,stage,dict(center_sha256=center['result_sha256'],gradient_sha256=GRADIENT_SHA,
        policy=RESPONSE,probe_radius=PROBE_RADIUS,trial_radii=list(RADII),scope=SCOPE))
    output = stage/'result'; output.mkdir()
    rt = admission._runtime(); rt.torch.set_num_threads(28)
    c = admission.common._read_coefficients(rt,center['coefficient_path'])
    original = admission.common._read_coefficients(rt,center['original_coefficient_path'])
    pairs=[(r['element'],r['l'],z) for r in gradient['energy_gradient']['channels'] for z in range(len(r['radials']))]
    if pairs != [('C',l,z) for l,n in enumerate((3,3,2)) for z in range(n)]:
        raise ValueError('all eight radial coordinates required')
    basis=[radial_descent_direction(c,gradient['energy_gradient'],e,l,z) for e,l,z in pairs]
    if any(b is None for b in basis): raise ValueError('unresolved radial gradient')
    rates=[r['horizontal_norm'] for ch in gradient['energy_gradient']['channels'] for r in ch['radials']]
    load_started=time.perf_counter()
    datasets,records,guard=rt.load_frozen_c(center['freeze_path'],center['result']['freeze_sha256'],original,output,
        active_cache_index=center['active_cache_index_path'],active_cache_index_sha256=center['result']['active_cache_index_sha256'],
        band_guard_policy=RESPONSE)
    load_seconds=time.perf_counter()-load_started
    admission.common.validate_dataset_extent(datasets,center)
    inputs,orbital_name,log_name=frozen_pbe_bundle(center)
    floor=center['occupied_capture_floor']; weights=center['quarter']['training_weights']
    measured_candidates={}; forward_pairs=[]
    def measure(name,w):
        slot=output/name; slot.mkdir()
        with torch.no_grad():
            displacement={e:[sum((x*b[e][l] for x,b in zip(w,basis)),torch.zeros_like(block))
                        for l,block in enumerate(channels)] for e,channels in c.items()}
            trial=retract_displacement(c,displacement,1.)
            bands,capture=screen_candidate(datasets,trial,guard,floor)
            diagnostics=guard(trial,diagnostics=True)
        write_periodic_optimizer_coefficients(slot/'COEFFICIENTS.txt',trial)
        write_abacus_orbital(slot/'C_3s3p2d.orb',trial,element='C',ecut_ry=100.,rcut_bohr=10.,dr_bohr=.01,smoothing_sigma_bohr=.1)
        proposal=dict(scope=SCOPE,name=name,radial_weights=w,center_sha256=center['result_sha256'],
            gradient_sha256=GRADIENT_SHA,source_commit=source['source_commit'],policy=RESPONSE,
            nu=[3,3,2,0,0],fixed_nu=[0]*5,ao_per_C=22,cheap_gate=bands,band_diagnostics=diagnostics,
            capture=capture,occupied_capture_floor=floor,
            coefficient_sha256=endpoint._sha256((slot/'COEFFICIENTS.txt').read_bytes()),
            orbital_sha256=endpoint._sha256((slot/'C_3s3p2d.orb').read_bytes()),physical_release_gate='hold')
        endpoint._write_json(slot/'CANDIDATE.json',proposal)
        newinputs=dict(inputs); newinputs[orbital_name]=(slot/'C_3s3p2d.orb').read_bytes()
        sample=run_scf(slot/'pbe',newinputs,log_name,launcher)
        sample.update(name=name,radial_weights=w,coefficient_sha256=proposal['coefficient_sha256'],
            orbital_sha256=proposal['orbital_sha256'],candidate_sha256=endpoint._sha256((slot/'CANDIDATE.json').read_bytes()))
        measured_candidates[name]=(proposal,sample)
        endpoint._write_json(slot/'MEASUREMENT.json',sample)
        print(json.dumps(dict(measurement=name,pbe_delta_mev_per_c=1000*sample['energy_delta_ev_per_c'],gate=sample['pbe_gate'])),flush=True)
        return sample
    def validate_center(first):
        previous=center['result']['candidate']; accepted=admission.refresh.accepted
        accepted._record(first,floor,weights)
        endpoint._match_parent_initial(first,previous)
        accepted._same_grid_reference(first,previous,reproduce=True)
        for n in ('pi','trace_log','energy'):
            accepted._near(first['rpa'][n+'_relative_squared_error'],previous['rpa'][n+'_relative_squared_error'],'unchanged center '+n)
    def evaluate(name):
        proposal,sample=measured_candidates[name]; slot=output/name
        admission.common._check_file(slot/'CANDIDATE.json',sample['candidate_sha256'])
        admission.common._check_file(slot/'COEFFICIENTS.txt',proposal['coefficient_sha256'])
        admission.common._check_file(slot/'C_3s3p2d.orb',proposal['orbital_sha256'])
        trial=admission.common._read_coefficients(rt,slot/'COEFFICIENTS.txt')
        first,new=evaluate_pair(datasets,c,trial,guard=guard,frequency_batch_size=center['quarter']['configuration']['frequency_batch_size'],
            weights=weights,occupied_capture_floor=floor,initial_validator=validate_center)
        admission.refresh.accepted._record(new,floor,weights)
        admission.refresh.accepted._same_grid_reference(new,first)
        endpoint._write_json(slot/'GALERKIN.json',dict(center=first,candidate=new))
        forward_pairs.append(name)
        return new
    result=calibrate_search(rates,center['result']['actual_pbe']['candidate_energy_ev'],center['result']['candidate'],measure,evaluate)
    result.update(status='success',scope=SCOPE,source_commit=source['source_commit'],center_result_sha256=center['result_sha256'],
        center_acceptance_sha256=center['acceptance_sha256'],gradient_sha256=GRADIENT_SHA,gradient_acceptance_sha256=ACCEPTANCE_SHA,
        center_stage=str(center['stage']),gradient_stage=str(GRADIENT_STAGE),policy=RESPONSE,
        freeze_sha256=center['result']['freeze_sha256'],active_cache_index_sha256=center['result']['active_cache_index_sha256'],
        original_coefficient_path=str(center['original_coefficient_path']),occupied_capture_floor=floor,
        center=center['result']['candidate'],center_actual_pbe=center['result']['actual_pbe'],
        actual_scf_count=len(measured_candidates),rpa_evaluations=2*len(forward_pairs),backward_passes=0,
        cache_loads=1,load_records=records,cache_load_seconds=load_seconds,total_seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ordinary_sos_qavg='pending',gw='pending',
        nu=[3,3,2,0,0],fixed_nu=[0]*5,ao_per_C=22,scheduler_gate='pending_external_validation')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','source-commit','deployment-sha256'): p.add_argument('--'+name,required=True)
    p.add_argument('--preflight-only',action='store_true'); args=p.parse_args()
    stage=Path(args.stage).resolve()
    source=admission.common.verify_source(stage,args.deployment_sha256,args.source_commit)
    required={admission.refresh._WORKFLOW+n for n in ('run_c_response_short_cycle.py','run_c_response_short_cycle.slurm','c_response_band_policy.py')}
    if not required.issubset(source['files']): raise ValueError('short-cycle source pins required')
    center,gradient=admission.admit_gradient(GRADIENT_STAGE,GRADIENT_SHA,ACCEPTANCE_SHA)
    validate_stage(stage,source,center)
    runtime=center['result']['actual_pbe']
    for pkey,hkey in (('abacus_binary','abacus_sha256'),('mpi_library','mpi_sha256')):
        admission.common._check_file(runtime[pkey],runtime[hkey])
    if (os.environ.get('SLURM_JOB_NUM_NODES')!='1' or os.environ.get('SLURM_NTASKS')!='4'
            or os.environ.get('OMP_NUM_THREADS')!='7'
            or Path(os.environ.get('I_MPI_PMI_LIBRARY','/missing')).resolve()!=Path(runtime['mpi_library']).resolve()):
        raise ValueError('one node / four MPI x seven OMP / frozen PMI2 required')
    frozen_pbe_bundle(center)
    if args.preflight_only: return
    endpoint._write_json(stage/'EXECUTION_PROVENANCE.json',dict(status='launching',source_commit=args.source_commit,
        deployment_sha256=args.deployment_sha256,job_id=os.environ['SLURM_JOB_ID'],actual_scf_budget=19,
        abacus_sha256=runtime['abacus_sha256'],pmi_sha256=runtime['mpi_sha256'],physical_release_gate='hold'))
    result=run_cycle(stage,source,center,gradient,['srun','--mpi=pmi2','--cpu-bind=none','-n','4',runtime['abacus_binary']])
    admission.common.verify_source(stage,args.deployment_sha256,args.source_commit)
    for pkey,hkey in (('abacus_binary','abacus_sha256'),('mpi_library','mpi_sha256')):
        admission.common._check_file(runtime[pkey],runtime[hkey])
    admission.admit_gradient(GRADIENT_STAGE,GRADIENT_SHA,ACCEPTANCE_SHA)
    result['deployment_sha256']=args.deployment_sha256
    endpoint._write_json(stage/'result/RESULT.json',result)
    endpoint._write_json(stage/'PROVENANCE.json',dict(status='success',source_commit=args.source_commit,
        deployment_sha256=args.deployment_sha256,result_sha256=endpoint._sha256((stage/'result/RESULT.json').read_bytes()),
        candidate_gate=result['candidate_gate'],actual_scf_count=result['actual_scf_count'],rpa_evaluations=result['rpa_evaluations'],
        physical_release_gate='hold'))


if __name__=='__main__': main()
