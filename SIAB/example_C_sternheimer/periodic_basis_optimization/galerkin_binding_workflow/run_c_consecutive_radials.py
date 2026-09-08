"""Consecutive full-q C radial updates; actual PBE, never a surrogate, accepts."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import subprocess
import time

ROOT = Path('/work1/ghj/c-solid-fd8-q13-standard-20260903')
PARENT = ROOT/'stage-c-radial-step-cap-7b25a6a4'
PARENT_SHA = '3d6a7c081119a4e3195c26c652c817afcaf4103ebaf0db867687c9bfd7da5203'
VERIFY_SHA = 'fd5bebee5512988c1be2d0dfbdfc520c3ade562cc6c959cb6c158b37c46789d9'
ORIGINAL_ENERGY_EV = -309.8590820440637
SCOPE = 'consecutive_eight_radial_actual_PBE_constrained_frozen_body'
sys.path[:0] = [str(Path(__file__).resolve().parents[4]/'SIAB/opt_orb_pytorch_dpsi'),
                str(Path(__file__).resolve().parent.parent)]


def finite(x):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError('finite number required')
    return float(x)


def controller_record(record, pbe):
    r = record['rpa']
    return dict(objective=abs(finite(r['candidate_energy_ha'])-finite(r['reference_energy_ha'])),
                loss=finite(record['loss']), pbe=finite(pbe))


def validate_parent_trial(row):
    pbe, p = row['actual_pbe'], row['proposal']
    if (row['radius'] != .04 or row['gate'] != 'diagnostic_measured'
            or pbe['pbe_gate'] != 'pass' or abs(finite(pbe['energy_delta_ev_per_c'])) > .010
            or pbe['baseline_energy_ev'] != ORIGINAL_ENERGY_EV
            or p['nu'] != [3,3,2,0,0] or p['fixed_nu'] != [0]*5 or p['ao_per_C'] != 22):
        raise ValueError('verified feasible all-free 3s3p2d radius .04 parent required')
    controller_record(row['rpa']['candidate'], pbe['energy_delta_ev_per_c'])


def candidate_nonconvergence(error, content):
    if str(error) != 'SCF log is not unambiguously converged':
        return False
    text=content.decode('utf-8')
    return (not re.search(r'#SCF IS CONVERGED#',text,re.I)
            and bool(re.search(r'SCF\s+(?:IS\s+)?NOT\s+CONVERGED|convergence\s+has\s+not|'
                               r'(?:SCF|charge\s+density)\s+(?:did\s+not|failed\s+to)\s+converge',text,re.I)))


def validate_continuation_payload(result, actual):
    if (result.get('status')!='success' or result.get('nu')!=[3,3,2,0,0]
            or result.get('fixed_nu')!=[0]*5 or result.get('ao_per_C')!=22
            or result.get('physical_release_gate')!='hold'
            or result.get('full_constrained_stationarity')!='not_established'
            or actual.get('pbe_gate')!='pass' or actual.get('baseline_energy_ev')!=ORIGINAL_ENERGY_EV):
        raise ValueError('same-size independently verified continuation required')
    model=result['optimization']['pbe_gradient']
    if len(model)!=248 or any(not math.isfinite(finite(x)) for x in model):
        raise ValueError('finite 248-coordinate PBE secant model required')
    pbe=finite(result['optimization']['record']['pbe'])
    if abs(pbe)>.010 or abs(pbe-finite(actual['energy_delta_ev_per_c']))>1e-12:
        raise ValueError('continuation must preserve measured PBE and original baseline')


def admit_parent():
    import diagnose_c_radial_step_cap as prior
    cycle = prior.cycle
    _, old = prior.parent.admit_parent()
    check = cycle.admission.common._check_file
    check(PARENT/'result/RESULT.json', PARENT_SHA)
    check(PARENT/'VERIFICATION.json', VERIFY_SHA)
    result = json.loads((PARENT/'result/RESULT.json').read_text())
    verification = json.loads((PARENT/'VERIFICATION.json').read_text())
    provenance = json.loads((PARENT/'PROVENANCE.json').read_text())
    if (result['status'] != 'success' or (PARENT/'STATUS').read_text() != 'success\n'
            or verification['status'] != 'verified_same_direction_step_cap_diagnostic_not_SOS_or_GW'
            or verification['result_sha256'] != PARENT_SHA
            or provenance['status'] != 'success' or provenance['result_sha256'] != PARENT_SHA):
        raise ValueError('independent parent verification required')
    states = [line.split('|')[1:3] for line in verification['scheduler'].splitlines() if line]
    if not states or any(s != ['COMPLETED','0:0'] for s in states):
        raise ValueError('parent scheduler not successful')
    rows = [r for r in result['trials'] if r['radius'] == .04]
    if len(rows) != 1:
        raise ValueError('unique parent endpoint required')
    row = rows[0]
    validate_parent_trial(row)
    slot = PARENT/'result/radius_040'
    for name, key in (('COEFFICIENTS.txt','coefficient_sha256'),('C_3s3p2d.orb','orbital_sha256')):
        check(slot/name, row['proposal'][key])
    check(slot/'CANDIDATE.json', row['candidate_sha256'])
    if json.loads((slot/'CANDIDATE.json').read_text()) != row['proposal']:
        raise ValueError('parent proposal changed')
    sample = row['actual_pbe']
    for name, digest in sample['input_sha256'].items():
        if Path(name).name != name:
            raise ValueError('flat PBE inputs required')
        check(slot/'pbe'/name, digest)
    inputs, orbital_name, _ = cycle.frozen_pbe_bundle(old)
    expected = {name:cycle.endpoint._sha256(data) for name,data in inputs.items()}
    expected[orbital_name] = row['proposal']['orbital_sha256']
    if sample['input_sha256'] != expected:
        raise ValueError('parent must retain the exact frozen pure-PBE bundle')
    log = Path(sample['candidate_log_path'])
    try:
        log.resolve().relative_to((slot/'pbe').resolve())
    except ValueError:
        raise ValueError('parent SCF log outside endpoint')
    check(log, sample['candidate_log_sha256'])
    measured = cycle._strict_log(log.read_bytes())
    if (not math.isclose(measured['energy_ev'], sample['candidate_energy_ev'], abs_tol=1e-10, rel_tol=0)
            or not math.isclose((measured['energy_ev']-ORIGINAL_ENERGY_EV)/2,
                                sample['energy_delta_ev_per_c'], abs_tol=1e-12, rel_tol=0)):
        raise ValueError('parent actual PBE cannot reset')
    if any(result[k] != old['result'][k] for k in ('freeze_sha256','active_cache_index_sha256')):
        raise ValueError('reference/cache changed')
    return cycle, old, row


def verify_parent_source(stage, digest, commit):
    """Reuse the parent's declared-source policy without deleting Python caches."""
    stage=Path(stage).resolve()
    raw=(stage/'DEPLOYMENT.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('parent deployment SHA256 changed')
    manifest=json.loads(raw)
    if manifest['source_commit']!=commit or manifest['source_directory']!=str(stage/'source'):
        raise ValueError('parent source identity changed')
    name='SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/refresh_c_combined_step_gradient.py'
    if hashlib.sha256((stage/'source'/name).read_bytes()).hexdigest()!=manifest['files'][name]:
        raise ValueError('parent validator bytes changed')
    code="""import json,sys
from pathlib import Path
stage=Path(sys.argv[1]).resolve()
workflow=stage/'source/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow'
sys.path.insert(0,str(workflow))
import refresh_c_combined_step_gradient as common
assert Path(common.__file__).resolve()==workflow/'refresh_c_combined_step_gradient.py'
assert common.SOURCE==stage/'source'
print(json.dumps(common.verify_source(stage,sys.argv[2],sys.argv[3])))
"""
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',LC_ALL='C',LANG='C')
    checked=json.loads(subprocess.check_output([sys.executable,'-I','-B','-c',code,str(stage),digest,commit],env=env))
    if checked!=manifest:
        raise ValueError('parent source validator returned a different manifest')
    return checked


def continuation_scf_root(prior, actual, measurement_actual):
    if actual!=measurement_actual:
        raise ValueError('checkpoint measurement and accepted SCF record must be identical')
    root=Path(actual['candidate_log_path']).resolve().parents[1]
    root.relative_to((Path(prior)/'result').resolve())
    if root.name!='pbe':
        raise ValueError('one accepted candidate PBE input/log directory required')
    return root


def admit_continuation(result_sha, verification_sha):
    """Reuse the verified last center and secants, never repeat its SCF stencils."""
    if any(not re.fullmatch('[0-9a-f]{64}',h or '') for h in (result_sha,verification_sha)):
        raise ValueError('two explicit continuation SHA256 pins required')
    cycle,old,origin=admit_parent()
    prior=ROOT/'stage-c-consecutive-radials-7d8eef3c'
    check=cycle.admission.common._check_file
    check(prior/'result/RESULT.json',result_sha)
    check(prior/'VERIFICATION.json',verification_sha)
    result=json.loads((prior/'result/RESULT.json').read_text())
    audit=json.loads((prior/'VERIFICATION.json').read_text())
    provenance=json.loads((prior/'PROVENANCE.json').read_text())
    if ((prior/'STATUS').read_text()!='success\n'
            or audit['status']!='verified_consecutive_frozen_body_not_SOS_qavg_or_GW'
            or audit['result_sha256']!=result_sha or audit['job_id']!='21913349'
            or provenance['status']!='success' or provenance['job_id']!='21913349'
            or provenance['result_sha256']!=result_sha
            or result['source_commit']!='7d8eef3c155c771c082ffe5c41836e91c79a3ee4'):
        raise ValueError('independently verified consecutive parent required')
    rows=[line.split('|') for line in audit['scheduler'].splitlines() if line]
    if not rows or not any(r[0]=='21913349' for r in rows) or any(r[1:3]!=['COMPLETED','0:0'] for r in rows):
        raise ValueError('parent scheduler gate failed')
    verify_parent_source(prior,result['deployment_sha256'],result['source_commit'])
    checkpoints=sorted((prior/'result').glob('checkpoint_*/CHECKPOINT.json'))
    if not checkpoints:
        raise ValueError('continuation requires an accepted checkpoint')
    checkpoint=json.loads(checkpoints[-1].read_text())
    actual=checkpoint['actual_pbe']
    validate_continuation_payload(result,actual)
    for name,key in (('COEFFICIENTS.txt','coefficient_sha256'),('C_3s3p2d.orb','orbital_sha256')):
        check(prior/'result/FINAL'/name,result[key])
        check(checkpoints[-1].parent/name,result[key])
        if checkpoint[key]!=result[key]:
            raise ValueError('final must be the last accepted best checkpoint')
    checker=cycle.admission.refresh.accepted
    checker._record(result['final_candidate'],old['occupied_capture_floor'],old['quarter']['training_weights'])
    cycle.endpoint._match_parent_initial(result['final_candidate'],checkpoint['candidate'])
    checker._same_grid_reference(result['final_candidate'],origin['rpa']['candidate'])
    if any(result[k]!=old['result'][k] for k in ('freeze_sha256','active_cache_index_sha256')):
        raise ValueError('continuation reference/cache changed')
    log=Path(actual['candidate_log_path'])
    log.resolve().relative_to((prior/'result').resolve())
    check(log,actual['candidate_log_sha256'])
    measured=cycle._strict_log(log.read_bytes())
    if abs(measured['energy_ev']-actual['candidate_energy_ev'])>1e-10:
        raise ValueError('parent SCF energy changed')
    if abs((measured['energy_ev']-ORIGINAL_ENERGY_EV)/2-actual['energy_delta_ev_per_c'])>1e-12:
        raise ValueError('parent PBE origin changed')
    pbe_path=continuation_scf_root(prior,actual,checkpoint['measurement']['actual_pbe'])
    inputs,orbital_name,_=cycle.frozen_pbe_bundle(old)
    expected={name:cycle.endpoint._sha256(data) for name,data in inputs.items()}
    expected[orbital_name]=result['orbital_sha256']
    if expected!=actual['input_sha256']:
        raise ValueError('continuation pure-PBE inputs changed')
    for name,digest in actual['input_sha256'].items():
        if Path(name).name!=name:
            raise ValueError('flat PBE inputs required')
        check(pbe_path/name,digest)
    parent=dict(proposal=dict(coefficient_sha256=result['coefficient_sha256'],orbital_sha256=result['orbital_sha256']),
                actual_pbe=actual,rpa=dict(candidate=result['final_candidate']))
    restart=dict(coefficient_path=str(prior/'result/FINAL/COEFFICIENTS.txt'),result_sha256=result_sha,
                 verification_sha256=verification_sha,pbe_gradient=result['optimization']['pbe_gradient'])
    return cycle,old,parent,restart


def run(stage, source, cycle, old, parent, launcher, continuation=None):
    import numpy as np
    import torch
    from periodic_galerkin_consecutive import run_consecutive
    from periodic_galerkin_basis import write_periodic_optimizer_coefficients
    from periodic_galerkin_radial_diagnostics import evaluate_radial_gradients, radial_descent_direction, retract_displacement
    from periodic_galerkin_fit import _global_rpa_loss, _prepare_block_contraction_caches, CandidateGuardError
    from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference
    from periodic_galerkin_rpa import prepare_periodic_rpa_reference
    from run_c_combined_step import screen_candidate
    from export_periodic_orbitals import write_abacus_orbital

    start = time.perf_counter()
    out = stage/'result'; out.mkdir()
    rt = cycle.admission._runtime(); torch.set_num_threads(28)
    read = cycle.admission.common._read_coefficients
    initial_path=(PARENT/'result/radius_040/COEFFICIENTS.txt' if continuation is None
                  else Path(continuation['coefficient_path']))
    parent_sha=PARENT_SHA if continuation is None else continuation['result_sha256']
    verify_sha=VERIFY_SHA if continuation is None else continuation['verification_sha256']
    scope=SCOPE if continuation is None else 'common_descent_eight_radial_actual_PBE_constrained_frozen_body'
    initial = read(rt, initial_path)
    original = read(rt, old['original_coefficient_path'])
    shapes = [b.shape for b in initial['C']]
    if [list(s) for s in shapes] != [[31,n] for n in (3,3,2,0,0)]:
        raise ValueError('exact 248 all-free coefficients required')
    def flatten(c):
        return np.concatenate([b.detach().numpy().reshape(-1) for b in c['C']]).copy()
    def unflatten(x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (248,) or not np.isfinite(x).all():
            raise ValueError('finite full 248 coefficient vector required')
        blocks, offset = [], 0
        for shape in shapes:
            size = shape[0]*shape[1]
            blocks.append(torch.from_numpy(x[offset:offset+size].copy()).reshape(shape))
            offset += size
        return {'C':blocks}
    def project(x, vector):
        c, v = unflatten(x), unflatten(vector)
        return flatten({'C':[g-b@(b.T@g) for b,g in zip(c['C'],v['C'])]})
    def retract(x, displacement):
        return flatten(retract_displacement(unflatten(x),unflatten(displacement),1.))
    x0 = flatten(initial)
    load_start = time.perf_counter()
    datasets, records, guard = rt.load_frozen_c(old['freeze_path'],old['result']['freeze_sha256'],original,out,
        active_cache_index=old['active_cache_index_path'],
        active_cache_index_sha256=old['result']['active_cache_index_sha256'],band_guard_policy=cycle.RESPONSE)
    cycle.admission.common.validate_dataset_extent(datasets,old)
    with torch.no_grad():
        datasets = tuple(prepare_periodic_occupied_reference(d) for d in datasets)
        datasets = _prepare_block_contraction_caches(datasets,initial,1)
        reference = prepare_periodic_rpa_reference(datasets)
    load_seconds = time.perf_counter()-load_start
    floor, weights = old['occupied_capture_floor'], old['quarter']['training_weights']
    batch_size = old['quarter']['configuration']['frequency_batch_size']
    inputs, orbital_name, log_name = cycle.frozen_pbe_bundle(old)
    checker = cycle.admission.refresh.accepted
    write = cycle.endpoint._write_json
    sha = cycle.endpoint._sha256
    counts = dict(actual_scf=0, forward=0, backward=0)
    points, gradients, loss_cache = {}, [], {}
    def json_safe(value):
        if isinstance(value,np.ndarray):
            return value.tolist()
        if isinstance(value,np.generic):
            return value.item()
        if isinstance(value,dict):
            return {k:json_safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return [json_safe(v) for v in value]
        return value
    def export(slot, c):
        slot.mkdir()
        write_periodic_optimizer_coefficients(slot/'COEFFICIENTS.txt', c)
        write_abacus_orbital(slot/'C_3s3p2d.orb',c,element='C',ecut_ry=100.,rcut_bohr=10.,
                            dr_bohr=.01,smoothing_sigma_bohr=.1)
        return {key:sha((slot/name).read_bytes()) for name,key in
            (('COEFFICIENTS.txt','coefficient_sha256'),('C_3s3p2d.orb','orbital_sha256'))}
    def decorate(record,c):
        record = dict(record)
        record['coefficient_guard'] = guard(c)
        record['energy_quantity'] = 'frozen_body_RPA_correlation_not_PBE_total'
        for prefix,key in (('rpa','candidate_energy_ha'),('reference_rpa','reference_energy_ha')):
            for suffix,divisor in (('cell',1),('c',2)):
                record[prefix+'_correlation_energy_ev_per_'+suffix] = record['rpa'][key]*cycle.endpoint.HARTREE_TO_EV/divisor
        checker._record(record,floor,weights)
        checker._same_grid_reference(record,parent['rpa']['candidate'])
        return record
    def fresh_gradient(x):
        c = unflatten(x)
        diagnostic = evaluate_radial_gradients(datasets,c,occupied_capture_tolerance=1-floor,
            frequency_batch_size=batch_size,weights=weights,energy_only=continuation is None)
        counts['forward'] += 1; counts['backward'] += 1 if continuation is None else 2
        record = decorate(diagnostic,c)
        vector = flatten({'C':[torch.tensor(r['horizontal_gradient'],dtype=torch.float64)
                              for r in diagnostic['energy_gradient']['channels']]})
        if record['rpa']['candidate_energy_ha'] < record['rpa']['reference_energy_ha']:
            vector *= -1
        if continuation is not None:
            loss_cache.update(x=x.copy(), vector=flatten({'C':[
                torch.tensor(r['horizontal_gradient'],dtype=torch.float64)
                for r in diagnostic['loss_gradient']['channels']]}))
        i = len(gradients)
        slot = out/('gradient_%03d'%i)
        hashes = export(slot,c)
        write(slot/'GRADIENT.json',dict(record,**hashes))
        gradients.append(dict(index=i,**hashes,gradient_sha256=sha((slot/'GRADIENT.json').read_bytes())))
        print(json.dumps(dict(gradient=i,body_error_ev_per_c=controller_record(record,0)['objective']*
            cycle.endpoint.HARTREE_TO_EV/2,seconds=diagnostic['forward_and_backward_seconds'])),flush=True)
        return vector,record

    g0, first = fresh_gradient(x0)
    cycle.endpoint._match_parent_initial(first,parent['rpa']['candidate'])
    checker._same_grid_reference(first,parent['rpa']['candidate'],reproduce=True)
    export(out/'INITIAL',initial)
    for name,key in (('COEFFICIENTS.txt','coefficient_sha256'),('C_3s3p2d.orb','orbital_sha256')):
        if sha((out/'INITIAL'/name).read_bytes()) != parent['proposal'][key]:
            raise ValueError('parent export changed')

    def measure_named(x, name):
        c,slot = unflatten(x),out/name
        hashes = export(slot,c)
        try:
            with torch.no_grad():
                bands,capture = screen_candidate(datasets,c,guard,floor)
        except CandidateGuardError as error:
            sample = dict(gate='cheap_reject',reason=str(error),**hashes)
            write(slot/'MEASUREMENT.json',sample)
            return sample
        proposal = dict(scope=scope,name=name,nu=[3,3,2,0,0],fixed_nu=[0]*5,ao_per_C=22,
            cheap_gate=bands,capture=capture,occupied_capture_floor=floor,
            parent_result_sha256=parent_sha,source_commit=source['source_commit'],
            physical_release_gate='hold',**hashes)
        write(slot/'CANDIDATE.json',proposal)
        newinputs = dict(inputs); newinputs[orbital_name]=(slot/'C_3s3p2d.orb').read_bytes()
        counts['actual_scf'] += 1
        try:
            actual = cycle.run_scf(slot/'pbe',newinputs,log_name,launcher)
        except ValueError as error:
            log=slot/'pbe'/log_name
            if not log.is_file() or not candidate_nonconvergence(error,log.read_bytes()):
                raise
            for input_name,data in newinputs.items():
                cycle.admission.common._check_file(slot/'pbe'/input_name,sha(data))
            sample=dict(gate='fail',actual_scf_gate='rejected_nonconvergence',
                scf_log_path=str(log),scf_log_sha256=sha(log.read_bytes()),
                input_sha256={name:sha(data) for name,data in newinputs.items()},
                name=name,candidate_sha256=sha((slot/'CANDIDATE.json').read_bytes()),**hashes)
            write(slot/'MEASUREMENT.json',sample)
            return sample
        sample = dict(gate=actual['pbe_gate'],pbe=actual['energy_delta_ev_per_c'],actual_pbe=actual,
            name=name,candidate_sha256=sha((slot/'CANDIDATE.json').read_bytes()),**hashes)
        points[name] = dict(x=x.copy(),sample=sample)
        write(slot/'MEASUREMENT.json',sample)
        print(json.dumps(dict(measured=name,pbe_mev_per_c=1000*sample['pbe'],gate=sample['gate'])),flush=True)
        return sample

    # Initial directional calibration; subsequent secants live in all coefficient
    # coordinates, not the changing eight-dimensional radial-gradient frame.
    if continuation is None:
        directions,slopes = [],[]
        for l,n in enumerate((3,3,2)):
            for z in range(n):
                d = radial_descent_direction(initial,first['energy_gradient'],'C',l,z)
                if d is None:
                    raise ValueError('initial calibration lacks a radial derivative')
                vector = flatten(d); energies=[]
                for sign,label in ((-1,'minus'),(1,'plus')):
                    sample = measure_named(retract(x0,sign*.001*vector),'calibration_%d_%d_%s'%(l,z,label))
                    if sample['gate'] == 'cheap_reject' or 'pbe' not in sample:
                        raise ValueError('initial PBE derivative stencil failed cheap guard')
                    energies.append(sample['pbe'])
                directions.append(vector); slopes.append((energies[1]-energies[0])/.002)
        basis = np.column_stack(directions)
        if not np.allclose(basis.T@basis,np.eye(8),atol=1e-10,rtol=0):
            raise ValueError('initial radial calibration basis not orthonormal')
        pbe_gradient = basis@np.asarray(slopes)
        model=dict(gradient=pbe_gradient.tolist(),radial_slopes=slopes,measured_span_rank=8)
    else:
        pbe_gradient=np.asarray(continuation['pbe_gradient'],dtype=np.float64)
        model=dict(gradient=pbe_gradient.tolist(),reused_from_result_sha256=parent_sha,
                   calibration_scf_count=0,model='ambient_secant_not_exact_PBE_derivative')
    write(out/'PBE_MODEL_INITIAL.json',dict(model,
        coordinate_metric='Euclidean_coefficient_signed_QR_frame',
        complete_PBE_derivative=False,actual_SCF_required=True))
    first_gradient = [g0]
    current_record = [first]
    def gradient(x):
        if first_gradient:
            return first_gradient.pop()
        g,record = fresh_gradient(x)
        cycle.endpoint._match_parent_initial(record,current_record[0])
        current_record[0]=record
        return g
    def loss_gradient(x):
        if not np.array_equal(x,loss_cache.get('x')):
            raise ValueError('loss derivative must share the freshly evaluated coefficient center')
        return loss_cache['vector'].copy()
    def measure(x,trial_id):
        return measure_named(x,'trial_%03d'%trial_id)
    def evaluate(x,trial_id):
        name='trial_%03d'%trial_id; point=points[name]; c=unflatten(x)
        if not np.array_equal(x,point['x']) or point['sample']['gate']!='pass':
            raise ValueError('actual PBE must precede the exact response candidate')
        with torch.no_grad():
            loss,capture,condition,_,rpa = _global_rpa_loss(datasets,c,occupied_capture_tolerance=1-floor,
                weights=weights,frequency_batch_size=batch_size,reference_cache=reference)
        counts['forward'] += 1
        record=decorate(dict(loss=float(loss),minimum_occupied_capture=capture,
                             maximum_overlap_condition=condition,rpa=rpa),c)
        write(out/name/'GALERKIN.json',record); point['record']=record
        return controller_record(record,point['sample']['pbe'])
    def checkpoint(state):
        accepted = [p for p in points.values() if np.array_equal(p['x'],state['x']) and 'record' in p]
        if len(accepted)!=1:
            raise ValueError('checkpoint must identify a unique measured and evaluated trial')
        point=accepted[0]; current_record[0]=point['record']
        slot=out/('checkpoint_%03d'%state['counts']['accepted_steps'])
        hashes=export(slot,unflatten(state['x']))
        payload=json_safe(state)
        payload.update(candidate=point['record'],actual_pbe=point['sample']['actual_pbe'],
                       measurement=point['sample'],**hashes,physical_release_gate='hold')
        write(slot/'CHECKPOINT.json',payload)
        print(json.dumps(dict(accepted_step=state['counts']['accepted_steps'],trial_count=state['counts']['trials'],
            body_error_ev_per_c=state['record']['objective']*cycle.endpoint.HARTREE_TO_EV/2,
            pbe_mev_per_c=1000*state['record']['pbe'],radius=state['radius'])),flush=True)
    settings=(dict(max_steps=20,max_trials=80,initial_radius=.02,max_radius=.04) if continuation is None
              else dict(max_steps=12,max_trials=40,initial_radius=.001,max_radius=.01,
                        loss_gradient=loss_gradient))
    state=run_consecutive(x0,controller_record(first,parent['actual_pbe']['energy_delta_ev_per_c']),
        pbe_gradient,gradient=gradient,retract=retract,project=project,measure=measure,evaluate=evaluate,
        checkpoint=checkpoint,**settings)
    hashes=export(out/'FINAL',unflatten(state['x']))
    if gradients[-1]['coefficient_sha256']!=hashes['coefficient_sha256']:
        _,terminal=fresh_gradient(state['x'])
        cycle.endpoint._match_parent_initial(terminal,current_record[0])
        current_record[0]=terminal
    state=json_safe(state)
    return dict(status='success',scope=scope,optimization=state,initial=first,initial_actual_pbe=parent['actual_pbe'],
        gradients=gradients,
        counts=counts,final_candidate=current_record[0],**hashes,parent_result_sha256=parent_sha,
        parent_verification_sha256=verify_sha,source_commit=source['source_commit'],
        freeze_sha256=old['result']['freeze_sha256'],active_cache_index_sha256=old['result']['active_cache_index_sha256'],
        original_coefficient_path=str(old['original_coefficient_path']),occupied_capture_floor=floor,
        cache_loads=1,cache_load_seconds=load_seconds,load_records=records,total_seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,nu=[3,3,2,0,0],fixed_nu=[0]*5,
        ao_per_C=22,delta_st_runs=0,ordinary_sos='not_run',gw='not_run',physical_release_gate='hold',
        calibration_scf_count=16 if continuation is None else 0,
        common_descent=continuation is not None,
        full_constrained_stationarity='not_established')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','source-commit','deployment-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--continuation-result-sha256')
    p.add_argument('--continuation-verification-sha256')
    args=p.parse_args(); stage=Path(args.stage).resolve()
    restart=None
    if args.continuation_result_sha256 or args.continuation_verification_sha256:
        cycle,old,parent,restart=admit_continuation(args.continuation_result_sha256,args.continuation_verification_sha256)
    else:
        cycle,old,parent=admit_parent()
    source=cycle.admission.common.verify_source(stage,args.deployment_sha256,args.source_commit)
    cycle.validate_stage(stage,source,old)
    required=('run_c_consecutive_radials.py','run_c_consecutive_radials.slurm')
    if any(cycle.admission.refresh._WORKFLOW+n not in source['files'] for n in required):
        raise ValueError('source pins missing')
    runtime=old['result']['actual_pbe']
    for pkey,hkey in (('abacus_binary','abacus_sha256'),('mpi_library','mpi_sha256')):
        cycle.admission.common._check_file(runtime[pkey],runtime[hkey])
    if (os.environ.get('SLURM_JOB_NUM_NODES')!='1' or os.environ.get('SLURM_NTASKS')!='4'
            or os.environ.get('OMP_NUM_THREADS')!='7'
            or Path(os.environ.get('I_MPI_PMI_LIBRARY','/missing')).resolve()!=Path(runtime['mpi_library']).resolve()):
        raise ValueError('frozen one node/four ranks/seven OMP layout required')
    cycle.frozen_pbe_bundle(old)
    if args.preflight_only:
        return
    reservation_key=(PARENT_SHA[:16]+'-v1' if restart is None else restart['result_sha256'][:16]+'-common-v1')
    reservation=ROOT/('consecutive-radials-'+reservation_key)
    reservation.mkdir()
    cycle.endpoint._write_json(reservation/'RESERVATION.json',dict(stage=str(stage),
        scope=SCOPE if restart is None else 'common_descent_eight_radial_actual_PBE_constrained_frozen_body',
        parent_sha256=PARENT_SHA if restart is None else restart['result_sha256'],job_id=os.environ['SLURM_JOB_ID'],
        maximum_accepted_steps=20 if restart is None else 12,maximum_trials=80 if restart is None else 40,
        calibration_scf_count=16 if restart is None else 0))
    result=run(stage,source,cycle,old,parent,['srun','--mpi=pmi2','--cpu-bind=none','-n','4',runtime['abacus_binary']],
               continuation=restart)
    if restart is None:
        admit_parent()
    else:
        admit_continuation(args.continuation_result_sha256,args.continuation_verification_sha256)
    cycle.admission.common.verify_source(stage,args.deployment_sha256,args.source_commit)
    for pkey,hkey in (('abacus_binary','abacus_sha256'),('mpi_library','mpi_sha256')):
        cycle.admission.common._check_file(runtime[pkey],runtime[hkey])
    result['deployment_sha256']=args.deployment_sha256
    cycle.endpoint._write_json(stage/'result/RESULT.json',result)
    cycle.endpoint._write_json(stage/'PROVENANCE.json',dict(status='success',source_commit=args.source_commit,
        deployment_sha256=args.deployment_sha256,result_sha256=cycle.endpoint._sha256((stage/'result/RESULT.json').read_bytes()),
        job_id=os.environ['SLURM_JOB_ID'],abacus_sha256=runtime['abacus_sha256'],pmi_sha256=runtime['mpi_sha256'],
        physical_release_gate='hold'))


if __name__=='__main__':
    main()
