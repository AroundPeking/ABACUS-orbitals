"""Bounded df-only optimizer comparison on the accepted, relocated C cache.

This does not regenerate response, run SCF, or release a physical basis. Both
arms start at the same free endpoint; only the optimization algorithm changes.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from c_portable_frozen_replay import (FREEZE_SHA, INDEX_SHA, PARENTS, INDICES, MULT,
    PHYSICAL_RELEASE, check_settings, check_replay, check_q_replay, hashed,
    require, validate_cache_contract, within, write)

BUNDLE_SHA = 'a0e285e12c05d1138f14f9e2e94b9a67641c25aacd6344eba70c9a0ed442a215'
REFERENCE = -0.5144842779929139
EV_PER_HA_PER_C = 27.211386245988/2


def objective_and_sign(energy, reference):
    require(math.isfinite(energy) and math.isfinite(reference)
            and abs(reference-REFERENCE) <= 1e-12, 'finite frozen reference required')
    delta = energy-reference
    return abs(delta), (1. if delta > 0 else -1. if delta < 0 else 0.)


def validate_budget(steps, forwards):
    require(type(steps) is int and 0 < steps <= 20
            and type(forwards) is int and 0 < forwards <= 40, 'bounded comparison required')


def execution_policy(continuation,steps,forwards):
    if not continuation:
        validate_budget(steps,forwards)
        return ('steepest','lbfgs')
    require(type(steps) is int and 0<steps<=60 and type(forwards) is int
            and 0<forwards<=120,'bounded continuation required')
    return ('lbfgs',)


def consecutive_evaluator(unflatten,evaluate,record):
    def evaluate_point(x,trial_id):
        return record(evaluate(unflatten(x)))
    return evaluate_point


def validate_recovery(r):
    require(r.get('status')=='failed_before_any_optimizer_evaluation'
            and r.get('job_id')=='3330163'
            and r.get('source_commit')=='5f8a50fd327319848bc708e2beae76c1de2d0c62'
            and r.get('bundle_sha256')==BUNDLE_SHA
            and r.get('scheduler_state')=='FAILED' and r.get('exit_code')=='1:0'
            and r.get('accepted_steps')==0 and r.get('evaluations')==0,
            'only the failed zero-step adapter attempt can be recovered')
    sha=r.get('shared_gradient_sha256','')
    require(len(sha)==64 and all(c in '0123456789abcdef' for c in sha),'recovery gradient hash required')


def parent_scheduler_state(text):
    rows=[row.strip().split('|') for row in text.splitlines() if row.strip()]
    require(rows==[['3330163','FAILED','1:0']], 'exact terminal FAILED/1:0 parent required')
    return rows[0][1],rows[0][2]


def prepare_recovery(parent):
    root=Path('/data/home/df_iopcas_ghj/app/siab/c-solid-rpa-continuation-20260912')
    require(parent==root/'optimizer-comparison-5f8a50fd','exact recovery directory required')
    state,exit_code=parent_scheduler_state(subprocess.check_output([
        'sacct','-n','-P','-X','-j','3330163','--format=JobID,State,ExitCode'],text=True))
    hashed(root/'source-5f8a50fd/SOURCE_MANIFEST.json',
           '710f1558b634d4e01fda318517fe17dbf2721970c9dd25a48e7ed78b0635b5fb')
    submission=json.loads((parent/'SUBMISSION.json').read_text())
    require(submission['job_id']=='3330163' and submission['bundle_sha256']==BUNDLE_SHA,
            'parent submission identity changed')
    error=(parent/'slurm-3330163.err').read_text()
    require('TypeError' in error and 'takes 1 positional argument but 2 were given' in error,
            'only the known zero-step callback failure can be recovered')
    require(not list((parent/'result').glob('*/checkpoint_*'))
            and not list((parent/'result').glob('*/evaluation_*')),
            'existing optimization work must not be replayed')
    gradient=parent/'result/SHARED_INITIAL_GRADIENT.json'
    data=gradient.read_bytes(); d=json.loads(data)
    require(abs(d['rpa']['candidate_energy_ha']-(-.43346679493776147))<=1e-9,
            'recovery initial energy changed')
    objective_and_sign(d['rpa']['candidate_energy_ha'],d['rpa']['reference_energy_ha'])
    receipt=dict(status='failed_before_any_optimizer_evaluation',job_id='3330163',
        source_commit=submission['source_commit'],bundle_sha256=BUNDLE_SHA,
        shared_gradient_sha256=hashlib.sha256(data).hexdigest(),scheduler_state=state,
        exit_code=exit_code,accepted_steps=0,evaluations=0)
    validate_recovery(receipt)
    path=parent/'RECOVERY.json'
    if path.exists(): require(json.loads(path.read_text())==receipt,'recovery receipt changed')
    else: write(path,receipt)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(bundle, output, source_commit, max_steps=12, max_forwards=24,
        recovery_root=None,recovery_sha256=None,continuation_root=None,continuation_sha256=None):
    import numpy as np
    import scipy
    import torch
    here = Path(__file__).resolve()
    repo = here.parents[4]
    sys.path.insert(0, str(repo/'SIAB/opt_orb_pytorch_dpsi'))
    from periodic_galerkin_basis import (read_periodic_optimizer_coefficients,
                                         write_periodic_optimizer_coefficients)
    from periodic_galerkin_dataset_cache import read_periodic_galerkin_dataset_cache
    from periodic_galerkin_fit import _prepare_block_contraction_caches, _global_rpa_loss, CandidateGuardError
    from periodic_galerkin_radial_diagnostics import evaluate_radial_gradients, retract_displacement
    from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference
    from periodic_galerkin_rpa import prepare_periodic_rpa_reference
    from periodic_galerkin_consecutive import run_consecutive
    from periodic_galerkin_lbfgs import GrassmannChart, EvaluationBudget, run_lbfgs
    from c_response_band_policy import prepare_c_band_guard, RESPONSE
    from run_c_combined_step import screen_candidate
    from optimize_c_all_radial_fast import _validate_c_dataset

    def safe(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {k:safe(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [safe(v) for v in value]
        return value

    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    start = time.perf_counter()
    arm_names=execution_policy(continuation_root is not None,max_steps,max_forwards)
    require(not (continuation_root is not None and recovery_root is not None),
            'continuation and failed-zero-step recovery are distinct')
    require(len(source_commit) == 40 and all(c in '0123456789abcdef' for c in source_commit),
            'immutable source commit required')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NUM_NODES') == '1'
            and os.environ.get('C_EXECUTION_HOST') == 'df_iopcas_ghj', 'single-node df job required')
    source_manifest = json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    require(source_manifest['commit'] == source_commit, 'source manifest commit mismatch')
    for name, sha in source_manifest['files'].items(): hashed(within(repo, name), sha)
    manifest = json.loads(hashed(bundle/'BUNDLE.json', BUNDLE_SHA))
    require(manifest['execution_host'] == 'df_iopcas_ghj', 'df-only bundle required')
    settings = manifest['evaluation_settings']; check_settings(settings)
    for name, sha in manifest['files'].items(): hashed(within(bundle, name), sha)
    # Existing numerical kernels must remain byte-identical to the verified replay.
    for name, sha in manifest['source_files'].items(): hashed(within(repo, name), sha)
    recovery=None
    if recovery_root is not None:
        require(recovery_root==bundle.parent/'optimizer-comparison-5f8a50fd', 'exact recovery parent required')
        recovery=json.loads(hashed(recovery_root/'RECOVERY.json',recovery_sha256))
        validate_recovery(recovery)
        require(not list((recovery_root/'result').glob('*/checkpoint_*'))
                and not list((recovery_root/'result').glob('*/evaluation_*')),
                'cannot replay an optimizer arm with completed evaluations')
    frozen = json.loads(hashed(bundle/'backoff_INPUT_FREEZE.json', FREEZE_SHA))
    index = json.loads(hashed(bundle/'backoff_ACTIVE_DATA_CACHE.json', INDEX_SHA))
    records = validate_cache_contract(frozen, index)
    read = lambda p: read_periodic_optimizer_coefficients(p, element='C', radial_rows=31,
                                                         expected_nu=(3,3,2,0,0))
    parent = json.loads(hashed(bundle/'energy_free/RESULT.json', PARENTS['energy_free']))
    cpath = bundle/'energy_free/FINAL/COEFFICIENTS.txt'
    hashed(cpath, parent['coefficient_sha256'])
    original = read(bundle/'backoff_ORIGINAL_COEFFICIENTS.txt')
    old = json.loads(hashed(bundle/'energy_free/gradient_020/GRADIENT.json',
                           parent['gradients'][-1]['gradient_sha256']))
    require(old['coefficient_sha256'] == parent['coefficient_sha256'], 'terminal center mismatch')
    continuation=None
    if continuation_root is not None:
        from c_optimizer_continuation import admit_parent
        require(continuation_root.parent==bundle.parent and continuation_root!=output.parent,
                'continuation must use a distinct sibling df run')
        continuation=admit_parent(continuation_root,continuation_sha256)
        cpath=Path(continuation['coefficient_path'])
        old=json.loads(hashed(Path(continuation['gradient_path']),continuation['gradient_sha256']))['diagnostic']
        parent=dict(coefficient_sha256=continuation['coefficient_sha256'],
                    final_candidate={'rpa':old['rpa']})
    initial = read(cpath)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    output.mkdir()
    if continuation is not None: write(output/'CONTINUATION_PARENT.json',continuation)
    datasets = []
    for item, rec, iq, mult in zip(frozen['datasets'], records, INDICES, MULT):
        d = read_periodic_galerkin_dataset_cache(
            within(bundle, manifest['cache_paths'][str(item['label'])]),
            cache_sha256=rec['complete_sha256'], active_coefficients=original, **rec['binding'])
        _validate_c_dataset(d, item, iq, mult); datasets.append(d)
        print(json.dumps(dict(loaded_q=item['label'], seconds=time.perf_counter()-start)), flush=True)
    guard = prepare_c_band_guard(datasets[0], original, policy=RESPONSE)
    with torch.no_grad():
        datasets = tuple(prepare_periodic_occupied_reference(d) for d in datasets)
        datasets = _prepare_block_contraction_caches(datasets, original, 1)
        ref = prepare_periodic_rpa_reference(datasets)
    load_seconds = time.perf_counter()-start
    shapes = [b.shape for b in initial['C']]
    require([list(s) for s in shapes] == [[31,k] for k in (3,3,2,0,0)], 'all 248 coefficients required')
    def flatten(c): return np.concatenate([b.detach().numpy().ravel() for b in c['C']]).copy()
    def unflatten(x):
        x = np.asarray(x, dtype=float)
        require(x.shape == (248,) and np.isfinite(x).all(), 'finite full coefficient vector required')
        blocks=[]; offset=0
        for n,k in shapes:
            blocks.append(torch.tensor(x[offset:offset+n*k].reshape(n,k), dtype=torch.float64))
            offset+=n*k
        return {'C':blocks}
    def raw(d, key='raw_gradient'):
        return {'C':[torch.tensor(b[key], dtype=torch.float64).reshape(b['shape'])
                     for b in d['energy_gradient']['channels']]}
    def screen(c):
        with torch.no_grad():
            return screen_candidate(datasets,c,lambda v:guard(v,enforce_accuracy=False),
                                     1e-12,enforce_band_accuracy=False)
    def gradient_kernel(c):
        return evaluate_radial_gradients(datasets,c,occupied_capture_tolerance=1-1e-12,
                                         energy_only=True,**settings)
    def energy_kernel(c):
        with torch.no_grad():
            loss,capture,condition,_,rpa = _global_rpa_loss(datasets,c,
                occupied_capture_tolerance=1-1e-12,reference_cache=ref,**settings)
        return dict(loss=float(loss),rpa=rpa,minimum_occupied_capture=capture,
                    maximum_condition_number=condition)
    def record(d):
        objective, _ = objective_and_sign(d['rpa']['candidate_energy_ha'],d['rpa']['reference_energy_ha'])
        return dict(objective=objective,loss=float(d['loss']),pbe=None,rpa=d['rpa'])

    screen(initial)
    shared_start = time.perf_counter()
    shared = (old if continuation is not None else gradient_kernel(initial) if recovery is None
              else json.loads(hashed(recovery_root/'result/SHARED_INITIAL_GRADIENT.json',
                                     recovery['shared_gradient_sha256'])))
    check_replay(shared['rpa']['candidate_energy_ha'],parent['final_candidate']['rpa']['candidate_energy_ha'],
                 flatten(raw(shared,'horizontal_gradient')),flatten(raw(old,'horizontal_gradient')),
                 shared['rpa']['reference_energy_ha'],REFERENCE)
    check_q_replay(shared['rpa'],old['rpa'])
    write(output/'SHARED_INITIAL_GRADIENT.json',shared)
    shared_seconds=time.perf_counter()-shared_start
    initial_record=record(shared)
    initial_x=flatten(initial)
    chart=GrassmannChart(initial)
    require(chart.dimension == 226,'full Grassmann coordinate dimension required')
    write(output/'CHART.json',chart.definition())
    chart_zero=chart.coefficients(np.zeros(chart.dimension))
    require(np.max(np.abs(flatten(chart_zero)-initial_x)) < 1e-12, 'chart origin changed')
    arms={}

    for arm in arm_names:
        arm_start=time.perf_counter(); root=output/arm; root.mkdir()
        counts=dict(forward=0,backward=0,cheap_screen=0,cache_hits=0)
        receipts=[]; accepted=[]
        cache={initial_x.tobytes():shared}
        screened={initial_x.tobytes()}
        def ensure_screen(c):
            key=flatten(c).tobytes()
            if key not in screened:
                counts['cheap_screen']+=1
                screen(c)
                screened.add(key)
        def evaluate(c, need_gradient=False):
            key=flatten(c).tobytes()
            prior=cache.get(key)
            if prior is not None and (not need_gradient or 'energy_gradient' in prior):
                counts['cache_hits']+=1
                return prior
            if counts['forward'] >= max_forwards: raise EvaluationBudget('kernel forward budget')
            t=time.perf_counter()
            ensure_screen(c)
            counts['forward']+=1
            if need_gradient: counts['backward']+=1
            d=gradient_kernel(c) if need_gradient else energy_kernel(c)
            record(d)
            slot=root/('evaluation_%03d'%counts['forward']); slot.mkdir()
            write_periodic_optimizer_coefficients(slot/'COEFFICIENTS.txt',c)
            row=dict(kind='forward_backward' if need_gradient else 'forward',
                     evaluation=counts['forward'],coefficient_sha256=digest(slot/'COEFFICIENTS.txt'),
                     seconds=time.perf_counter()-t,arm_seconds=time.perf_counter()-arm_start,
                     counts=counts.copy(),diagnostic=d)
            write(slot/'EVALUATION.json',row); receipts.append({k:v for k,v in row.items() if k!='diagnostic'})
            cache[key]=d
            print(json.dumps(dict(arm=arm,evaluation=counts['forward'],backwards=counts['backward'],
                ec=d['rpa']['candidate_energy_ha'],error_ev_per_c=record(d)['objective']*EV_PER_HA_PER_C,
                seconds=row['seconds'])),flush=True)
            return d
        last=dict(x=initial_x.copy(),record=initial_record,accepted_steps=0)
        def checkpoint(c, state):
            nonlocal last
            d=evaluate(c)
            r=record(d)
            require(r['objective'] <= last['record']['objective']+1e-13,'accepted objective increased')
            slot=root/('checkpoint_%03d'%(len(accepted)+1)); slot.mkdir()
            write_periodic_optimizer_coefficients(slot/'COEFFICIENTS.txt',c)
            row=dict(accepted_steps=len(accepted)+1,record=r,counts=counts.copy(),
                     coefficient_sha256=digest(slot/'COEFFICIENTS.txt'),
                     arm_seconds=time.perf_counter()-arm_start,optimizer=safe(state))
            write(slot/'CHECKPOINT.json',row); accepted.append(row)
            last=dict(x=flatten(c),record=r,accepted_steps=len(accepted),path=str(slot/'COEFFICIENTS.txt'))
            print(json.dumps(dict(arm=arm,accepted=len(accepted),error_ev_per_c=r['objective']*EV_PER_HA_PER_C)),flush=True)
        if arm == 'steepest':
            def grad(x):
                d=evaluate(unflatten(x),True)
                return objective_and_sign(d['rpa']['candidate_energy_ha'],REFERENCE)[1]*flatten(raw(d))
            def project(x,v):
                c,g=unflatten(x),unflatten(v)
                return flatten({'C':[b-a@(a.T@b) for a,b in zip(c['C'],g['C'])]})
            def measure(x,trial):
                try: ensure_screen(unflatten(x))
                except CandidateGuardError as e: return dict(gate='fail',reason=str(e))
                return dict(gate='pass',pbe=None)
            try:
                state=run_consecutive(initial_x,initial_record,np.zeros(248),gradient=grad,
                    retract=lambda x,v:flatten(retract_displacement(unflatten(x),unflatten(v),1.)),
                    project=project,measure=measure,evaluate=consecutive_evaluator(unflatten,evaluate,record),
                    checkpoint=lambda s:checkpoint(unflatten(s['x']),s),max_steps=max_steps,
                    max_trials=max_forwards*3,initial_radius=parent['optimization']['radius'],
                    max_radius=.04,min_radius=1e-6,enforce_pbe=False,require_nonincreasing_loss=False)
            except EvaluationBudget:
                state=dict(status='budget_stop',stop_reason='max_evaluations')
        else:
            # A fixed chart means L-BFGS history lives in one coordinate system.
            def fg(z):
                c=chart.coefficients(z)
                # Avoid QR rounding at z=0 causing a second initial kernel call.
                d=shared if not np.any(z) else evaluate(c,True)
                f,sign=objective_and_sign(d['rpa']['candidate_energy_ha'],REFERENCE)
                return f,sign*chart.pullback(z,raw(d))
            state=run_lbfgs(np.zeros(chart.dimension),fg,
                lambda s:checkpoint(chart.coefficients(s['x']),s),max_steps=max_steps,
                max_evaluations=max_forwards+1,coordinate_bound=.05)
        final=root/'FINAL'; final.mkdir()
        if accepted: shutil.copyfile(last['path'],final/'COEFFICIENTS.txt')
        else: shutil.copyfile(cpath,final/'COEFFICIENTS.txt')
        final_sha=digest(final/'COEFFICIENTS.txt')
        require(final_sha == (accepted[-1]['coefficient_sha256'] if accepted else parent['coefficient_sha256']),
                'FINAL must equal accepted best checkpoint')
        elapsed=time.perf_counter()-arm_start
        gain=(initial_record['objective']-last['record']['objective'])*EV_PER_HA_PER_C*1000
        terminal=cache.get(last['x'].tobytes(),{})
        terminal_gradient_norm=(terminal['energy_gradient']['horizontal_gradient_norm']
                                if 'energy_gradient' in terminal else None)
        result=dict(status='success',scope='optimizer_diagnostic_not_physical_basis',
            algorithm=arm,optimizer=safe(state),counts=counts,accepted=accepted,evaluations=receipts,
            initial_record=initial_record,final_record=last['record'],coefficient_sha256=final_sha,
            accepted_steps=len(accepted),seconds=elapsed,gain_mev_per_c=gain,
            gain_mev_per_c_per_minute=gain/(elapsed/60),physical_release_gate=PHYSICAL_RELEASE,
            terminal_horizontal_gradient_norm=terminal_gradient_norm,
            terminal_gradient_status='evaluated' if terminal_gradient_norm is not None else 'not_evaluated_within_budget',
            terminal_pbe='not_recomputed_parent_already_fails_10mev',scf_runs=0)
        write(root/'RESULT.json',result); arms[arm]=result
        (root/'STATUS').write_text('success\n')

    fresh_shared=recovery is None and continuation is None
    result=dict(status='success',scope='lbfgs_continuation' if continuation else 'bounded_optimizer_comparison',arms=arms,
        physical_release_gate=PHYSICAL_RELEASE,reference_energy_ha=REFERENCE,
        parent_result_sha256=continuation_sha256 if continuation else PARENTS['energy_free'],bundle_sha256=BUNDLE_SHA,
        source_commit=source_commit,source_manifest_sha256=digest(repo/'SOURCE_MANIFEST.json'),
        chart_sha256=digest(output/'CHART.json'),
        torch_version=torch.__version__,numpy_version=np.__version__,scipy_version=scipy.__version__,
        max_steps=max_steps,max_forwards_per_arm=max_forwards,chart_dimension=226,
        coefficient_count=248,chart_coordinate_bound=.05,load_seconds=load_seconds,
        shared_initial_seconds=shared_seconds,shared_initial_forward=int(fresh_shared),
        shared_initial_backward=int(fresh_shared),recovery=recovery,continuation=continuation,
        curvature_history='restarted_at_verified_parent' if continuation else 'fresh_comparison',
        job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        delta_st_runs=0,scf_runs=0,sos_runs=0)
    write(output/'RESULT.json',result)
    (output/'STATUS').write_text('success\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--source-commit')
    p.add_argument('--prepare-recovery',type=Path)
    p.add_argument('--recovery-root',type=Path)
    p.add_argument('--recovery-sha256')
    p.add_argument('--continuation-root',type=Path)
    p.add_argument('--continuation-sha256')
    p.add_argument('--max-steps',type=int,default=12)
    p.add_argument('--max-forwards',type=int,default=24)
    args=p.parse_args()
    if args.prepare_recovery:
        print(prepare_recovery(args.prepare_recovery.resolve()))
        return
    if args.bundle is None or args.output is None or args.source_commit is None:
        p.error('--bundle, --output and --source-commit are required for optimization')
    run(args.bundle.resolve(),args.output.resolve(),args.source_commit,
        max_steps=args.max_steps,max_forwards=args.max_forwards,
        recovery_root=args.recovery_root.resolve() if args.recovery_root else None,
        recovery_sha256=args.recovery_sha256,
        continuation_root=args.continuation_root.resolve() if args.continuation_root else None,
        continuation_sha256=args.continuation_sha256)


if __name__ == '__main__': main()
