"""Read-only numerical admission for a completed frozen-body optimizer parent."""

import hashlib
import json
import math
from pathlib import Path
import subprocess

from c_portable_frozen_replay import INDICES, MULT, hashed, require, within

REFERENCE=-0.5144842779929139
BUNDLE_SHA='a0e285e12c05d1138f14f9e2e94b9a67641c25aacd6344eba70c9a0ed442a215'
CONVERSION=27.211386245988/2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_state(text,job):
    require(job.isdigit(), 'numeric parent job required')
    rows=[s.strip().split('|') for s in text.splitlines() if s.strip()]
    require(rows==[[job,'COMPLETED','0:0']], 'exact COMPLETED/0:0 parent required')


def close(a,b,tol=1e-10):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,'finite identity mismatch')


def validate_rpa(r):
    close(r['reference_energy_ha'],REFERENCE,1e-12)
    require(r['complete_q_weight'] is True,'incomplete q weight')
    close(r['q_weight_coverage'],1.,1e-12)
    qs=r['per_q']
    require([q['selected_iq'] for q in qs]==list(INDICES),'eight canonical q required')
    candidate=reference=0.; frequencies=None
    for q,m in zip(qs,MULT):
        close(q['q_weight'],m/64,1e-12)
        f=q['frequency_ha']; c=q['candidate_contributions_ha']; ref=q['reference_contributions_ha']
        require(len(f)==len(c)==len(ref)==12,'twelve frequencies required')
        require(all(math.isfinite(v) for v in f+c+ref),'nonfinite q contributions')
        require(f[0]>0 and all(a<b for a,b in zip(f,f[1:])),'ordered positive frequencies required')
        if frequencies is not None: require(f==frequencies,'frequency grid changed across q')
        frequencies=f
        candidate+=sum(c); reference+=sum(ref)
    close(candidate,r['candidate_energy_ha']); close(reference,REFERENCE)


def validate_stderr(text):
    allowed=('UserWarning: Converting a tensor with requires_grad=True to a scalar',
             'Consider using tensor.detach() first.',
             'if not all(math.isfinite(v) for v in rpa.values())')
    require(all(not line.strip() or any(v in line for v in allowed)
                for line in text.splitlines()),'unreviewed stderr content')


def winner(arms):
    require(set(arms)=={'steepest','lbfgs'},'comparison arms required')
    a,b=arms['steepest'],arms['lbfgs']
    for r in (a,b):
        require(r['accepted_steps']>0 and math.isfinite(r['seconds']) and r['seconds']>0
                and math.isfinite(r['gain_mev_per_c']) and r['gain_mev_per_c']>0,
                'positive measured progress required')
    ratio=(b['gain_mev_per_c']/b['seconds'])/(a['gain_mev_per_c']/a['seconds'])
    require(b['gain_mev_per_c']>a['gain_mev_per_c']
            and b['final_record']['objective']<a['final_record']['objective']
            and ratio>1.,'no measured L-BFGS advantage')
    return dict(method='lbfgs',efficiency_ratio=ratio)


def finite_diagnostic(d):
    require(math.isfinite(d['loss']) and d['loss']>=0,'nonfinite loss')
    capture=d['minimum_occupied_capture']
    condition=d.get('maximum_overlap_condition',d.get('maximum_condition_number'))
    require(math.isfinite(capture) and 0<capture<=1+1e-9
            and math.isfinite(condition) and condition>0,'invalid capture/condition')
    def walk(value):
        if isinstance(value,dict):
            for v in value.values(): walk(v)
        elif isinstance(value,list):
            for v in value: walk(v)
        elif isinstance(value,(int,float)):
            require(math.isfinite(value),'nonfinite gradient')
    if 'energy_gradient' in d: walk(d['energy_gradient'])


def validate_parent(parent,expected_sha,scheduler_text,*,verify_source=True):
    """Validate saved work without loading the response cache or evaluating RPA."""
    parent=Path(parent); result=parent/'result'
    r=json.loads(hashed(result/'RESULT.json',expected_sha))
    job=r['job_id']; terminal_state(scheduler_text,job)
    require(r['status']=='success' and r['physical_release_gate']=='hold'
            and r['bundle_sha256']==BUNDLE_SHA,'wrong scope or bundle')
    require(r['scope'] in ('bounded_optimizer_comparison','lbfgs_continuation'), 'wrong optimizer scope')
    require(all(r[k]==0 for k in ('delta_st_runs','scf_runs','sos_runs')),'unexpected physical run')
    close(r['reference_energy_ha'],REFERENCE,1e-12)
    require(r['coefficient_count']==248 and r['chart_dimension']==226,'basis dimension changed')
    for p in (parent/'BATCH_STATUS',result/'STATUS'):
        require(p.read_text().strip()=='success','parent status not successful')
    submission=json.loads((parent/'SUBMISSION.json').read_text())
    require(submission['job_id']==job and submission['source_commit']==r['source_commit']
            and submission['source_manifest_sha256']==r['source_manifest_sha256'],
            'submission/source mismatch')
    if verify_source:
        source=Path(submission['source'])
        manifest=json.loads(hashed(source/'SOURCE_MANIFEST.json',r['source_manifest_sha256']))
        require(manifest['commit']==r['source_commit'],'source commit changed')
        for name,h in manifest['files'].items(): hashed(within(source,name),h)
    validate_stderr((parent/('slurm-'+job+'.err')).read_text())
    hashed(result/'CHART.json',r['chart_sha256'])
    shared=json.loads((result/'SHARED_INITIAL_GRADIENT.json').read_text())
    validate_rpa(shared['rpa']); finite_diagnostic(shared)
    if r.get('recovery'):
        require(r['shared_initial_forward']==r['shared_initial_backward']==0,
                'recovery repeated initial kernel work')
        hashed(result/'SHARED_INITIAL_GRADIENT.json',r['recovery']['shared_gradient_sha256'])
    if r.get('continuation'):
        require(r['shared_initial_forward']==r['shared_initial_backward']==0,
                'continuation repeated initial kernel work')
    require(math.isfinite(r['seconds']) and r['seconds']>0
            and 0<r['peak_rss_kib']<190000*1024,'time or memory exceeds admission')
    summaries={}; endpoint=None
    for name,arm in r['arms'].items():
        root=result/name
        require(json.loads((root/'RESULT.json').read_text())==arm,'arm aggregate mismatch')
        require(arm['status']=='success' and (root/'STATUS').read_text().strip()=='success',
                'arm status failed')
        require(arm['physical_release_gate']=='hold' and arm['scf_runs']==0,'arm scope changed')
        n=arm['counts']['forward']; nb=arm['counts']['backward']
        require(type(n) is int and type(nb) is int and 0<=nb<=n<=r['max_forwards_per_arm'],
                'kernel budget violated')
        require(len(arm['evaluations'])==n==len(list(root.glob('evaluation_*'))),'missing evaluations')
        diagnostics=[]; backward=0
        for i,row in enumerate(arm['evaluations'],1):
            slot=root/('evaluation_%03d'%i)
            e=json.loads((slot/'EVALUATION.json').read_text())
            require({k:v for k,v in e.items() if k!='diagnostic'}==row,'evaluation receipt changed')
            require(e['evaluation']==i,'evaluation ordering changed')
            hashed(slot/'COEFFICIENTS.txt',e['coefficient_sha256'])
            validate_rpa(e['diagnostic']['rpa'])
            finite_diagnostic(e['diagnostic'])
            require(math.isfinite(e['seconds']) and e['seconds']>0,'invalid evaluation time')
            backward+=e['kind']=='forward_backward'
            diagnostics.append((slot,e))
        require(backward==nb,'backward count mismatch')
        previous=arm['initial_record']['objective']
        require(arm['initial_record']['rpa']==shared['rpa'],'arm initial response changed')
        for record in (arm['initial_record'],arm['final_record']):
            validate_rpa(record['rpa'])
            close(record['objective'],abs(record['rpa']['candidate_energy_ha']-REFERENCE))
        checkpoints=arm['accepted']
        require(0<len(checkpoints)==arm['accepted_steps']<=r['max_steps']
                and len(list(root.glob('checkpoint_*')))==len(checkpoints),'checkpoint count mismatch')
        for i,check in enumerate(checkpoints,1):
            slot=root/('checkpoint_%03d'%i)
            require(json.loads((slot/'CHECKPOINT.json').read_text())==check,'checkpoint metadata changed')
            hashed(slot/'COEFFICIENTS.txt',check['coefficient_sha256'])
            objective=check['record']['objective']; validate_rpa(check['record']['rpa'])
            require(math.isfinite(objective) and objective<=previous+1e-13,'nonmonotone checkpoint')
            require(any(e['coefficient_sha256']==check['coefficient_sha256']
                        and e['diagnostic']['rpa']['candidate_energy_ha']==check['record']['rpa']['candidate_energy_ha']
                        for _,e in diagnostics),'checkpoint lacks evaluated energy')
            previous=objective
        last=checkpoints[-1]
        require(last['record']==arm['final_record'],'FINAL record not last accepted')
        require(last['coefficient_sha256']==arm['coefficient_sha256'],'FINAL hash not best')
        final=root/'FINAL/COEFFICIENTS.txt'; hashed(final,arm['coefficient_sha256'])
        require(final.read_bytes()==(root/('checkpoint_%03d'%len(checkpoints))/'COEFFICIENTS.txt').read_bytes(),
                'FINAL not byte-identical to best')
        gain=(arm['initial_record']['objective']-previous)*CONVERSION*1000
        close(gain,arm['gain_mev_per_c'],1e-8)
        close(gain/(arm['seconds']/60),arm['gain_mev_per_c_per_minute'],1e-9)
        summaries[name]=dict(energy_ha=arm['final_record']['rpa']['candidate_energy_ha'],
            error_ev_per_c=previous*CONVERSION,gain_mev_per_c=gain,seconds=arm['seconds'],
            efficiency_mev_per_minute=arm['gain_mev_per_c_per_minute'],counts=arm['counts'],
            accepted_steps=arm['accepted_steps'],gradient_norm=arm['terminal_horizontal_gradient_norm'],
            coefficient_sha256=arm['coefficient_sha256'])
        if name=='lbfgs':
            matches=[(p,e) for p,e in diagnostics if e['coefficient_sha256']==arm['coefficient_sha256']
                     and 'energy_gradient' in e['diagnostic']]
            require(matches and arm['terminal_gradient_status']=='evaluated','fresh endpoint gradient required')
            p,e=matches[-1]
            close(e['diagnostic']['energy_gradient']['horizontal_gradient_norm'],
                  arm['terminal_horizontal_gradient_norm'])
            require(arm['optimizer']['active_bounds']==0,'chart boundary requires separate review')
            endpoint=dict(coefficient_path=str(final),coefficient_sha256=sha(final),
                gradient_path=str(p/'EVALUATION.json'),gradient_sha256=sha(p/'EVALUATION.json'))
    require(endpoint is not None,'missing L-BFGS endpoint')
    selection=winner(r['arms']) if r['scope']=='bounded_optimizer_comparison' else {'method':'lbfgs'}
    return dict(status='success',scope='validated_optimizer_continuation_parent',
        job_id=job,parent_root=str(parent),parent_result_sha256=expected_sha,
        source_manifest_sha256=r['source_manifest_sha256'],bundle_sha256=BUNDLE_SHA,
        physical_release_gate='hold',summary=summaries,selection=selection,**endpoint)


def admit_parent(parent,expected_sha):
    r=json.loads(hashed(Path(parent)/'result/RESULT.json',expected_sha))
    job=r['job_id']; require(job.isdigit(),'numeric job required')
    state=subprocess.check_output(['sacct','-n','-P','-X','-j',job,'--format=JobID,State,ExitCode'],text=True)
    return validate_parent(parent,expected_sha,state)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),'preserve existing validation receipt')
    r=admit_parent(args.parent.resolve(),args.sha256)
    with args.output.open('x') as f: json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps(r,indent=2,allow_nan=False))


if __name__=='__main__': main()
