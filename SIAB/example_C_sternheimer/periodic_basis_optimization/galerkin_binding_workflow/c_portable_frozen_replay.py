"""Hash-preserving C cache relocation and endpoint/line-search diagnostics.

No reference generation, SCF, optimized candidate export or physical release.
Historical JSON paths identify the source; only bundle-relative paths are opened.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

FREEZE_SHA = '2bad5c827e83911769be5288d1649f7b2b3ccc256371b9c16971737135fba56b'
INDEX_SHA = '3bbbf45939799cc8abbbb8cfcf3014f63aa6d44ef475992dc71186e2b1230e76'
PARENTS = dict(energy_free='612193d272d8923dddfcd7bc0b5bec9c3dca764df0d2f78791c09503f346f935',
               energy_pbe='011afc3dd98334e375888864f59c4968bc6b316ec975f899ec25007f4663ad0a')
LABELS = (1, 2, 3, 6, 7, 8, 11, 28)
INDICES = (1, 22, 43, 6, 27, 23, 11, 55)
MULT = (1, 8, 4, 6, 24, 12, 3, 6)
PROBE_STEPS = (-1e-4, 1e-4, 5e-4, 1e-3)
PHYSICAL_RELEASE = 'hold'


def require(value, message):
    if not value:
        raise ValueError(message)


def within(root, name):
    root = Path(root).resolve()
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'relative bundle path required')
    out = (root/path).resolve()
    require(out != root and root in out.parents, 'path outside bundle')
    return out


def hashed(path, expected):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'SHA256 mismatch: '+str(path))
    return data


def validate_cache_contract(frozen, index):
    require(frozen['status'] == index['status'] == 'success', 'accepted inputs required')
    require(index['scope'] == 'exact_active_dataset_derivative_not_new_reference', 'exact cache required')
    items, records = frozen['datasets'], index['records']
    require([r['label'] for r in items] == list(LABELS)
            and [r['label'] for r in records] == list(LABELS), 'complete canonical eight-q set required')
    for item, record, iq, mult in zip(items, records, INDICES, MULT):
        b = record['binding']; ids = b['identifiers']
        require(item['selected_iq'] == iq and item['q_weight'] == mult/64., 'q identity changed')
        require(b['manifest_sha256'] == item['manifest_sha256']
                and b['status_sha256'] == item['status_sha256']
                and ids['acceptance_sha256'] == item['acceptance_sha256']
                and ids['freeze_sha256'] == FREEZE_SHA, 'cache binding changed')
    return records


def check_replay(energy, expected, gradient, reference_gradient, ref, expected_ref):
    require(all(math.isfinite(x) for x in [energy, expected, ref, expected_ref,
                                          *gradient, *reference_gradient]), 'nonfinite replay')
    require(abs(energy-expected) <= 1e-9 and abs(ref-expected_ref) <= 1e-12,
            'frozen energy reproduction failed')
    require(len(gradient) == len(reference_gradient) and len(gradient) > 0,
            'gradient dimensions changed')
    require(all(abs(x-y) <= 1e-8+1e-6*abs(y) for x, y in zip(gradient, reference_gradient)),
            'horizontal gradient reproduction failed')


def check_settings(settings):
    require(settings == dict(frequency_batch_size=3, weights=dict(
        energy_weight=1., pi_weight=1., trace_log_weight=1.)), 'frozen evaluation settings changed')


def check_q_replay(actual, before):
    require(actual.get('complete_q_weight') is True
            and abs(actual['q_weight_coverage']-1.) <= 1e-12, 'full q weight required')
    require([q['selected_iq'] for q in actual['per_q']] == list(INDICES)
            and len(before['per_q']) == 8, 'canonical q result required')
    for a,b,n in zip(actual['per_q'], before['per_q'], MULT):
        require(a['q_weight'] == b['q_weight'] == n/64.
                and a['selected_iq'] == b['selected_iq']
                and len(a['frequency_ha']) == 12
                and a['frequency_ha'] == b['frequency_ha'], 'q/frequency identity changed')
        for key in ('candidate_contributions_ha', 'reference_contributions_ha'):
            require(len(a[key]) == len(b[key]) == 12 and all(
                math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 1e-11+1e-7*abs(y)
                for x,y in zip(a[key],b[key])), 'q/frequency contributions changed')


def write(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(bundle, output, manifest_sha):
    import numpy as np
    import torch
    here = Path(__file__).resolve()
    sys.path[:0] = [str(here.parents[4]/'SIAB/opt_orb_pytorch_dpsi'), str(here.parent)]
    from periodic_galerkin_basis import read_periodic_optimizer_coefficients
    from periodic_galerkin_dataset_cache import read_periodic_galerkin_dataset_cache
    from periodic_galerkin_fit import _prepare_block_contraction_caches, _global_rpa_loss, CandidateGuardError
    from periodic_galerkin_radial_diagnostics import evaluate_radial_gradients, retract_displacement
    from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference
    from periodic_galerkin_rpa import prepare_periodic_rpa_reference
    from c_response_band_policy import prepare_c_band_guard, RESPONSE
    from run_c_combined_step import screen_candidate
    from optimize_c_all_radial_fast import _validate_c_dataset

    start = time.perf_counter()
    manifest = json.loads(hashed(bundle/'BUNDLE.json', manifest_sha))
    settings = manifest['evaluation_settings']
    check_settings(settings)
    require(manifest['execution_host'] == 'df_iopcas_ghj', 'df-only execution required')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('C_EXECUTION_HOST') == 'df_iopcas_ghj',
            'explicit df scheduler runtime required')
    require(os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'one node required')
    for name, digest in manifest['files'].items():
        hashed(within(bundle, name), digest)
    for name, digest in manifest['source_files'].items():
        hashed(within(here.parents[4], name), digest)
    frozen = json.loads(hashed(bundle/'backoff_INPUT_FREEZE.json', FREEZE_SHA))
    index = json.loads(hashed(bundle/'backoff_ACTIVE_DATA_CACHE.json', INDEX_SHA))
    records = validate_cache_contract(frozen, index)
    read = lambda p: read_periodic_optimizer_coefficients(p, element='C', radial_rows=31,
                                                         expected_nu=(3, 3, 2, 0, 0))
    original = read(bundle/'backoff_ORIGINAL_COEFFICIENTS.txt')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    output.mkdir()
    datasets = []
    for item, rec, iq, mult in zip(frozen['datasets'], records, INDICES, MULT):
        cache = within(bundle, manifest['cache_paths'][str(item['label'])])
        d = read_periodic_galerkin_dataset_cache(cache, cache_sha256=rec['complete_sha256'],
                                               active_coefficients=original, **rec['binding'])
        _validate_c_dataset(d, item, iq, mult)
        datasets.append(d)
        print(json.dumps(dict(loaded_q=item['label'], seconds=time.perf_counter()-start)), flush=True)
    guard = prepare_c_band_guard(datasets[0], original, policy=RESPONSE)
    centers = {}
    with torch.no_grad():
        datasets = tuple(prepare_periodic_occupied_reference(d) for d in datasets)
        datasets = _prepare_block_contraction_caches(datasets, original, 1)
        ref = prepare_periodic_rpa_reference(datasets)
    flatten_gradient = lambda d: np.concatenate([
        np.asarray(c['horizontal_gradient'], dtype=float).ravel() for c in d['energy_gradient']['channels']])
    for arm, digest in PARENTS.items():
        parent = json.loads(hashed(bundle/arm/'RESULT.json', digest))
        c = read(bundle/arm/'FINAL/COEFFICIENTS.txt')
        hashed(bundle/arm/'FINAL/COEFFICIENTS.txt', parent['coefficient_sha256'])
        terminal = parent['gradients'][-1]
        old = json.loads(hashed(bundle/arm/'gradient_020/GRADIENT.json', terminal['gradient_sha256']))
        require(old['coefficient_sha256'] == parent['coefficient_sha256'], 'terminal center changed')
        free = arm == 'energy_free'
        floor = parent['occupied_capture_floor']
        guarded = lambda c: guard(c, enforce_accuracy=not free)
        with torch.no_grad():
            bands, capture = screen_candidate(datasets, c, guarded, floor, enforce_band_accuracy=not free)
        d = evaluate_radial_gradients(datasets, c, occupied_capture_tolerance=1-floor,
                                     energy_only=True, **settings)
        g, g_old = flatten_gradient(d), flatten_gradient(old)
        check_replay(d['rpa']['candidate_energy_ha'], parent['final_candidate']['rpa']['candidate_energy_ha'],
                     g, g_old, d['rpa']['reference_energy_ha'], parent['final_candidate']['rpa']['reference_energy_ha'])
        check_q_replay(d['rpa'], old['rpa'])
        write(output/(arm+'_REPLAY.json'), dict(d, replay_gate='pass', band_guard=bands,
              gradient_max_abs_difference=float(np.max(np.abs(g-g_old))), parent_result_sha256=digest))
        centers[arm] = (c, d, g, parent)
        print(json.dumps(dict(arm=arm, replay_gate='pass', ec=d['rpa']['candidate_energy_ha'])), flush=True)

    c, center, g, parent = centers['energy_free']
    sign = 1 if center['rpa']['candidate_energy_ha'] >= center['rpa']['reference_energy_ha'] else -1
    norm = float(np.linalg.norm(g)); require(norm > 1e-12, 'stationary endpoint: do not construct a probe')
    blocks = center['energy_gradient']['channels']
    direction = {'C': [torch.tensor(-sign*np.asarray(b['horizontal_gradient'])/norm,
                                   dtype=torch.float64).reshape(b['shape']) for b in blocks]}
    rows = []
    for step in (*PROBE_STEPS, parent['optimization']['radius']):
        candidate = retract_displacement(c, direction, step)
        with torch.no_grad():
            try:
                bands, capture = screen_candidate(datasets, candidate,
                    lambda x: guard(x, enforce_accuracy=False), 1e-12, enforce_band_accuracy=False)
            except CandidateGuardError as e:
                rows.append(dict(step=step, status='numerical_guard_reject', reason=str(e)))
                continue
            loss, capture, condition, _, rpa = _global_rpa_loss(datasets, candidate,
                occupied_capture_tolerance=1-1e-12, reference_cache=ref, **settings)
        gain = abs(center['rpa']['candidate_energy_ha']-center['rpa']['reference_energy_ha'])-abs(
            rpa['candidate_energy_ha']-rpa['reference_energy_ha'])
        row = dict(step=step, status='evaluated', objective_gain_ha=gain,
                   predicted_gain_ha=step*norm, model_ratio=gain/(step*norm), rpa=rpa,
                   capture=capture, condition=condition, loss=float(loss))
        write(output/('PROBE_%02d.json'%len(rows)), row); rows.append(row)
        print(json.dumps({k:row[k] for k in ('step','objective_gain_ha','model_ratio')}), flush=True)
    tiny = {r['step']:r for r in rows if r['status'] == 'evaluated'}
    finite_difference = None
    if all(s in tiny for s in (-1e-4, 1e-4)):
        derivative = (tiny[-1e-4]['objective_gain_ha']-tiny[1e-4]['objective_gain_ha'])/2e-4
        finite_difference = dict(step=1e-4, objective_derivative_ha=derivative,
            analytic_derivative_ha=-norm, relative_difference=abs(derivative+norm)/norm)
    result = dict(status='success', scope='df_portability_and_local_descent_diagnostic',
        probes=rows, replay_gate='pass', physical_release_gate=PHYSICAL_RELEASE,
        finite_difference=finite_difference,
        bundle_sha256=manifest_sha, source_commit=manifest['source_commit'],
        torch_version=torch.__version__, numpy_version=np.__version__,
        job_id=os.environ['SLURM_JOB_ID'], seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        delta_st_runs=0, scf_runs=0, optimizer_steps=0, candidate_export='none')
    write(output/'RESULT.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    run(args.bundle.resolve(), args.output.resolve(), args.manifest_sha256)


if __name__ == '__main__': main()
