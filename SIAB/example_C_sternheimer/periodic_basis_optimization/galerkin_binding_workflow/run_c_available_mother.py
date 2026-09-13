"""DF-only, one-pass per-q diagnostic of the frozen available spd mother.

This is cached-matrix algebra, not ABACUS, reference generation, NAO fitting,
or a production SOS run. Every q retains its original physical star weight.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from c_portable_frozen_replay import (FREEZE_SHA, INDEX_SHA, INDICES, MULT,
                                    hashed, require, validate_cache_contract, within, write)
from run_c_optimizer_comparison import BUNDLE_SHA, REFERENCE, EV_PER_HA_PER_C

TARGET_BASELINE_SHA = '1150140a725f40ad5f96f58954231d58516e5480bf95bfc19b03d3f7f175adf4'


def validate_target_job(job, baseline, relative_rank_tolerance):
    require(relative_rank_tolerance == 1e-10
            and job.get('stage') == 'available_response_target_preparation'
            and job.get('target_baseline') == str(baseline)
            and job.get('target_baseline_summary_sha256') == TARGET_BASELINE_SHA,
            'response target must bind accepted 1e-10 control and explicit job stage')


def summarize(rows):
    require(len(rows) == 8 and [r['selected_iq'] for r in rows] == list(INDICES),
            'complete canonical eight-q set required, no extrapolation')
    shared = ('source_commit', 'bundle_sha256', 'shared_protocol', 'space',
              'available_primitive_count', 'original_primitive_count',
              'relative_rank_tolerance', 'frequency_ha', 'frequency_weights_ha')
    for row, multiplicity in zip(rows, MULT):
        require(row['status'] == 'success' and row['physical_release_gate'] == 'hold'
                and row['k_record_count'] == 64 and row['q_weight'] == multiplicity/64.
                and len(row['frequency_ha']) == len(row['frequency_weights_ha']) == 12,
                'incomplete q response contract')
        require(all(row[k] == rows[0][k] for k in shared), 'mixed mother protocol')
        require(all(math.isfinite(row[k]) for k in ('candidate_energy_ha', 'reference_energy_ha')),
                'nonfinite q energy')
    candidate = math.fsum(row['candidate_energy_ha'] for row in rows)
    reference = math.fsum(row['reference_energy_ha'] for row in rows)
    require(abs(reference-REFERENCE) <= 1e-12, 'frozen full-q reference not recovered')
    error = (candidate-reference)*EV_PER_HA_PER_C
    return dict(status='success', stage='available_mother_vs_reference_not_NAO_fit',
                candidate_energy_ha=candidate, reference_energy_ha=reference,
                signed_error_ev_per_C=error, absolute_error_ev_per_C=abs(error),
                available_mother_energy_gate=abs(error) < .1,
                exact_reference_snapshot_fit=False, physical_release_gate='hold',
                complete_q_weight=True, q_weight_coverage=math.fsum(r['q_weight'] for r in rows),
                **{k: rows[0][k] for k in shared}, per_q=rows)


def validate_rank_tolerance(value):
    require(math.isfinite(value) and 0 < value < 1, 'invalid relative rank tolerance')
    return value


def validate_job_contract(job, job_id, source_commit, relative_rank_tolerance):
    validate_rank_tolerance(relative_rank_tolerance)
    require(job.get('job_id') == job_id and job.get('source_commit') == source_commit
            and job.get('bundle_sha256') == BUNDLE_SHA
            and job.get('relative_rank_tolerance') == relative_rank_tolerance,
            'unique registered job and rank tolerance required')


def run(bundle, output, q_slot, source_commit, *, relative_rank_tolerance=1e-12,
        prepare_targets_from=None):
    validate_rank_tolerance(relative_rank_tolerance)
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo/'SIAB/opt_orb_pytorch_dpsi'))
    from periodic_available_mother import available_mother_response
    from periodic_galerkin_basis import read_periodic_optimizer_coefficients
    from periodic_galerkin_dataset_cache import read_periodic_galerkin_dataset_cache
    from periodic_galerkin_rpa import periodic_rpa_objective

    require(0 <= q_slot < 8, 'invalid q slot')
    require(os.environ.get('C_EXECUTION_HOST') == 'df_iopcas_ghj'
            and os.environ.get('SLURM_JOB_NUM_NODES') == '1'
            and int(os.environ.get('SLURM_ARRAY_TASK_ID', '-1')) == q_slot,
            'registered single-node DF array task required')
    require(len(source_commit) == 40, 'immutable source required')
    src = json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    require(src['commit'] == source_commit, 'source commit mismatch')
    for name, sha in src['files'].items():
        hashed(within(repo, name), sha)
    manifest = json.loads(hashed(bundle/'BUNDLE.json', BUNDLE_SHA))
    require(manifest['execution_host'] == 'df_iopcas_ghj', 'DF bundle required')
    for name, sha in manifest['files'].items():
        hashed(within(bundle, name), sha)
    for name, sha in manifest['source_files'].items():
        hashed(within(repo, name), sha)
    frozen = json.loads(hashed(bundle/'backoff_INPUT_FREEZE.json', FREEZE_SHA))
    index = json.loads(hashed(bundle/'backoff_ACTIVE_DATA_CACHE.json', INDEX_SHA))
    records = validate_cache_contract(frozen, index)
    job = json.loads((output/'JOB.json').read_text())
    validate_job_contract(job, os.environ.get('SLURM_ARRAY_JOB_ID'), source_commit,
                          relative_rank_tolerance)
    baseline = None
    if prepare_targets_from is not None:
        validate_target_job(job, prepare_targets_from, relative_rank_tolerance)
        baseline = json.loads(hashed(prepare_targets_from/'SUMMARY.json', TARGET_BASELINE_SHA))
        require(baseline['status'] == 'success' and baseline['bundle_sha256'] == BUNDLE_SHA
                and baseline['relative_rank_tolerance'] == relative_rank_tolerance,
                'accepted target baseline mismatch')
    target = output/('q%02d' % q_slot)
    target.mkdir()
    start = time.perf_counter()
    try:
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        original = read_periodic_optimizer_coefficients(
            bundle/'backoff_ORIGINAL_COEFFICIENTS.txt', element='C', radial_rows=31,
            expected_nu=(3, 3, 2, 0, 0))
        item, record = frozen['datasets'][q_slot], records[q_slot]
        dataset = read_periodic_galerkin_dataset_cache(
            within(bundle, manifest['cache_paths'][str(item['label'])]),
            cache_sha256=record['complete_sha256'], active_coefficients=original,
            **record['binding'])
        require(dataset.selected_iq == item['selected_iq'] == INDICES[q_slot]
                and dataset.q_count == 64
                and dataset.q_weight == item['q_weight'] == MULT[q_slot]/64.
                and dataset.physics_hash == item['physics_hash'], 'q identity mismatch')
        require(len(dataset.kpoints) == 64 and dataset.frequency_ha.numel() == 12
                and dataset.primitive_count == 558
                and dataset.active_primitive_reduction.original_primitive_count == 1550,
                'full-k/frequency/exact-reduction contract mismatch')
        load_seconds = time.perf_counter()-start
        target_rows = []
        with (target/'K_PROGRESS.jsonl').open('x') as progress:
            def log(entry):
                entry = dict(entry, elapsed_seconds=time.perf_counter()-start)
                progress.write(json.dumps(entry, allow_nan=False)+'\n')
                progress.flush()
                print(json.dumps(dict(q_slot=q_slot, **entry)), flush=True)
            if baseline is None:
                response, details = available_mother_response(
                    dataset, relative_rank_tolerance=relative_rank_tolerance, progress=log)
            else:
                from response_target import build_available_response_targets
                def consume(covariance, embedding, pi, metadata):
                    values = np.linalg.eigvalsh(covariance)
                    require(np.isfinite(embedding).all() and values[-1] > 0
                            and values[0] >= -1e-10*values[-1], 'invalid fixed target spectrum')
                    stem = 'k%04d' % metadata['source_ik']
                    array_file = target/(stem+'.npz')
                    with array_file.open('xb') as stream:
                        np.savez(stream, covariance=covariance, embedding=embedding)
                    row = dict(metadata, file=array_file.name,
                        sha256=hashlib.sha256(array_file.read_bytes()).hexdigest(),
                        covariance_dimension=len(covariance), primitive_count=embedding.shape[1],
                        covariance_minimum=float(values[0]), covariance_maximum=float(values[-1]),
                        source_kpoint=list(dataset.kpoints[len(target_rows)].source_kpoint),
                        target_kpoint=list(dataset.kpoints[len(target_rows)].target_kpoint))
                    target_rows.append(row)
                response, details = build_available_response_targets(dataset, consume,
                    relative_rank_tolerance=relative_rank_tolerance, progress=log)
                expected = baseline['per_q'][q_slot]
                baseline_q = prepare_targets_from/('q%02d' % q_slot)
                previous = json.loads((baseline_q/'PROVENANCE.json').read_text())
                require(previous['status'] == 'success', 'baseline q provenance failed')
                for name, sha in previous['files'].items():
                    hashed(within(baseline_q, name), sha)
                old_pi = np.load(baseline_q/'PI.npy', allow_pickle=False)
                pi_error = float(np.linalg.norm(response-old_pi)/max(np.linalg.norm(old_pi), 1e-300))
                require(expected['selected_iq'] == dataset.selected_iq
                        and expected['cache_complete_sha256'] == record['complete_sha256']
                        and expected['frequency_ha'] == dataset.frequency_ha.tolist()
                        and expected['frequency_weights_ha'] == dataset.frequency_weights_ha.tolist()
                        and pi_error <= 1e-10 and len(target_rows) == 64,
                        'target failed accepted full-q mother consistency')
                details.update(target_sector_count=len(target_rows),
                    target_norm2=math.fsum(r['target_norm2'] for r in target_rows),
                    baseline_pi_relative_error=pi_error,
                    target_baseline_summary_sha256=TARGET_BASELINE_SHA)
                write(target/'TARGETS.json', dict(status='success',
                    source_commit=source_commit, bundle_sha256=BUNDLE_SHA,
                    cache_complete_sha256=record['complete_sha256'],
                    relative_rank_tolerance=relative_rank_tolerance,
                    primitive_blocks=[asdict(block) for block in dataset.primitive_blocks],
                    frequency_ha=dataset.frequency_ha.tolist(),
                    frequency_weights_ha=dataset.frequency_weights_ha.tolist(),
                    sector_count=len(target_rows), target_norm2=details['target_norm2'],
                    exact_reference_snapshot_fit=False, physical_release_gate='hold',
                    target_truncation='none', coordinate_merge='none_per_q_source_target_record',
                    target_baseline_summary_sha256=TARGET_BASELINE_SHA, sectors=target_rows))
        np.save(target/'PI.npy', response, allow_pickle=False)
        result = periodic_rpa_objective((dataset,), (torch.from_numpy(response),))
        qrecord = {k: (v.tolist() if isinstance(v, torch.Tensor) else v)
                   for k, v in asdict(result.q_records[0]).items()}
        details.update(status='success', source_commit=source_commit, bundle_sha256=BUNDLE_SHA,
                       source_manifest_sha256=hashlib.sha256((repo/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
                       cache_complete_sha256=record['complete_sha256'],
                       source_job_id=os.environ['SLURM_JOB_ID'], array_slot=q_slot,
                       shared_protocol={k: getattr(dataset, k) for k in (
                           'abacus_commit', 'executable_sha256', 'orbital_sha256',
                           'pseudopotential_sha256', 'auxiliary_basis_sha256', 'q_count')},
                       frequency_ha=dataset.frequency_ha.tolist(),
                       frequency_weights_ha=dataset.frequency_weights_ha.tolist(),
                       k_record_count=len(details['k_records']),
                       candidate_energy_ha=float(result.candidate_energy_ha),
                       reference_energy_ha=float(result.reference_energy_ha),
                       pi_relative_error=math.sqrt(float(result.pi_relative_squared_error)),
                       tracelog_relative_error=math.sqrt(float(result.trace_log_relative_squared_error)),
                       per_frequency=qrecord, load_seconds=load_seconds,
                       elapsed_seconds=time.perf_counter()-start,
                       max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       python_version=sys.version, torch_version=torch.__version__,
                       numpy_version=np.__version__)
        write(target/'RESULT.json', details)
        write(target/'PROVENANCE.json', dict(status='success', source_commit=source_commit,
            job_id=os.environ['SLURM_JOB_ID'], array_slot=q_slot,
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in ((target/'PI.npy', target/'RESULT.json', target/'K_PROGRESS.jsonl')
                             + ((target/'TARGETS.json',) if baseline is not None else ()))}))
        (target/'STATUS').write_text('success\n')
    except Exception as error:
        write(target/'FAILURE.json', dict(status='failed', exception=type(error).__name__,
            message=str(error), elapsed_seconds=time.perf_counter()-start,
            source_commit=source_commit, relative_rank_tolerance=relative_rank_tolerance,
            physical_release_gate='hold'))
        (target/'STATUS').write_text('failed\n')
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--q-slot', type=int, required=True)
    parser.add_argument('--relative-rank-tolerance', type=float, default=1e-12,
                        help='normalized overlap rank cutoff; not auxiliary PCA threshold')
    parser.add_argument('--prepare-targets-from', type=Path,
                        help='explicit fixed-target mode; binds the accepted 1e-10 mother run')
    args = parser.parse_args(argv)
    validate_rank_tolerance(args.relative_rank_tolerance)
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args.bundle, args.output, args.q_slot, args.source_commit,
        relative_rank_tolerance=args.relative_rank_tolerance,
        prepare_targets_from=args.prepare_targets_from)
