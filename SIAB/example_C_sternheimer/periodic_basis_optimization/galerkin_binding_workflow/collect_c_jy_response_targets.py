"""Collect one complete, occupied-augmented C spdf response target set."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                    encoding='ascii')


def finalize_collection(checked, source_summary, *, collector_source_commit,
                        collector_source_manifest_sha256, source_summary_sha256,
                        scheduler_job_id, scheduler_partition, elapsed_seconds,
                        max_rss_kb):
    require(checked.get('status') == 'success'
            and checked.get('target_completeness_gate') == 'pass'
            and checked.get('occupied_constraints_complete') is True,
            'complete occupied target collection required')
    require(source_summary.get('status') == 'success'
            and source_summary.get('lmax') == 3
            and source_summary.get('primitive_count') == 992
            and source_summary.get('frequency_count') == 12
            and source_summary.get('target_sector_count') == 512,
            'incompatible source collection summary')
    reference = source_summary.get('reference_energy_ha_per_cell')
    mother = source_summary.get('mother_candidate_energy_ha_per_cell')
    error = source_summary.get('mother_error_ev_per_C')
    require(all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in (reference, mother, error)),
            'finite source reference and mother energies required')
    result = dict(checked)
    result.update(
        stage='occupied_augmented_complete_spdf_target_collection',
        exact_reference_snapshot_fit=False,
        collector_source_commit=collector_source_commit,
        collector_source_manifest_sha256=collector_source_manifest_sha256,
        source_summary_sha256=source_summary_sha256,
        target_source_commits=sorted({row['source_commit']
                                      for row in checked['per_q']}),
        reference_energy_ha_per_cell=float(reference),
        mother_candidate_energy_ha_per_cell=float(mother),
        mother_error_ev_per_C=float(error),
        scheduler_job_id=str(scheduler_job_id),
        scheduler_partition=str(scheduler_partition),
        elapsed_seconds=float(elapsed_seconds),
        max_rss_kb=int(max_rss_kb),
        physical_release_gate='hold')
    return result


def run(args):
    start = time.perf_counter()
    source = args.source.resolve()
    target_root = args.target_root.resolve()
    source_summary_path = args.source_summary.resolve()
    output = args.output.resolve()
    require(os.environ.get('C_EXECUTION_HOST') == 'df_iopcas_ghj'
            and os.environ.get('SLURM_JOB_ID'), 'registered DF job required')
    source_manifest_path = source / 'SOURCE_MANIFEST.json'
    source_manifest = json.loads(source_manifest_path.read_text(encoding='ascii'))
    require(source_manifest.get('commit') == args.source_commit,
            'collector source commit mismatch')
    for relative, expected in source_manifest['files'].items():
        require(sha(source / relative) == expected,
                'collector source file hash mismatch: ' + relative)
    require(sha(source_summary_path) == args.source_summary_sha256,
            'source collection summary hash mismatch')
    source_summary = json.loads(source_summary_path.read_text(encoding='ascii'))

    sys.path.insert(0, str(source / 'SIAB/opt_orb_pytorch_dpsi'))
    from c_jy_response_target_collection import collect_target_collection

    checked = collect_target_collection(
        target_root, source_summary['primitive_blocks'], q_slots=tuple(range(8)),
        k_record_count=64, frequency_count=12, primitive_count=992, lmax=3)
    result = finalize_collection(
        checked, source_summary, collector_source_commit=args.source_commit,
        collector_source_manifest_sha256=sha(source_manifest_path),
        source_summary_sha256=args.source_summary_sha256,
        scheduler_job_id=os.environ['SLURM_JOB_ID'],
        scheduler_partition=os.environ.get('SLURM_JOB_PARTITION', ''),
        elapsed_seconds=time.perf_counter() - start,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.mkdir()
    write_new(output / 'SUMMARY.json', result)
    write_new(output / 'PROVENANCE.json', dict(
        status='success', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=args.source_commit,
        files={'SUMMARY.json': sha(output / 'SUMMARY.json')}))
    (output / 'STATUS').write_text('success\n', encoding='ascii')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--source-summary', type=Path, required=True)
    parser.add_argument('--source-summary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
