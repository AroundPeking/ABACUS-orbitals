"""Fit one added f radial against the complete frozen jY response collection.

The original C 3s3p2d radial spans are preserved as a strict prefix. Only the
added f column is optimized, so this stage cannot rotate the trusted occupied
subspace. It fits response snapshots only; RPA acceptance is a later gate.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='ascii')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    source = Path(args.source).resolve()
    target_root = Path(args.target_root).resolve()
    bundle = Path(args.bundle).resolve()
    rank_root = Path(args.rank_root).resolve()
    output = Path(args.output).resolve()
    require(os.environ.get('C_EXECUTION_HOST') == 'df_iopcas_ghj', 'DF execution host required')
    require(os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'one-node fit required')
    require(args.source_commit, 'immutable source commit is required')
    manifest_path = source / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='ascii'))
    require(manifest.get('commit') == args.source_commit, 'source manifest commit mismatch')
    for relative, expected in manifest['files'].items():
        require(sha(source / relative) == expected, 'source file hash mismatch: ' + relative)

    import torch
    import sys
    sys.path.insert(0, str(source / 'SIAB/opt_orb_pytorch_dpsi'))
    from c_jy_response_target_collection import collect_target_collection
    from periodic_galerkin_basis import (
        read_periodic_optimizer_coefficients,
        write_periodic_optimizer_coefficients,
    )
    from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock
    from response_fit_iteration import fit_shared_response_radials
    from response_radial_fit import ResponseFitSector

    summary_path = target_root / 'collection-30d4390e' / 'SUMMARY.json'
    summary_expected = args.collection_summary_sha256
    require(sha(summary_path) == summary_expected, 'target collection summary hash mismatch')
    collection = json.loads(summary_path.read_text(encoding='ascii'))
    primitive_block_rows = collection['primitive_blocks']
    require(collection['primitive_count'] == 992 and collection['lmax'] == 3,
            'unexpected jY target mother contract')
    checked = collect_target_collection(
        target_root / 'collection-30d4390e', primitive_block_rows,
        q_slots=tuple(range(8)), k_record_count=64, frequency_count=12,
        primitive_count=992, lmax=3)
    require(checked['status'] == 'success' and checked['target_completeness_gate'] == 'pass',
            'complete response target collection required')

    sector_metadata = {}
    for q_slot in range(8):
        manifest_q = json.loads((target_root / ('q%02d/targets/TARGETS.json' % q_slot)).read_text())
        for row in manifest_q['sectors']:
            sector_metadata[(q_slot, row['file'])] = row
    primitive_block_tuple = tuple(
        PeriodicGalerkinPrimitiveBlock(**row) for row in primitive_block_rows)
    sectors = []
    for row in checked['target_files']:
        q_slot = int(row['q_slot'])
        relative = row['file']
        file_name = Path(relative).name
        metadata = sector_metadata[(q_slot, file_name)]
        array_path = target_root / 'collection-30d4390e' / relative
        with np.load(array_path, allow_pickle=False) as arrays:
            covariance = arrays['covariance']
            embedding = arrays['embedding']
        sectors.append(ResponseFitSector(
            'q%02d/%s' % (q_slot, file_name), primitive_block_tuple, embedding, covariance,
            occupied_rank=int(metadata['occupied_rank'])))
    require(len(sectors) == 512, 'complete 512-sector fit is required')

    original = read_periodic_optimizer_coefficients(
        bundle / 'backoff_ORIGINAL_COEFFICIENTS.txt', element='C', radial_rows=31,
        expected_nu=(3, 3, 2, 0, 0))['C']
    seed = read_periodic_optimizer_coefficients(
        rank_root / 'profile-3321/INITIAL_COEFFICIENTS.txt', element='C', radial_rows=31,
        expected_nu=(3, 3, 2, 1, 0))['C']
    initial = {'C': tuple(original[l] if l < 3 else seed[l] for l in range(5))}
    frozen_prefix = {'C': (3, 3, 2, 0, 0)}
    output.mkdir(parents=True)
    start = time.perf_counter()
    progress = []
    try:
        fitted, report = fit_shared_response_radials(
            initial, sectors, max_steps=args.max_steps,
            max_evaluations=args.max_evaluations, radius=args.radius,
            progress=progress.append, frozen_prefix=frozen_prefix)
        coefficient_path = output / 'FINAL_COEFFICIENTS.txt'
        write_periodic_optimizer_coefficients(coefficient_path, fitted)
        initial_path = output / 'INITIAL_COEFFICIENTS.txt'
        write_periodic_optimizer_coefficients(initial_path, initial)
        (output / 'PROGRESS.jsonl').write_text(
            ''.join(json.dumps(row, allow_nan=False) + '\n' for row in progress),
            encoding='ascii')
        result = dict(
            status='success', stage='frozen_spd_prefix_response_fit_not_RPA_acceptance',
            physical_release_gate='hold', source_commit=args.source_commit,
            target_collection_summary_sha256=summary_expected,
            target_collection_result_sha256=sha(target_root / 'collection-30d4390e/SUMMARY.json'),
            sector_count=len(sectors), target_norm2=checked['target_norm2'],
            input_profile=[3, 3, 2, 1, 0], frozen_prefix=[3, 3, 2, 0, 0],
            initial_coefficients_sha256=sha(initial_path),
            final_coefficients_sha256=sha(coefficient_path),
            max_steps=args.max_steps, max_evaluations=args.max_evaluations,
            radius=args.radius, report=report,
            elapsed_seconds=time.perf_counter() - start,
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_new(output / 'RESULT.json', result)
        provenance = dict(status='success', job_id=os.environ.get('SLURM_JOB_ID'),
            source_commit=args.source_commit, target_collection_summary_sha256=summary_expected,
            files={str(path.relative_to(output)): sha(path)
                   for path in output.rglob('*') if path.is_file()})
        write_new(output / 'PROVENANCE.json', provenance)
        (output / 'STATUS').write_text('success\n', encoding='ascii')
    except Exception as error:
        write_new(output / 'FAILURE.json', dict(status='failed', error=str(error),
            physical_release_gate='hold'))
        (output / 'STATUS').write_text('failed\n', encoding='ascii')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--rank-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collection-summary-sha256', required=True)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--max-evaluations', type=int, default=20)
    parser.add_argument('--radius', type=float, default=0.05)
    run(parser.parse_args())
