"""Occupied-safe all-channel contraction of the accepted C spdf jY mother.

This is cached response algebra.  It does not run ABACUS, LibRPA, or Delta-ST,
and its output remains a candidate until an ordinary SOS RPA validation passes.
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

import numpy as np


PROFILES = ((3, 3, 2, 1, 0), (4, 4, 3, 2, 0),
            (5, 5, 4, 3, 0), (6, 6, 5, 4, 0))


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
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='ascii')


def ao_per_element(profile):
    return sum((2*l+1)*count for l, count in enumerate(profile))


def profile_name(profile):
    return ''.join(str(count) for count in profile[:4])


def select_profiles(value):
    if value == 'all':
        return PROFILES
    names = value.split(',') if isinstance(value, str) else []
    lookup = {profile_name(profile): profile for profile in PROFILES}
    if (not names or len(names) != len(set(names))
            or any(name not in lookup for name in names)):
        raise ValueError('profiles must be unique members of the fixed rank ladder')
    return tuple(lookup[name] for name in names)


def validate_collection(collection):
    require(collection.get('status') == 'success'
            and collection.get('target_completeness_gate') == 'pass',
            'successful complete target collection required')
    require(collection.get('occupied_constraints_complete') is True,
            'occupied-augmented response targets required')
    require(collection.get('target_sector_count') == 512
            and collection.get('primitive_count') == 992
            and collection.get('lmax') == 3
            and collection.get('q_array_slots') == list(range(8))
            and collection.get('k_record_count_per_q') == 64,
            'unexpected complete spdf target collection')


def read_sector_arrays(path, metadata):
    with np.load(path, allow_pickle=False) as arrays:
        require('occupied_embedding' in arrays,
                'occupied embedding array is required')
        covariance = arrays['covariance']
        embedding = arrays['embedding']
        occupied = arrays['occupied_embedding']
    require(covariance.shape == (metadata['covariance_dimension'],)*2
            and embedding.shape == (metadata['embedding_rows'],
                                    metadata['embedding_columns'])
            and occupied.shape == (metadata['occupied_embedding_rows'],
                                   metadata['embedding_columns'])
            and metadata['occupied_embedding_rows'] == metadata['occupied_rank']
            and all(np.isfinite(value).all()
                    for value in (covariance, embedding, occupied)),
            'occupied response sector shape or finite-value mismatch')
    return covariance, occupied, embedding


def run(args):
    start = time.perf_counter()
    source = Path(args.source).resolve()
    target_root = Path(args.target_root).resolve()
    initial_root = Path(args.initial_root).resolve()
    output = Path(args.output).resolve()
    require(os.environ.get('C_EXECUTION_HOST') == 'df_iopcas_ghj',
            'DF execution host required')
    require(os.environ.get('SLURM_JOB_NUM_NODES') == '1',
            'one-node contraction required')
    require(args.source_commit and 0 < args.occupied_capture_floor <= 1
            and math.isfinite(args.occupied_weight) and args.occupied_weight >= 0,
            'invalid immutable source or occupied controls')
    manifest_path = source/'SOURCE_MANIFEST.json'
    source_manifest = json.loads(manifest_path.read_text(encoding='ascii'))
    require(source_manifest.get('commit') == args.source_commit,
            'source manifest commit mismatch')
    for relative, expected in source_manifest['files'].items():
        require(sha(source/relative) == expected,
                'source file hash mismatch: '+relative)

    sys.path.insert(0, str(source/'SIAB/opt_orb_pytorch_dpsi'))
    import torch
    from c_jy_response_target_collection import collect_target_collection
    from periodic_galerkin_basis import (read_periodic_optimizer_coefficients,
                                         write_periodic_optimizer_coefficients)
    from periodic_galerkin_data import PeriodicGalerkinPrimitiveBlock
    from response_fit_iteration import fit_shared_response_radials
    from response_radial_fit import ResponseFitSector

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    summary_path = Path(args.collection_summary).resolve()
    require(sha(summary_path) == args.collection_summary_sha256,
            'target collection summary hash mismatch')
    summary = json.loads(summary_path.read_text(encoding='ascii'))
    primitive_rows = summary['primitive_blocks']
    checked = collect_target_collection(
        target_root, primitive_rows, q_slots=tuple(range(8)),
        k_record_count=64, frequency_count=12, primitive_count=992, lmax=3)
    validate_collection(checked)
    require(checked['target_norm2'] == summary['target_norm2'],
            'target collection norm changed')

    metadata = {}
    for slot in range(8):
        manifest = json.loads((target_root/('q%02d/targets/TARGETS.json' % slot))
                              .read_text(encoding='ascii'))
        for row in manifest['sectors']:
            metadata[(slot, row['file'])] = row
    blocks = tuple(PeriodicGalerkinPrimitiveBlock(**row) for row in primitive_rows)
    sectors = []
    for row in checked['target_files']:
        slot = int(row['q_slot'])
        name = Path(row['file']).name
        sector_metadata = metadata[(slot, name)]
        covariance, occupied, embedding = read_sector_arrays(
            target_root/row['file'], sector_metadata)
        sectors.append(ResponseFitSector(
            'q%02d/%s' % (slot, name), blocks, embedding, covariance,
            occupied_rank=int(sector_metadata['occupied_rank']),
            occupied_embedding=occupied))
    require(len(sectors) == 512, 'complete 512-sector contraction required')

    output.mkdir(parents=True)
    profile_results = []
    profiles = select_profiles(args.profiles)
    for profile in profiles:
        name = profile_name(profile)
        stage = output/('profile-'+name)
        stage.mkdir()
        initial_path_source = initial_root/('profile-'+name)/'INITIAL_COEFFICIENTS.txt'
        initial = read_periodic_optimizer_coefficients(
            initial_path_source, element='C', radial_rows=31, expected_nu=profile)
        progress = []
        fitted, report = fit_shared_response_radials(
            initial, sectors, max_steps=args.max_steps,
            max_evaluations=args.max_evaluations, radius=args.radius,
            progress=progress.append, occupied_weight=args.occupied_weight,
            occupied_capture_floor=args.occupied_capture_floor)
        initial_path = stage/'INITIAL_COEFFICIENTS.txt'
        final_path = stage/'FINAL_COEFFICIENTS.txt'
        best_path = stage/'BEST_COEFFICIENTS.txt'
        write_periodic_optimizer_coefficients(initial_path, initial)
        write_periodic_optimizer_coefficients(final_path, fitted)
        write_periodic_optimizer_coefficients(best_path, fitted)
        (stage/'PROGRESS.jsonl').write_text(
            ''.join(json.dumps(row, allow_nan=False)+'\n' for row in progress),
            encoding='ascii')
        result = dict(status='success',
            stage='all_channel_occupied_safe_response_contraction',
            physical_release_gate='hold', profile=list(profile),
            ao_per_C=ao_per_element(profile), all_radial_channels_free=True,
            occupied_capture_floor=args.occupied_capture_floor,
            occupied_weight=args.occupied_weight,
            initial_source_sha256=sha(initial_path_source),
            initial_coefficients_sha256=sha(initial_path),
            final_coefficients_sha256=sha(final_path),
            best_coefficients_sha256=sha(best_path), report=report)
        require(result['final_coefficients_sha256'] == result['best_coefficients_sha256'],
                'final coefficients differ from best checkpoint')
        write_new(stage/'RESULT.json', result)
        profile_results.append(result)

    final = dict(status='success',
        stage='all_channel_occupied_safe_response_contraction',
        physical_release_gate='hold', source_commit=args.source_commit,
        source_manifest_sha256=sha(manifest_path),
        target_collection_summary_sha256=args.collection_summary_sha256,
        occupied_capture_floor=args.occupied_capture_floor,
        occupied_weight=args.occupied_weight, profile_results=profile_results,
        selected_profiles=[profile_name(profile) for profile in profiles],
        elapsed_seconds=time.perf_counter()-start,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_threads=torch.get_num_threads())
    write_new(output/'RESULT.json', final)
    write_new(output/'PROVENANCE.json', dict(status='success',
        job_id=os.environ.get('SLURM_JOB_ID'), source_commit=args.source_commit,
        files={str(path.relative_to(output)): sha(path)
               for path in output.rglob('*') if path.is_file()}))
    (output/'STATUS').write_text('success\n', encoding='ascii')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--collection-summary', type=Path, required=True)
    parser.add_argument('--collection-summary-sha256', required=True)
    parser.add_argument('--initial-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=12)
    parser.add_argument('--max-evaluations', type=int, default=32)
    parser.add_argument('--radius', type=float, default=.05)
    parser.add_argument('--occupied-weight', type=float, default=1.)
    parser.add_argument('--occupied-capture-floor', type=float, default=.999999)
    parser.add_argument('--torch-threads', type=int, default=0)
    parser.add_argument('--profiles', default='all')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
