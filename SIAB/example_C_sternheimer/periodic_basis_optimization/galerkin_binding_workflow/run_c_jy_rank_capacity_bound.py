"""Compute an optimistic rank-only lower bound from cached C jY targets."""

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


MODULES = Path(__file__).resolve().parents[3]/'opt_orb_pytorch_dpsi'
sys.path.insert(0, str(MODULES))

from response_radial_fit import (independent_sector_covariance_spectrum,
                                 independent_sector_rank_lower_bound_from_spectrum)


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


def aggregate_rank_bounds(sectors, *, profiles, atoms_per_cell):
    """Aggregate independent-sector Ky Fan bounds over fixed response weights."""
    if (not sectors or type(atoms_per_cell) is not int or atoms_per_cell <= 0
            or not profiles):
        raise ValueError('nonempty sectors, profiles and atom count required')
    checked = []
    for sector in sectors:
        if (not isinstance(sector, dict) or type(sector.get('q_slot')) is not int
                or sector['q_slot'] < 0 or type(sector.get('occupied_rank')) is not int
                or sector['occupied_rank'] < 0):
            raise ValueError('invalid rank-bound sector')
        checked.append(sector)

    sector_spectra = [independent_sector_covariance_spectrum(
        sector['covariance']) for sector in checked]

    profile_results = []
    for profile in profiles:
        ao_count = ao_per_element(profile)
        cell_ao = atoms_per_cell*ao_count
        reports = []
        for sector, eigenvalues in zip(checked, sector_spectra):
            available = cell_ao-sector['occupied_rank']
            if available <= 0:
                raise ValueError('candidate has no virtual rank after occupied space')
            bound = independent_sector_rank_lower_bound_from_spectrum(
                eigenvalues, available)
            reports.append(dict(q_slot=sector['q_slot'], **bound))
        target = math.fsum(row['target_norm2'] for row in reports)
        residual = math.fsum(row['residual_norm2_lower_bound'] for row in reports)
        per_q = []
        for q_slot in sorted({row['q_slot'] for row in reports}):
            selected = [row for row in reports if row['q_slot'] == q_slot]
            q_target = math.fsum(row['target_norm2'] for row in selected)
            q_residual = math.fsum(row['residual_norm2_lower_bound'] for row in selected)
            per_q.append(dict(q_slot=q_slot, sector_count=len(selected),
                target_norm2=q_target, residual_norm2_lower_bound=q_residual,
                response_loss_lower_bound=q_residual/q_target))
        profile_results.append(dict(profile=list(profile), ao_per_C=ao_count,
            nominal_cell_ao=cell_ao,
            virtual_rank_by_sector=[row['available_virtual_rank'] for row in reports],
            target_norm2=target, residual_norm2_lower_bound=residual,
            captured_norm2_upper_bound=target-residual,
            response_loss_lower_bound=residual/target, per_q=per_q,
            physical_release_gate='hold'))
    return dict(status='success', scope='independent_sector_rank_only_lower_bound',
        no_physics_run=True, shared_localized_basis_constraint=False,
        physical_release_gate='hold', atoms_per_cell=atoms_per_cell,
        sector_count=len(checked), profile_results=profile_results)


def run(args):
    start = time.perf_counter()
    source = Path(args.source).resolve()
    target_root = Path(args.target_root).resolve()
    output = Path(args.output).resolve()
    require(os.environ.get('C_EXECUTION_HOST') == 'df_iopcas_ghj',
            'DF execution host required')
    require(os.environ.get('SLURM_JOB_NUM_NODES') == '1',
            'one-node rank bound required')
    manifest_path = source/'SOURCE_MANIFEST.json'
    source_manifest = json.loads(manifest_path.read_text(encoding='ascii'))
    require(source_manifest.get('commit') == args.source_commit,
            'source manifest commit mismatch')
    for relative, expected in source_manifest['files'].items():
        require(sha(source/relative) == expected,
                'source file hash mismatch: '+relative)

    sys.path.insert(0, str(source/'SIAB/opt_orb_pytorch_dpsi'))
    from c_jy_response_target_collection import collect_target_collection

    summary_path = Path(args.collection_summary).resolve()
    require(sha(summary_path) == args.collection_summary_sha256,
            'target collection summary hash mismatch')
    summary = json.loads(summary_path.read_text(encoding='ascii'))
    checked = collect_target_collection(
        target_root, summary['primitive_blocks'], q_slots=tuple(range(8)),
        k_record_count=64, frequency_count=12, primitive_count=992, lmax=3)
    require(checked.get('status') == 'success'
            and checked.get('target_completeness_gate') == 'pass'
            and checked.get('occupied_constraints_complete') is True
            and checked.get('target_sector_count') == 512,
            'successful occupied-complete target collection required')
    require(checked['target_norm2'] == summary['target_norm2'],
            'target collection norm changed')

    metadata = {}
    for slot in range(8):
        manifest = json.loads((target_root/('q%02d/targets/TARGETS.json' % slot))
                              .read_text(encoding='ascii'))
        for row in manifest['sectors']:
            metadata[(slot, Path(row['file']).name)] = row
    sectors = []
    for row in checked['target_files']:
        slot = int(row['q_slot'])
        path = target_root/row['file']
        meta = metadata[(slot, path.name)]
        with np.load(path, allow_pickle=False) as arrays:
            covariance = arrays['covariance'].copy()
        require(covariance.shape == (meta['covariance_dimension'],)*2
                and np.isfinite(covariance).all(),
                'rank-bound covariance shape or finite-value mismatch')
        sectors.append(dict(q_slot=slot, occupied_rank=int(meta['occupied_rank']),
                            covariance=covariance))
    atoms = {(row['element'], row['atom_index'])
             for row in summary['primitive_blocks']}
    require(atoms and {element for element, _ in atoms} == {'C'},
            'C primitive atom layout required')
    result = aggregate_rank_bounds(
        sectors, profiles=select_profiles(args.profiles), atoms_per_cell=len(atoms))
    result.update(source_commit=args.source_commit,
        source_manifest_sha256=sha(manifest_path),
        target_collection_summary_sha256=args.collection_summary_sha256,
        target_norm2=checked['target_norm2'], elapsed_seconds=time.perf_counter()-start,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        evidence_boundary=('Each sector chooses an independent optimal subspace; '
                           'the bound does not construct one shared atom-centred basis.'))
    output.mkdir(parents=True)
    write_new(output/'RESULT.json', result)
    write_new(output/'PROVENANCE.json', dict(status='success',
        job_id=os.environ.get('SLURM_JOB_ID'), source_commit=args.source_commit,
        files={'RESULT.json': sha(output/'RESULT.json')}))
    (output/'STATUS').write_text('success\n', encoding='ascii')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--collection-summary', type=Path, required=True)
    parser.add_argument('--collection-summary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profiles', default='all')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
