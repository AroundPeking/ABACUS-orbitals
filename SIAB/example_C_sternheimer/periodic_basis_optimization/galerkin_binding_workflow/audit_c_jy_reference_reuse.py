"""Read-only all-q check: reuse original Gamma S/H/O at each destination k.

This does not generate finite-q sources or admit a regenerated Hamiltonian.
It verifies only the existing spd overlap between two accepted data archives.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

from audit_c_jy_operator_restart import OperatorFiles, difference, occupied_gauge, sha


def compare_block(old, new):
    s = difference(old['s'], new['s'])
    h = difference(.5*old['h_ry'], new['h_ha'])
    a, o = occupied_gauge(old['o'], new['o'])
    source_energy = difference(old['source_eigen_ha'], new['source_eigen_ha'])
    target_energy = difference(old['target_eigen_ha'], new['target_eigen_ha'])
    commutator = float(np.max(np.abs(new['target_eigen_ha'][:, None]*a
                                    -a*old['target_eigen_ha'][None, :])))
    passed = (s['max_abs'] <= 1e-10 and h['max_abs'] <= 1e-10
              and o['relative'] <= 1e-8 and o['unitarity'] <= 1e-8
              and source_energy['max_abs'] <= 1e-10 and target_energy['max_abs'] <= 1e-10
              and commutator <= 1e-10)
    return dict(pass_gate=passed, overlap=s, hamiltonian_ha=h, occupied=o,
                source_energy_ha=source_energy, target_energy_ha=target_energy,
                occupied_energy_commutator_ha=commutator)


def validate_routing(sources, targets, coordinates, gamma):
    if (sorted(sources) != sorted(gamma) or sorted(targets) != sorted(gamma)
            or len(coordinates) != len(targets)):
        raise ValueError('source and destination k routing must be bijective')
    for target, coordinate in zip(targets, coordinates):
        if difference(np.asarray(gamma[target]), np.asarray(coordinate))['max_abs'] > 1e-12:
            raise ValueError('destination k coordinate mismatch')


def validate_reader_tree(root, bindings):
    root = root.resolve()
    for name, digest in bindings.items():
        path = (root/name).resolve()
        if root not in path.parents or not path.is_file() or sha(path) != digest:
            raise ValueError('frozen cache reader source mismatch: '+name)


def audit(bundle, reference, output, source_commit, reader_source):
    from c_portable_frozen_replay import (FREEZE_SHA, INDEX_SHA, INDICES, MULT,
                                         hashed, require, validate_cache_contract, within)
    from run_c_optimizer_comparison import BUNDLE_SHA
    repo = Path(__file__).resolve().parents[4]
    start = time.perf_counter()
    src = json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    require(src['commit'] == source_commit, 'immutable source commit mismatch')
    for name, digest in src['files'].items():
        hashed(within(repo, name), digest)
    manifest = json.loads(hashed(bundle/'BUNDLE.json', BUNDLE_SHA))
    for name, digest in manifest['files'].items():
        hashed(within(bundle, name), digest)
    # Cache identity belongs to its frozen reader, not later audit/test revisions.
    validate_reader_tree(reader_source, manifest['source_files'])
    sys.path.insert(0, str(reader_source/'SIAB/opt_orb_pytorch_dpsi'))
    from periodic_galerkin_basis import read_periodic_optimizer_coefficients
    from periodic_galerkin_dataset_cache import read_periodic_galerkin_dataset_cache
    frozen = json.loads(hashed(bundle/'backoff_INPUT_FREEZE.json', FREEZE_SHA))
    index = json.loads(hashed(bundle/'backoff_ACTIVE_DATA_CACHE.json', INDEX_SHA))
    records = validate_cache_contract(frozen, index)
    original = read_periodic_optimizer_coefficients(bundle/'backoff_ORIGINAL_COEFFICIENTS.txt',
        element='C', radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
    gamma = OperatorFiles(reference, False)
    require(gamma.scalar['selected_iq'] == '1' and gamma.scalar['k_count'] == '64'
            and gamma.scalar['primitive_count'] == '1550', 'complete original Gamma mother required')
    require(sha(reference/'manifest.dat') == records[0]['binding']['manifest_sha256']
            and sha(reference/'status.dat') == records[0]['binding']['status_sha256'],
            'Gamma archive must match accepted cache binding')
    qrows, failures, old_blocks = [], [], {}
    output.mkdir()
    with (output/'PROGRESS.jsonl').open('x') as progress:
        for slot, (item, record) in enumerate(zip(frozen['datasets'], records)):
            dataset = read_periodic_galerkin_dataset_cache(
                within(bundle, manifest['cache_paths'][str(item['label'])]),
                cache_sha256=record['complete_sha256'], active_coefficients=original, **record['binding'])
            require(dataset.selected_iq == item['selected_iq'] == INDICES[slot]
                    and dataset.q_weight == MULT[slot]/64. and len(dataset.kpoints) == 64
                    and dataset.q_count == 64 and dataset.primitive_count == 558,
                    'canonical all-q dataset required')
            for key in ('orbital_sha256', 'pseudopotential_sha256', 'auxiliary_basis_sha256',
                        'primitive_blocks_sha256', 'abacus_commit'):
                require(getattr(dataset, key) == gamma.scalar[key], 'reference identity mismatch: '+key)
            require(np.array_equal(dataset.frequency_ha.numpy(), np.asarray(gamma.frequencies)[:, 0])
                    and np.array_equal(dataset.frequency_weights_ha.numpy(), np.asarray(gamma.frequencies)[:, 1]),
                    'exact twelve-frequency protocol mismatch')
            reduction = dataset.active_primitive_reduction
            indices = tuple(reduction.source_indices)
            require(reduction.original_primitive_count == 1550, 'wrong mother layout')
            if slot == 0:
                initial_indices = indices
                for ik in range(1, 65):
                    require(gamma.kpoints[ik][0] == ik, 'original archive is not Gamma')
                    old_blocks[ik] = dict(s=gamma.array(1, ik)[np.ix_(indices, indices)],
                        h_ry=gamma.array(6, ik)[np.ix_(indices, indices)],
                        o=gamma.array(7, ik)[:, indices])
            require(indices == initial_indices, 'inconsistent exact active reduction')
            validate_routing([k.source_ik for k in dataset.kpoints], [k.target_ik for k in dataset.kpoints],
                [k.target_kpoint for k in dataset.kpoints],
                {ik: value[1][3:6] for ik, value in gamma.kpoints.items()})
            energies = {k.source_ik: k.source_eigenvalue_ha.numpy() for k in dataset.kpoints}
            per_k = []
            for k in dataset.kpoints:
                source, target = k.source_ik, k.target_ik
                require(difference(np.asarray(k.source_kpoint), gamma.kpoints[source][1][:3])['max_abs'] <= 1e-12
                        and k.k_weight == .03125 and np.array_equal(k.occupation.numpy(), np.ones(4)),
                        'source k or occupied contract mismatch')
                old = dict(old_blocks[target], source_eigen_ha=.5*gamma.eigenvalues[source],
                           target_eigen_ha=.5*gamma.eigenvalues[target])
                new = dict(s=k.overlap.numpy(), h_ha=k.hamiltonian_ha.numpy(), o=k.occupied_projection.numpy(),
                           source_eigen_ha=energies[source], target_eigen_ha=energies[target])
                row = dict(source_ik=source, target_ik=target, **compare_block(old, new))
                per_k.append(row)
                if not row['pass_gate']:
                    failures.append('q%s:k%d' % (item['label'], source))
            qrow = dict(label=item['label'], selected_iq=dataset.selected_iq, q_weight=dataset.q_weight,
                pass_gate=all(r['pass_gate'] for r in per_k), per_k=per_k,
                cache_sha256=record['complete_sha256'], cache_binding=record['binding'],
                max_hamiltonian_ha=max(r['hamiltonian_ha']['max_abs'] for r in per_k),
                max_overlap=max(r['overlap']['max_abs'] for r in per_k),
                max_occupied_relative=max(r['occupied']['relative'] for r in per_k))
            qrows.append(qrow)
            summary = {key: value for key, value in qrow.items() if key not in ('per_k', 'cache_binding')}
            progress.write(json.dumps(summary, allow_nan=False)+'\n')
            progress.flush()
            print(json.dumps(summary, allow_nan=False), flush=True)
            del dataset
    require(len(qrows) == 8 and sum(len(q['per_k']) for q in qrows) == 512, 'incomplete audit')
    result = dict(status='success', original_gamma_operator_reuse_gate='pass' if not failures else 'hold',
        failure_reasons=failures, bundle_sha256=BUNDLE_SHA, source_commit=source_commit,
        reader_source=str(reader_source), reader_source_hashes=manifest['source_files'],
        reference=str(reference), reference_hashes=gamma.hashes, per_q=qrows,
        elapsed_seconds=time.perf_counter()-start, compared_space='existing_spd_subblocks_only',
        missing_finite_q_sources_admitted=False, regenerated_hamiltonian_admitted=False,
        physical_release_gate='hold')
    (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'STATUS').write_text('success\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'reference', 'output', 'reader-source'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    audit(args.bundle, args.reference, args.output, args.source_commit, args.reader_source)
