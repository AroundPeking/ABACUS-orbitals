"""Audit new finite-q D against frozen spd; never consume regenerated H.

Occupied maps are derived at target k, then routed back to source k for D.
The output binds full new D to original operators; it is not an RPA result.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

from audit_c_jy_operator_restart import (OperatorFiles, difference, occupied_gauge,
                                        metric_signs, transform_source, sha)
from audit_c_jy_reference_reuse import validate_reader_tree
from prepare_c_jy_operator_restart import (REFERENCE_REUSE_SHA, GAMMA_SOURCE_AUDIT_SHA,
                                          validate_source_extension_evidence, FINITE_Q_SPECS,
                                          Q2_SOURCE_AUDIT_SHA, validate_finite_q_pilot)


def target_to_source(kpoints, count=64):
    if (sorted(kpoints) != list(range(1, count+1))
            or sorted(v[0] for v in kpoints.values()) != list(range(1, count+1))):
        raise ValueError('finite-q source/target routing is not bijective')
    return {target: source for source, (target, _) in kpoints.items()}


def restore_and_compare_source(old, new, source_gauge, auxiliary_map):
    restored = transform_source(new, source_gauge.conj().T, auxiliary_map.conj().T)
    result = difference(old, restored)
    result['pass_gate'] = result['relative'] <= 1e-6
    return restored, result


def finite_q_dataset_position(datasets, iq):
    label = int(FINITE_Q_SPECS[iq][0][1:])
    positions = [i for i, item in enumerate(datasets) if item['label'] == label]
    if len(positions) != 1 or datasets[positions[0]]['selected_iq'] != iq:
        raise ValueError('unique cache label and internal q index required')
    return positions[0]


def audit(run, bundle, reader_source, output, source_commit):
    from c_portable_frozen_replay import (FREEZE_SHA, INDEX_SHA, hashed, require,
                                         validate_cache_contract, within)
    from run_c_optimizer_comparison import BUNDLE_SHA
    start = time.perf_counter()
    repo = Path(__file__).resolve().parents[4]
    src = json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    require(src['commit'] == source_commit, 'immutable audit source mismatch')
    for name, digest in src['files'].items():
        hashed(within(repo, name), digest)
    require((run/'STATUS').read_text().strip() == 'success', 'source export not successful')
    contract = json.loads((run/'CONTRACT.json').read_text())
    iq = contract['selected_iq']
    require(contract['scope'] == 'finite_q_source_extension_original_operators_only'
            and iq in FINITE_Q_SPECS
            and contract['hamiltonian_origin'] == 'original_Gamma_at_target_k'
            and contract['regenerated_hamiltonian_admitted'] is False
            and contract['first_order_equations_allowed'] == 0,
            'explicit original-operator source-extension contract required')
    if iq != 22:
        pilot = json.loads(hashed(Path(contract['source_extension_pilot_audit']), Q2_SOURCE_AUDIT_SHA))
        validate_finite_q_pilot(pilot)
        require(contract['source_extension_pilot_sha256'] == Q2_SOURCE_AUDIT_SHA
                and contract['frozen_auxiliary']['metadata_sha256'] == FINITE_Q_SPECS[iq][2],
                'remaining source extension lacks locked pilot or same-q auxiliary cache')
    reuse = json.loads(hashed(Path(contract['reference_reuse_audit']), REFERENCE_REUSE_SHA))
    gamma = json.loads(hashed(Path(contract['gamma_source_audit']), GAMMA_SOURCE_AUDIT_SHA))
    validate_source_extension_evidence(reuse, gamma)
    require(contract['original_operator_hashes'] == reuse['reference_hashes']
            and contract['original_operator_directory'] == reuse['reference'],
            'original operator identity changed')
    for name, digest in contract['inputs'].items():
        hashed(within(run, name), digest)
    hashed(Path(contract['binary']), contract['binary_sha256'])
    manifest = json.loads(hashed(bundle/'BUNDLE.json', BUNDLE_SHA))
    for name, digest in manifest['files'].items():
        hashed(within(bundle, name), digest)
    validate_reader_tree(reader_source, manifest['source_files'])
    sys.path.insert(0, str(reader_source/'SIAB/opt_orb_pytorch_dpsi'))
    from periodic_galerkin_basis import read_periodic_optimizer_coefficients
    from periodic_galerkin_dataset_cache import read_periodic_galerkin_dataset_cache
    frozen = json.loads(hashed(bundle/'backoff_INPUT_FREEZE.json', FREEZE_SHA))
    index = json.loads(hashed(bundle/'backoff_ACTIVE_DATA_CACHE.json', INDEX_SHA))
    records = validate_cache_contract(frozen, index)
    original = read_periodic_optimizer_coefficients(bundle/'backoff_ORIGINAL_COEFFICIENTS.txt',
        element='C', radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
    position = finite_q_dataset_position(frozen['datasets'], iq)
    item, record = frozen['datasets'][position], records[position]
    dataset = read_periodic_galerkin_dataset_cache(
        within(bundle, manifest['cache_paths'][str(item['label'])]),
        cache_sha256=record['complete_sha256'], active_coefficients=original, **record['binding'])
    require(dataset.selected_iq == iq and dataset.q_count == 64
            and dataset.q_weight == FINITE_Q_SPECS[iq][1]/64
            and len(dataset.kpoints) == 64 and dataset.primitive_count == 558,
            'wrong frozen finite-q cache')
    new = OperatorFiles(run/('OUT.'+contract['suffix'])/'STERNHEIMER_BASIS_OPERATORS_V1', True)
    require(new.scalar['frozen_charge_sha256'] == contract['density_sha256']
            and new.scalar['executable_sha256'] == contract['binary_sha256']
            and new.scalar['abacus_commit'] == contract['source_commit'], 'native identity mismatch')
    for key in ('orbital_sha256', 'pseudopotential_sha256', 'auxiliary_basis_sha256',
                'primitive_blocks_sha256', 'selected_iq', 'q_count',
                'raw_auxiliary_dimension', 'whitened_auxiliary_rank'):
        require(new.scalar[key] == str(getattr(dataset, key)), 'source metadata mismatch: '+key)
    require(new.scalar['kernel'] == 'full_coulomb' and new.scalar['primitive_count'] == '1550'
            and new.scalar['k_count'] == '64'
            and float(new.scalar['q_weight']) == dataset.q_weight
            and np.array_equal(np.array(new.scalar['qpoint'].split(), float), dataset.qpoint)
            and np.array_equal(np.array(new.frequencies)[:, 0], dataset.frequency_ha.numpy())
            and np.array_equal(np.array(new.frequencies)[:, 1], dataset.frequency_weights_ha.numpy()),
            'source dimensions, q or twelve-frequency protocol changed')
    inverse_route = target_to_source(new.kpoints)
    indices = tuple(dataset.active_primitive_reduction.source_indices)
    require(indices == tuple(range(279))+tuple(range(775, 1054)), 'exact spd indices changed')
    v, w = dataset.coulomb_metric.numpy(), dataset.coulomb_whitening.numpy()
    vn, wn = new.array(4), new.array(5)
    signs = metric_signs(v, vn)
    metric = difference(vn, signs[:, None]*v*signs[None, :])
    t = w.conj().T @ v @ (signs[:, None]*wn)
    require(t.shape == (dataset.whitened_auxiliary_rank,)*2, 'white-space rank changed')
    unitary = float(np.max(np.abs(t.conj().T @ t-np.eye(len(t)))))
    failures = []
    if metric['relative'] > 1e-8 or unitary > 5e-6:
        failures.append('auxiliary_gauge')
    gauges, rows = {}, []
    energies = {k.source_ik: k.source_eigenvalue_ha.numpy() for k in dataset.kpoints}
    output.mkdir()
    for k in dataset.kpoints:
        source, target = k.source_ik, k.target_ik
        route, meta = new.kpoints[source]
        require(route == target and inverse_route[target] == source
                and np.array_equal(meta[:3], k.source_kpoint)
                and np.array_equal(meta[3:6], k.target_kpoint)
                and np.array_equal(meta[6:9], k.reciprocal_shift)
                and meta[9] == k.k_weight == .03125 and meta[10] == 4
                and np.array_equal(meta[11:], k.occupation.numpy()), 'k/occupation metadata mismatch')
        overlap = difference(k.overlap.numpy(), new.array(1, source)[np.ix_(indices, indices)])
        a, occupied = occupied_gauge(k.occupied_projection.numpy(), new.array(7, source)[:, indices])
        gauges[target] = a
        energy = difference(energies[source], .5*new.eigenvalues[source])
        commutator = float(np.max(np.abs(.5*new.eigenvalues[target][:, None]*a
                                        -a*energies[target][None, :])))
        passed = (overlap['max_abs'] <= 1e-10 and occupied['relative'] <= 1e-6
                  and occupied['unitarity'] <= 1e-6 and energy['max_abs'] <= 5e-7
                  and commutator <= 5e-7)
        rows.append(dict(source_ik=source, target_ik=target, overlap=overlap, occupied=occupied,
            source_eigenvalue_ha=energy, occupied_energy_commutator_ha=commutator, pass_gate=passed))
    with (output/'PROGRESS.jsonl').open('x') as progress:
        for k, row in zip(dataset.kpoints, rows):
            full = new.array(2, k.source_ik).reshape(4, len(t), 1550)
            # D carries the source occupied state, not the target occupied state.
            _, source_check = restore_and_compare_source(k.source.numpy(), full[:, :, indices],
                                                        gauges[k.source_ik], t)
            row['source'] = source_check
            row['pass_gate'] = row['pass_gate'] and source_check['pass_gate']
            if not row['pass_gate']:
                failures.append('k%d' % k.source_ik)
            progress.write(json.dumps(row, allow_nan=False)+'\n')
            progress.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
    map_path = output/'GAUGE_MAPS.npz'
    with map_path.open('xb') as stream:
        np.savez(stream, auxiliary_map=t, raw_auxiliary_signs=signs,
                 **{'occupied_at_k%d' % k: a for k, a in gauges.items()})
    result = dict(status='success', finite_q_spd_source_compatibility='pass' if not failures else 'hold',
        failure_reasons=failures, selected_iq=iq, compared_primitive_count=558,
        full_source_primitive_count=1550, per_k=rows, metric=metric, auxiliary_map_unitarity=unitary,
        source_commit=source_commit, run=str(run), contract_sha256=sha(run/'CONTRACT.json'),
        bundle_sha256=BUNDLE_SHA, cache_sha256=record['complete_sha256'], cache_binding=record['binding'],
        reader_source=str(reader_source), reader_source_hashes=manifest['source_files'],
        native_hashes=new.hashes, gauge_maps_sha256=sha(map_path),
        original_operator_directory=contract['original_operator_directory'],
        original_operator_hashes=contract['original_operator_hashes'],
        regenerated_hamiltonian_read=False, regenerated_hamiltonian_admitted=False,
        known_subblock_check_only=True, high_l_reference_comparison_available=False,
        full_q_admitted=False, physical_release_gate='hold', elapsed_seconds=time.perf_counter()-start)
    (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'STATUS').write_text('success\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bundle', 'reader-source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    audit(args.run, args.bundle, args.reader_source, args.output, args.source_commit)
