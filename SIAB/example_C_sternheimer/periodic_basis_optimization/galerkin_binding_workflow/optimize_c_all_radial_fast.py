"""Bounded full-q C optimization with free s/p/d radials and SCF release held."""

import argparse
import json
from pathlib import Path
import re
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]/'opt_orb_pytorch_dpsi'))

import torch
from periodic_galerkin_basis import (
    read_periodic_optimizer_coefficients, write_periodic_optimizer_coefficients,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset, _require, _sha256
from periodic_galerkin_fit import optimize_periodic_galerkin_basis
from c_response_band_policy import prepare_c_band_guard, validate_band_policy
from periodic_galerkin_pbe_guard import prepare_frozen_band_guard
from periodic_galerkin_dataset_cache import (
    write_periodic_galerkin_dataset_cache, read_periodic_galerkin_dataset_cache,
)
from optimize_periodic_basis import write_best_checkpoint
from export_periodic_orbitals import write_abacus_orbital


def save_json(path, value):
    temporary = path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def check(path, expected):
    _require(_sha256(str(path)) == expected, 'SHA256 mismatch: '+str(path))


def iteration_settings(max_steps):
    return dict(max_steps=max_steps, minimum_steps=min(5, max_steps),
                plateau_patience=2, plateau_relative_improvement=1e-6)


def _cache_options(index, digest, persist):
    _require((index is None) == (digest is None),
             'active cache index and SHA256 must be supplied together')
    _require(index is None or not persist, 'cache reuse and persistence are incompatible')


def _source_checks(item):
    root = Path(item['dataset'])
    return ((root/'manifest.dat', item['manifest_sha256']),
            (root/'status.dat', item['status_sha256']),
            (Path(item['acceptance']), item['acceptance_sha256']))


def _validate_c_dataset(dataset, item, iq, multiplicity):
    _require(dataset.selected_iq == iq == item['selected_iq']
             and dataset.q_count == 64
             and dataset.q_weight == multiplicity/64. == item['q_weight']
             and dataset.physics_hash == item['physics_hash'], 'q identity mismatch')
    _require(len(dataset.kpoints) == 64 and dataset.frequency_ha.numel() == 12
             and dataset.primitive_count == 558
             and dataset.active_primitive_reduction.original_primitive_count == 1550,
             'full-k/frequency/exact-reduction contract mismatch')


def _preflight_cache_index(path, digest, frozen, freeze_hash, initial, labels, indices, mult):
    check(path, digest)
    index = json.loads(Path(path).read_text())
    check(path, digest)
    _require(isinstance(index, dict) and index.get('status') == 'success'
             and index.get('scope') == 'exact_active_dataset_derivative_not_new_reference',
             'accepted active cache index required')
    records = index.get('records')
    _require(isinstance(records, list) and len(records) == len(labels)
             and all(isinstance(r, dict) and type(r.get('label')) is int for r in records)
             and [r['label'] for r in records] == labels, 'canonical complete cache q order required')
    _require(set(initial) == {'C'} and len(initial['C']) == 5 and all(
        isinstance(c, torch.Tensor) and tuple(c.shape) == (31, n)
        for c, n in zip(initial['C'], (3, 3, 2, 0, 0))), 'cache input coefficient nu/profile mismatch')
    seen = set()
    for record, item, iq, multiplicity in zip(records, frozen['datasets'], indices, mult):
        _require(item['selected_iq'] == iq and item['q_weight'] == multiplicity/64.,
                 'frozen q identity mismatch')
        cache_path = record.get('path')
        _require(isinstance(cache_path, str) and Path(cache_path).is_absolute(),
                 'cache path must be absolute')
        resolved = str(Path(cache_path).resolve())
        _require(resolved not in seen, 'duplicate cache path')
        seen.add(resolved)
        binding = record.get('binding')
        _require(isinstance(binding, dict) and set(binding) == {
            'manifest_sha256', 'status_sha256', 'identifiers'}, 'invalid cache binding')
        identifiers = binding['identifiers']
        _require(isinstance(identifiers, dict) and set(identifiers) == {
            'freeze_sha256', 'acceptance_sha256', 'mapping_sha256'}, 'invalid cache identifiers')
        _require(binding['manifest_sha256'] == item['manifest_sha256']
                 and binding['status_sha256'] == item['status_sha256']
                 and identifiers['freeze_sha256'] == freeze_hash
                 and identifiers['acceptance_sha256'] == item['acceptance_sha256'],
                 'cache and frozen input binding mismatch')
        for value in (record.get('complete_sha256'), identifiers['mapping_sha256']):
            _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
                     'invalid cache completion/mapping SHA256')
    # Verify every q's small source files before reading arrays or screening q1.
    for item in frozen['datasets']:
        for source, expected in _source_checks(item):
            check(source, expected)
    return records


def load_frozen_c(freeze, digest, initial, output, persist_active_cache=False, *,
                  active_cache_index=None, active_cache_index_sha256=None,
                  band_guard_policy='legacy_four_virtual'):
    validate_band_policy(band_guard_policy)
    guard_factory = prepare_frozen_band_guard if band_guard_policy == 'legacy_four_virtual' else prepare_c_band_guard
    guard_options = {'atoms_per_cell': 2} if band_guard_policy == 'legacy_four_virtual' else {'policy': band_guard_policy}
    _cache_options(active_cache_index, active_cache_index_sha256, persist_active_cache)
    check(freeze, digest)
    frozen = json.loads(Path(freeze).read_text())
    _require(frozen['status'] == 'success', 'accepted input freeze required')
    labels, indices, mult = [1, 2, 3, 6, 7, 8, 11, 28], [1, 22, 43, 6, 27, 23, 11, 55], [1, 8, 4, 6, 24, 12, 3, 6]
    _require([r['label'] for r in frozen['datasets']] == labels, 'canonical eight-q order required')
    datasets, records, cache_records = [], [], []
    if active_cache_index is not None:
        cache_records = _preflight_cache_index(active_cache_index, active_cache_index_sha256,
            frozen, digest, initial, labels, indices, mult)
        for item, cached, iq, multiplicity in zip(frozen['datasets'], cache_records, indices, mult):
            save_json(output/'PROGRESS.json', dict(stage='loading_cached_q', label=item['label'], completed=records))
            started = time.perf_counter()
            dataset = read_periodic_galerkin_dataset_cache(
                cached['path'], cache_sha256=cached['complete_sha256'], source_directory=item['dataset'],
                active_coefficients=initial, **cached['binding'])
            _validate_c_dataset(dataset, item, iq, multiplicity)
            _require(dataset.active_primitive_reduction.mapping_sha256
                     == cached['binding']['identifiers']['mapping_sha256'], 'cache mapping mismatch')
            datasets.append(dataset)
            records.append(dict(label=item['label'], seconds=time.perf_counter()-started,
                mapping_sha256=dataset.active_primitive_reduction.mapping_sha256))
            print(json.dumps(records[-1]), flush=True)
        check(freeze, digest)
        check(active_cache_index, active_cache_index_sha256)
        for item in frozen['datasets']:
            for source, expected in _source_checks(item):
                check(source, expected)
        # No physical evaluation until all eight cached q datasets have passed.
        guard = guard_factory(datasets[0], initial, **guard_options)
        save_json(output/'INITIAL_BAND_SCREEN.json', guard(initial))
        return tuple(datasets), records, guard
    for item, iq, multiplicity in zip(frozen['datasets'], indices, mult):
        root = Path(item['dataset'])
        checks = _source_checks(item)
        for path, expected in checks:
            check(path, expected)
        save_json(output/'PROGRESS.json', dict(stage='loading_q', label=item['label'], completed=records))
        started = time.perf_counter()
        dataset = read_periodic_galerkin_dataset(
            root, include_reference_projection=False, verify_omitted_chunks=True,
            active_coefficients=initial)
        for path, expected in checks:
            check(path, expected)
        _validate_c_dataset(dataset, item, iq, multiplicity)
        if persist_active_cache:
            cache_path = output/'active-data-cache'/('q'+str(item['label']))
            binding = dict(manifest_sha256=item['manifest_sha256'],
                           status_sha256=item['status_sha256'],
                           identifiers=dict(freeze_sha256=digest,
                               acceptance_sha256=item['acceptance_sha256'],
                               mapping_sha256=dataset.active_primitive_reduction.mapping_sha256))
            cache_started = time.perf_counter()
            cache_hash = write_periodic_galerkin_dataset_cache(
                cache_path, dataset, source_directory=root, **binding)
            del dataset
            dataset = read_periodic_galerkin_dataset_cache(
                cache_path, cache_sha256=cache_hash, source_directory=root,
                active_coefficients=initial, **binding)
            cache_records.append(dict(label=item['label'], path=str(cache_path.resolve()),
                                      complete_sha256=cache_hash, binding=binding,
                                      save_and_reload_seconds=time.perf_counter()-cache_started))
            save_json(output/'ACTIVE_DATA_CACHE.json', dict(
                status='success' if len(cache_records) == 8 else 'building',
                scope='exact_active_dataset_derivative_not_new_reference', records=cache_records))
        if not datasets:
            guard = guard_factory(dataset, initial, **guard_options)
            save_json(output/'INITIAL_BAND_SCREEN.json', guard(initial))
        datasets.append(dataset)
        records.append(dict(label=item['label'], seconds=time.perf_counter()-started,
                            mapping_sha256=dataset.active_primitive_reduction.mapping_sha256))
        print(json.dumps(records[-1]), flush=True)
    return tuple(datasets), records, guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'freeze-sha256', 'coefficients', 'coefficients-sha256',
                 'benchmark', 'benchmark-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--persist-active-cache', action='store_true')
    parser.add_argument('--active-cache-index')
    parser.add_argument('--active-cache-index-sha256')
    args = parser.parse_args()
    _require(1 <= args.max_steps <= 50, 'bounded calibration permits 1..50 steps')
    _cache_options(args.active_cache_index, args.active_cache_index_sha256, args.persist_active_cache)
    stopping = iteration_settings(args.max_steps)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        check(args.benchmark, args.benchmark_sha256)
        benchmark = json.loads(Path(args.benchmark).read_text())
        _require(benchmark['status'] == 'success' and all(r['equivalence_gate'] == 'pass'
                 for r in benchmark['rows']), 'same-input full-gradient benchmark must pass')
        _require(benchmark['coefficients_sha256'] == args.coefficients_sha256
                 and benchmark['input']['parent_freeze_sha256'] == args.freeze_sha256,
                 'benchmark and optimization must use the same frozen inputs')
        fastest = benchmark['fastest']
        torch.set_num_threads(fastest['threads'])
        check(args.coefficients, args.coefficients_sha256)
        initial = read_periodic_optimizer_coefficients(
            args.coefficients, element='C', radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
        datasets, load_records, guard = load_frozen_c(
            args.freeze, args.freeze_sha256, initial, output, args.persist_active_cache,
            active_cache_index=args.active_cache_index,
            active_cache_index_sha256=args.active_cache_index_sha256)
        load_seconds = time.perf_counter()-started
        initial_guard = guard(initial)
        save_json(output/'INITIAL_BAND_SCREEN.json', initial_guard)
        last = [time.perf_counter()]

        def progress(row):
            row = dict(row)
            now = time.perf_counter()
            row['elapsed_since_previous_record_seconds'] = now-last[0]
            last[0] = now
            with (output/'OPTIMIZATION_HISTORY.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            save_json(output/'PROGRESS.json', dict(stage='optimizing', **row))
            print(json.dumps({k: row[k] for k in ('step', 'loss', 'minimum_occupied_capture',
                                                 'elapsed_since_previous_record_seconds')}), flush=True)

        fit_start = time.perf_counter()
        fit = optimize_periodic_galerkin_basis(
            datasets, initial, fixed_nu={'C': (0, 0, 0, 0, 0)},
            objective='rpa', rpa_weights={'pi_weight': 1., 'trace_log_weight': 1., 'energy_weight': 1.},
            learning_rate=0.001, **stopping, occupied_capture_reference='initial_candidate',
            occupied_capture_degradation_tolerance=1e-4, maximum_backtracks=12,
            frequency_batch_size=fastest['frequency_batch_size'], cache_rpa_reference=True,
            coefficient_guard=guard, progress_callback=progress,
            best_callback=lambda step, loss, coeff: write_best_checkpoint(
                output, step, loss, coeff, objective='rpa'))
        final = output/'ORBITAL_RESULTS.txt'
        write_periodic_optimizer_coefficients(final, fit.coefficients)
        _require(final.read_bytes() == (output/'BEST_ORBITAL_CHECKPOINT.txt').read_bytes(),
                 'best/final checkpoint mismatch')
        final_guard = guard(fit.coefficients)
        orbital = output/'C_3s3p2d_optimized.orb'
        write_abacus_orbital(orbital, fit.coefficients, element='C', ecut_ry=100.,
                            rcut_bohr=10., dr_bohr=0.01, smoothing_sigma_bohr=0.1)
        first, best = fit.history[0], fit.history[fit.best_step]
        result = dict(status='success', scope='optimized_full_q_frozen_body_rpa_calibration',
                      iteration_settings=stopping,
                      fixed_nu=[0, 0, 0, 0, 0], nu=[3, 3, 2, 0, 0], ao_per_C=22,
                      initial=first, best=best, steps_completed=fit.steps_completed,
                      best_step=fit.best_step, stop_reason=fit.stop_reason,
                      total_backtracks=fit.total_backtracks,
                      initial_band_screen=initial_guard, best_band_screen=final_guard,
                      actual_scf_pbe_gate='pending', ordinary_sos_qavg_gate='pending',
                      mother_pca_sufficiency_gate='pending', physical_release_gate='hold',
                      training_weights={'pi_weight': 1., 'trace_log_weight': 1., 'energy_weight': 1.},
                      capture_degradation=1e-4, capture_limit_scope='numerical_only_not_10meV_equivalence',
                      benchmark_sha256=args.benchmark_sha256, configuration=fastest,
                      coefficient_sha256=_sha256(str(final)), orbital_sha256=_sha256(str(orbital)),
                      freeze_sha256=args.freeze_sha256, initial_sha256=args.coefficients_sha256,
                      load_records=load_records, load_seconds=load_seconds,
                      optimization_seconds=time.perf_counter()-fit_start,
                      total_seconds=time.perf_counter()-started,
                      active_cache_mode=('reuse' if args.active_cache_index is not None else
                                         'persist' if args.persist_active_cache else 'disabled'),
                      active_cache_index_path=(str(Path(args.active_cache_index).resolve())
                                              if args.active_cache_index is not None else
                                              str((output/'ACTIVE_DATA_CACHE.json').resolve())
                                              if args.persist_active_cache else None),
                      active_cache_index_sha256=(args.active_cache_index_sha256
                                                 if args.active_cache_index is not None else
                                                 _sha256(str(output/'ACTIVE_DATA_CACHE.json'))
                                                 if args.persist_active_cache else None),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        for path, expected in ((args.freeze, args.freeze_sha256),
                               (args.coefficients, args.coefficients_sha256),
                               (args.benchmark, args.benchmark_sha256)):
            check(path, expected)
        if args.active_cache_index is not None:
            check(args.active_cache_index, args.active_cache_index_sha256)
        save_json(output/'RESULT.json', result)
        (output/'STATUS').write_text('success\n')
    except Exception:
        (output/'STATUS').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
