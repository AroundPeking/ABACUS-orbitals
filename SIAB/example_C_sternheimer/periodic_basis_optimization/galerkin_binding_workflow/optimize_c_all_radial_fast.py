"""Bounded full-q C optimization with free s/p/d radials and SCF release held."""

import argparse
import json
from pathlib import Path
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


def load_frozen_c(freeze, digest, initial, output, persist_active_cache=False):
    check(freeze, digest)
    frozen = json.loads(Path(freeze).read_text())
    _require(frozen['status'] == 'success', 'accepted input freeze required')
    labels, indices, mult = [1, 2, 3, 6, 7, 8, 11, 28], [1, 22, 43, 6, 27, 23, 11, 55], [1, 8, 4, 6, 24, 12, 3, 6]
    _require([r['label'] for r in frozen['datasets']] == labels, 'canonical eight-q order required')
    datasets, records, cache_records = [], [], []
    for item, iq, multiplicity in zip(frozen['datasets'], indices, mult):
        root = Path(item['dataset'])
        checks = ((root/'manifest.dat', item['manifest_sha256']),
                  (root/'status.dat', item['status_sha256']),
                  (Path(item['acceptance']), item['acceptance_sha256']))
        for path, expected in checks:
            check(path, expected)
        save_json(output/'PROGRESS.json', dict(stage='loading_q', label=item['label'], completed=records))
        started = time.perf_counter()
        dataset = read_periodic_galerkin_dataset(
            root, include_reference_projection=False, verify_omitted_chunks=True,
            active_coefficients=initial)
        for path, expected in checks:
            check(path, expected)
        _require(dataset.selected_iq == iq == item['selected_iq']
                 and dataset.q_weight == multiplicity/64. == item['q_weight']
                 and dataset.physics_hash == item['physics_hash'], 'q identity mismatch')
        _require(len(dataset.kpoints) == 64 and dataset.frequency_ha.numel() == 12
                 and dataset.primitive_count == 558
                 and dataset.active_primitive_reduction.original_primitive_count == 1550,
                 'full-k/frequency/exact-reduction contract mismatch')
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
            guard = prepare_frozen_band_guard(dataset, initial, atoms_per_cell=2)
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
    args = parser.parse_args()
    _require(1 <= args.max_steps <= 50, 'bounded calibration permits 1..50 steps')
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
            args.freeze, args.freeze_sha256, initial, output, args.persist_active_cache)
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
            learning_rate=0.001, max_steps=args.max_steps, minimum_steps=min(20, args.max_steps),
            plateau_patience=10, occupied_capture_reference='initial_candidate',
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
                      active_cache_index_sha256=(_sha256(str(output/'ACTIVE_DATA_CACHE.json'))
                                                 if args.persist_active_cache else None),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        for path, expected in ((args.freeze, args.freeze_sha256),
                               (args.coefficients, args.coefficients_sha256),
                               (args.benchmark, args.benchmark_sha256)):
            check(path, expected)
        save_json(output/'RESULT.json', result)
        (output/'STATUS').write_text('success\n')
    except Exception:
        (output/'STATUS').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
