"""One complete-q initial evaluation, with no optimizer step or new physics."""

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

OPT = Path(__file__).resolve().parents[3]/'opt_orb_pytorch_dpsi'
sys.path.insert(0, str(OPT))

import torch

from periodic_galerkin_data import _sha256, _require, read_periodic_galerkin_dataset
from periodic_galerkin_basis import read_periodic_optimizer_coefficients
from periodic_galerkin_fit import (
    _validate_inputs, _retract_variables, _assemble,
    _prepare_block_contraction_caches, _global_rpa_loss,
)
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


def evaluate_initial(datasets, initial, fixed_nu):
    started = time.monotonic()
    fixed, variable, parameters = _validate_inputs(datasets, initial, fixed_nu)
    datasets = tuple(prepare_periodic_occupied_reference(q) for q in datasets)
    _retract_variables(fixed, variable)
    coefficients = _assemble(fixed, variable)
    datasets = _prepare_block_contraction_caches(datasets, coefficients, workers=1)
    prepared_seconds = time.monotonic()-started
    evaluation_start = time.monotonic()
    loss, capture, condition, _, diagnostics = _global_rpa_loss(
        datasets, coefficients, occupied_capture_tolerance=1.-1.e-12,
        weights={'pi_weight': 1., 'trace_log_weight': 1., 'energy_weight': 1.})
    backward_start = time.monotonic()
    loss.backward()
    _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
             'initial gradient must be finite')
    norm = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters))
    return dict(loss=float(loss.detach()), minimum_occupied_capture=capture,
                maximum_overlap_condition=condition, gradient_norm=norm,
                rpa=diagnostics, optimizer_steps=0, record_count=sum(len(q.kpoints) for q in datasets),
                preparation_seconds=prepared_seconds,
                forward_backward_seconds=time.monotonic()-evaluation_start,
                backward_seconds=time.monotonic()-backward_start)


def compare_history(snapshot, history):
    _require(len(history) >= 2 and history[0].get('rpa') is not None
             and history[1].get('previous_step_gradient_norm') is not None,
             'accepted initial history and first gradient are required')
    initial = history[0]
    differences = {}

    def close(label, a, b, rtol=1.e-9, atol=1.e-11):
        _require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=rtol, abs_tol=atol),
                 'initial equivalence failed: '+label)
        differences[label] = abs(a-b)

    for key in ('loss', 'minimum_occupied_capture', 'maximum_overlap_condition'):
        close(key, snapshot[key], initial[key])
    close('gradient_norm', snapshot['gradient_norm'], history[1]['previous_step_gradient_norm'],
          rtol=1.e-8, atol=1.e-10)
    actual, expected = snapshot['rpa'], initial['rpa']
    _require(actual['complete_q_weight'] and expected['complete_q_weight'], 'complete q weight required')
    close('q_weight_coverage', actual['q_weight_coverage'], expected['q_weight_coverage'], rtol=0, atol=1.e-14)
    for key in ('pi_relative_squared_error', 'trace_log_relative_squared_error',
                'energy_relative_squared_error', 'candidate_energy_ha', 'reference_energy_ha'):
        close(key, actual[key], expected[key])
    _require(len(actual['per_q']) == len(expected['per_q']), 'q record count changed')
    for a, b in zip(actual['per_q'], expected['per_q']):
        _require(a['selected_iq'] == b['selected_iq'] and a['q_weight'] == b['q_weight'],
                 'q identity or weight changed')
        for key in ('frequency_ha', 'candidate_contributions_ha', 'reference_contributions_ha'):
            _require(len(a[key]) == len(b[key]), 'frequency record count changed')
            for iw, (x, y) in enumerate(zip(a[key], b[key])):
                close('q{}_{}_{}'.format(a['selected_iq'], key, iw), x, y)
    return dict(equivalence_gate='pass', absolute_differences=differences,
                scalar_tolerances=dict(rtol=1.e-9, atol=1.e-11),
                gradient_norm_tolerances=dict(rtol=1.e-8, atol=1.e-10))


def _check(path, expected):
    _require(_sha256(str(path)) == expected, 'SHA256 mismatch: '+str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'freeze-sha256', 'history', 'history-sha256',
                 'coefficients', 'coefficients-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    try:
        for path, digest in ((args.freeze, args.freeze_sha256), (args.history, args.history_sha256),
                             (args.coefficients, args.coefficients_sha256)):
            _check(path, digest)
        frozen = json.loads(Path(args.freeze).read_text())
        _require(frozen['status'] == 'success', 'frozen inputs must be accepted')
        labels = [1, 2, 3, 6, 7, 8, 11, 28]
        indices = [1, 22, 43, 6, 27, 23, 11, 55]
        multiplicities = [1, 8, 4, 6, 24, 12, 3, 6]
        _require([q['label'] for q in frozen['datasets']] == labels, 'canonical eight-q order required')
        history = [json.loads(line) for line in Path(args.history).read_text().splitlines() if line]
        coefficients = read_periodic_optimizer_coefficients(
            args.coefficients, element='C', radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
        datasets, load_records = [], []
        for item, iq, multiplicity in zip(frozen['datasets'], indices, multiplicities):
            root = Path(item['dataset'])
            checks = ((root/'manifest.dat', item['manifest_sha256']),
                      (root/'status.dat', item['status_sha256']),
                      (Path(item['acceptance']), item['acceptance_sha256']))
            for path, digest in checks:
                _check(path, digest)
            save('PROGRESS.json', dict(stage='loading_q', label=item['label'], completed=load_records))
            load_start = time.monotonic()
            q = read_periodic_galerkin_dataset(root, include_reference_projection=False,
                                               verify_omitted_chunks=True, active_coefficients=coefficients)
            for path, digest in checks:
                _check(path, digest)
            _require(q.selected_iq == iq == item['selected_iq'] and q.q_weight == multiplicity/64.
                     and q.q_weight == item['q_weight'] and q.physics_hash == item['physics_hash'],
                     'frozen q physics or weight mismatch')
            _require(len(q.kpoints) == 64 and q.frequency_ha.numel() == 12
                     and q.active_primitive_reduction.original_primitive_count == 1550
                     and q.primitive_count == 558, 'full k/frequency/primitive contract mismatch')
            datasets.append(q)
            load_records.append(dict(label=item['label'], selected_iq=q.selected_iq,
                load_reduce_seconds=time.monotonic()-load_start,
                mapping_sha256=q.active_primitive_reduction.mapping_sha256,
                operator_payload_bytes=sum(getattr(r, key).numel()*getattr(r, key).element_size()
                    for r in q.kpoints for key in ('overlap', 'hamiltonian_ha', 'source', 'occupied_projection')),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(json.dumps(load_records[-1]), flush=True)
        load_seconds = time.monotonic()-started
        save('PROGRESS.json', dict(stage='initial_forward_backward', completed=load_records))
        snapshot = evaluate_initial(tuple(datasets), coefficients, {'C': (2, 2, 1, 0, 0)})
        # Preserve numerical evidence even if the equivalence gate rejects it.
        save('INITIAL_EVALUATION.json', snapshot)
        comparison = compare_history(snapshot, history)
        for path, digest in ((args.freeze, args.freeze_sha256), (args.history, args.history_sha256),
                             (args.coefficients, args.coefficients_sha256)):
            _check(path, digest)
        save('RESULT.json', dict(status='success', scope='all_q_initial_cost_not_optimization_or_physical_acceptance',
            freeze_sha256=args.freeze_sha256, history_sha256=args.history_sha256,
            coefficients_sha256=args.coefficients_sha256, reader_mode='exact_active_per_k',
            evaluation=snapshot, comparison=comparison, load_records=load_records,
            load_seconds=load_seconds, total_seconds=time.monotonic()-started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            formal_optimization_gate='hold', physical_release_gate='hold'))
        (output/'STATUS').write_text('success\n')
        save('PROGRESS.json', dict(stage='completed', equivalence_gate='pass'))
    except Exception:
        (output/'STATUS').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
