"""Sequential frozen-C Pi residual audit; no optimizer, SCF, or new response."""
import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

OPT = Path(__file__).resolve().parents[3] / 'opt_orb_pytorch_dpsi'
sys.path.insert(0, str(OPT))

import torch

from periodic_galerkin_data import _sha256, _require, read_periodic_galerkin_dataset
from periodic_galerkin_basis import read_periodic_optimizer_coefficients
from periodic_galerkin_fit import _validate_inputs, _retract_variables, _assemble, _prepare_block_contraction_caches
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference
from periodic_galerkin_optimization import evaluate_periodic_galerkin_coefficient_response
from periodic_galerkin_rpa import periodic_rpa_objective
from periodic_galerkin_matrix_diagnostics import periodic_rpa_matrix_diagnostics


@torch.no_grad()
def evaluate_matrix_q(dataset, initial, fixed_nu):
    fixed, variable, _ = _validate_inputs((dataset,), initial, fixed_nu)
    dataset = prepare_periodic_occupied_reference(dataset)
    _retract_variables(fixed, variable)
    coefficients = _assemble(fixed, variable)
    (dataset,) = _prepare_block_contraction_caches((dataset,), coefficients, workers=1)
    response = evaluate_periodic_galerkin_coefficient_response(
        dataset, coefficients, contraction_backend='block',
        occupied_capture_tolerance=1.-1.e-12)
    matrices = periodic_rpa_matrix_diagnostics((dataset,), (response.response,))
    objective = periodic_rpa_objective((dataset,), (response.response,))
    record = dict(matrices['per_q'][0])
    record.update(qpoint=list(dataset.qpoint),
        raw_auxiliary_dimension=dataset.raw_auxiliary_dimension,
        whitened_auxiliary_rank=dataset.whitened_auxiliary_rank,
        minimum_occupied_capture=response.minimum_occupied_capture,
        maximum_overlap_condition=response.maximum_overlap_condition,
        candidate_contributions_ha=objective.q_records[0].candidate_contributions_ha.tolist(),
        reference_contributions_ha=objective.q_records[0].reference_contributions_ha.tolist(),
        metric_definition=matrices['metric_definition'],
        whitening_sha256_definition=matrices['whitening_sha256_definition'])
    return record


def merge_and_compare(records, initial):
    expected = initial['rpa']['per_q']
    _require([r['selected_iq'] for r in records] == [r['selected_iq'] for r in expected]
             and len({r['selected_iq'] for r in records}) == len(records), 'q identity mismatch')
    differences = {}

    def close(name, a, b):
        _require(math.isfinite(a) and math.isfinite(b)
                 and math.isclose(a, b, rel_tol=1.e-9, abs_tol=1.e-11), 'baseline equivalence: '+name)
        differences[name] = abs(a-b)

    for record, old in zip(records, expected):
        _require(record['q_weight'] == old['q_weight'], 'q weight mismatch')
        for name in ('frequency_ha', 'candidate_contributions_ha', 'reference_contributions_ha'):
            _require(len(record[name]) == len(old[name]), 'frequency count mismatch')
            for iw, (a, b) in enumerate(zip(record[name], old[name])):
                close('q{}_{}_{}'.format(record['selected_iq'], name, iw), a, b)
    numerator = math.fsum(r['pi_numerator'] for r in records)
    denominator = math.fsum(r['pi_denominator'] for r in records)
    _require(denominator > 0, 'zero global Pi reference norm')
    close('pi_relative_squared_error', numerator/denominator, initial['rpa']['pi_relative_squared_error'])
    close('minimum_occupied_capture', min(r['minimum_occupied_capture'] for r in records), initial['minimum_occupied_capture'])
    close('maximum_overlap_condition', max(r['maximum_overlap_condition'] for r in records), initial['maximum_overlap_condition'])
    finite_q = [r for r in records if any(x != 0.0 for x in r['qpoint'])]
    worst = min(finite_q, key=lambda r: (-r['pi_numerator'], r['selected_iq'])) if finite_q else None
    return dict(status='success', baseline_equivalence_gate='pass', absolute_differences=differences,
        scope='candidate_minus_frozen_reference_not_mother_or_cross_pca',
        metric_definition=records[0]['metric_definition'], optimizer_steps=0,
        whitening_sha256_definition=records[0]['whitening_sha256_definition'],
        pi_numerator=numerator, pi_denominator=denominator, pi_relative_squared_error=numerator/denominator,
        q_weight_coverage=math.fsum(r['q_weight'] for r in records), per_q=records,
        largest_weighted_residual_finite_q=worst['selected_iq'] if worst else None,
        formal_optimization_gate='hold', physical_release_gate='hold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('freeze', 'freeze-sha256', 'accepted-initial', 'accepted-initial-sha256',
                'coefficients', 'coefficients-sha256', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    def check(path, digest):
        _require(_sha256(str(path)) == digest, 'SHA256 mismatch: '+str(path))

    pins = ((args.freeze, args.freeze_sha256), (args.accepted_initial, args.accepted_initial_sha256),
            (args.coefficients, args.coefficients_sha256))
    try:
        for path, digest in pins:
            check(path, digest)
        frozen = json.loads(Path(args.freeze).read_text())
        baseline = json.loads(Path(args.accepted_initial).read_text())
        _require(frozen['status'] == baseline['status'] == 'success'
                 and baseline['comparison']['equivalence_gate'] == 'pass'
                 and baseline['freeze_sha256'] == args.freeze_sha256
                 and baseline['coefficients_sha256'] == args.coefficients_sha256
                 and baseline['evaluation']['optimizer_steps'] == 0, 'accepted initial contract mismatch')
        labels, indices, multiplicities = [1, 2, 3, 6, 7, 8, 11, 28], [1, 22, 43, 6, 27, 23, 11, 55], [1, 8, 4, 6, 24, 12, 3, 6]
        _require([q['label'] for q in frozen['datasets']] == labels, 'canonical eight q required')
        coefficients = read_periodic_optimizer_coefficients(
            args.coefficients, element='C', radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
        records, shared = [], None
        for item, iq, mult in zip(frozen['datasets'], indices, multiplicities):
            root = Path(item['dataset'])
            evidence = ((root/'manifest.dat', item['manifest_sha256']),
                        (root/'status.dat', item['status_sha256']),
                        (Path(item['acceptance']), item['acceptance_sha256']))
            for path, digest in evidence:
                check(path, digest)
            save('PROGRESS.json', dict(stage='reading_and_evaluating_q', label=item['label'], completed=records))
            qstart = time.monotonic()
            dataset = read_periodic_galerkin_dataset(root, include_reference_projection=False,
                                                    verify_omitted_chunks=True, active_coefficients=coefficients)
            _require(dataset.selected_iq == iq == item['selected_iq']
                     and dataset.q_weight == mult/64. == item['q_weight']
                     and dataset.physics_hash == item['physics_hash'], 'frozen q contract mismatch')
            _require(len(dataset.kpoints) == 64 and dataset.frequency_ha.numel() == 12
                     and dataset.primitive_count == 558
                     and dataset.active_primitive_reduction.original_primitive_count == 1550,
                     'full k/frequency/primitive contract mismatch')
            protocol = tuple(getattr(dataset, k) for k in ('abacus_commit', 'executable_sha256',
                'orbital_sha256', 'pseudopotential_sha256', 'auxiliary_basis_sha256',
                'primitive_blocks_sha256', 'q_count')) + (tuple(dataset.frequency_ha.tolist()),
                tuple(dataset.frequency_weights_ha.tolist()))
            _require(shared is None or protocol == shared, 'cross-q protocol mismatch')
            shared = protocol
            record = evaluate_matrix_q(dataset, coefficients, {'C': (2, 2, 1, 0, 0)})
            record.update(label=item['label'], elapsed_seconds=time.monotonic()-qstart,
                          process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            del dataset
            for path, digest in evidence:
                check(path, digest)
            records.append(record)
            save('q{}.json'.format(item['label']), record)
            print(json.dumps(dict(label=item['label'], elapsed_seconds=record['elapsed_seconds'],
                                  pi_numerator=record['pi_numerator'])), flush=True)
        report = merge_and_compare(records, baseline['evaluation'])
        _require(report['q_weight_coverage'] == 1., 'complete q weight required')
        for path, digest in pins:
            check(path, digest)
        report.update(freeze_sha256=args.freeze_sha256, coefficients_sha256=args.coefficients_sha256,
            accepted_initial_sha256=args.accepted_initial_sha256,
            sample_frequency_indices_zero_based=[0, 6, 11],
            total_seconds=time.monotonic()-started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save('RESULT.json', report)
        (output/'STATUS').write_text('success\n')
        save('PROGRESS.json', dict(stage='completed', baseline_equivalence_gate='pass'))
    except Exception:
        (output/'STATUS').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
