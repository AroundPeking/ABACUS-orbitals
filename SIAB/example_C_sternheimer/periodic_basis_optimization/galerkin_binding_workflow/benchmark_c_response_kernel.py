"""Bounded real-input timing with full-gradient equality, not a physics run."""

import argparse
import gc
import json
from pathlib import Path
import resource
import time

from compare_active_primitive_record import (
    _check_hash, _load_sample, read_periodic_optimizer_coefficients,
    prepare_periodic_occupied_reference, reduce_periodic_active_primitives,
    evaluate_periodic_galerkin_coefficient_response, periodic_rpa_objective,
)
from periodic_galerkin_fit import _prepare_block_contraction_caches
import torch


def benchmark(sample, initial):
    sample = prepare_periodic_occupied_reference(sample)
    sample = reduce_periodic_active_primitives(sample, initial)
    sample = _prepare_block_contraction_caches((sample,), initial, workers=1)[0]
    expected = None
    rows = []
    previous_threads = torch.get_num_threads()
    try:
        for threads, batch in ((30, None), (1, None), (2, None), (4, None),
                               (1, 3), (2, 3), (4, 3), (1, 12), (2, 12)):
            torch.set_num_threads(threads)
            elapsed = []
            for repeat in range(2):
                gc.collect()
                coefficients = {e: [v.detach().clone().requires_grad_(True) for v in cs]
                                for e, cs in initial.items()}
                started = time.perf_counter()
                response = evaluate_periodic_galerkin_coefficient_response(
                    sample, coefficients, contraction_backend='block',
                    frequency_batch_size=batch)
                objective = periodic_rpa_objective((sample,), (response.response,))
                forward = time.perf_counter() - started
                objective.loss.backward()
                elapsed.append(dict(forward_seconds=forward,
                                    total_seconds=time.perf_counter()-started))
                snapshot = [response.response.detach().clone(), objective.loss.detach().clone()]
                snapshot.extend(c.grad.detach().clone() for cs in coefficients.values()
                                for c in cs if c.numel())
                if expected is None:
                    expected = snapshot
                for a, b in zip(snapshot, expected):
                    torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
                gradient_error = max(float((a-b).abs().max())
                                     for a, b in zip(snapshot[2:], expected[2:]))
                del response, objective, coefficients, snapshot
            rows.append(dict(threads=threads, frequency_batch_size=batch,
                             timings=elapsed, equivalence_gate='pass',
                             max_gradient_absolute_difference=gradient_error))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    fastest = min(rows, key=lambda row: min(t['total_seconds'] for t in row['timings']))
    return dict(status='success', scope='single_k_kernel_timing_not_physical_energy',
                rows=rows, fastest=fastest,
                physical_release_gate='hold', optimizer_steps=0,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'freeze-sha256', 'coefficients', 'coefficients-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        _check_hash(args.coefficients, args.coefficients_sha256)
        torch.set_num_threads(2)
        sample, evidence = _load_sample(args.freeze, args.freeze_sha256, 1, 1)
        initial = read_periodic_optimizer_coefficients(
            args.coefficients, element='C', radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
        result = benchmark(sample, initial)
        _check_hash(args.coefficients, args.coefficients_sha256)
        _check_hash(args.freeze, args.freeze_sha256)
        result.update(input=evidence, coefficients_sha256=args.coefficients_sha256)
        (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        (output/'STATUS').write_text('success\n')
    except Exception:
        (output/'STATUS').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
