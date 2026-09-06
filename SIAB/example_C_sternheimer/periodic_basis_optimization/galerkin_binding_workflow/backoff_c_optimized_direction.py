"""Evaluate one retracted interpolation of an accepted C direction, without fitting."""

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / 'opt_orb_pytorch_dpsi'))

import torch
from check_c_optimized_pbe import (
    _optimizer_artifacts, _read_hashed, _sha256, _write_json, _match_parent_initial, _require_backoff_improvement,
    BACKOFF_SCOPE, BACKOFF_RETRACTION, HARTREE_TO_EV,
)
from export_periodic_orbitals import write_abacus_orbital
from optimize_c_all_radial_fast import load_frozen_c
from periodic_galerkin_basis import read_periodic_optimizer_coefficients, write_periodic_optimizer_coefficients
from periodic_galerkin_fit import (
    _retract_variables, _global_rpa_loss, _prepare_block_contraction_caches, _evaluate_coefficient_guard,
)
from periodic_galerkin_rpa import prepare_periodic_rpa_reference
from periodic_galerkin_sternheimer import prepare_periodic_occupied_reference


SCOPE = BACKOFF_SCOPE
RETRACTION = BACKOFF_RETRACTION


def _alpha(value, endpoints=False):
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            or not (0 <= value <= 1 if endpoints else 0 < value < 1)):
        raise ValueError('alpha must be finite and inside ' + ('[0,1]' if endpoints else '(0,1)'))
    return float(value)


def retract_coefficients(coefficients):
    """Use the fitter's exact all-free signed QR, without creating Parameters."""
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError('nonempty coefficient dictionary required')
    fixed, variable = {}, {}
    for element, channels in coefficients.items():
        if not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError('coefficient channels must be nonempty')
        fixed[element], variable[element] = [], []
        for c in channels:
            if (not isinstance(c, torch.Tensor) or c.ndim != 2 or c.device.type != 'cpu'
                    or c.dtype != torch.float64 or c.shape[0] < 1 or c.shape[1] > c.shape[0]
                    or not bool(torch.isfinite(c).all())):
                raise ValueError('coefficients must be finite full-rank CPU float64 matrices')
            fixed[element].append(c.new_empty((c.shape[0], 0)))
            variable[element].append(c.detach().clone())
    _retract_variables(fixed, variable)
    return variable


def interpolate_direction(initial, best, alpha):
    """Endpoint values are allowed for algebra tests; production requires 0<a<1."""
    alpha = _alpha(alpha, endpoints=True)
    initial = retract_coefficients(initial)
    # Validate the endpoint without changing its accepted coefficient frame.
    retract_coefficients(best)
    if set(initial) != set(best) or any(len(initial[e]) != len(best[e]) for e in initial):
        raise ValueError('initial and best coefficient shapes differ')
    mixed = {}
    with torch.no_grad():
        for element in initial:
            mixed[element] = []
            for a, b in zip(initial[element], best[element]):
                if a.shape != b.shape:
                    raise ValueError('initial and best coefficient shapes differ')
                mixed[element].append((1 - alpha) * a + alpha * b.detach())
    return retract_coefficients(mixed)


def evaluate_pair(datasets, initial, candidate, *, guard, frequency_batch_size, weights):
    """Exactly two full response evaluations, with the same frozen reference."""
    with torch.no_grad():
        datasets = tuple(prepare_periodic_occupied_reference(dataset) for dataset in datasets)
        datasets = _prepare_block_contraction_caches(datasets, initial, 1)
        reference = prepare_periodic_rpa_reference(datasets)

        def evaluate(coefficients, tolerance):
            started = time.perf_counter()
            screen = _evaluate_coefficient_guard(guard, coefficients)
            loss, capture, condition, _, diagnostics = _global_rpa_loss(
                datasets, coefficients, occupied_capture_tolerance=tolerance,
                weights=weights, frequency_batch_size=frequency_batch_size, reference_cache=reference)
            energy, reference_energy = diagnostics['candidate_energy_ha'], diagnostics['reference_energy_ha']
            return dict(loss=float(loss), minimum_occupied_capture=capture, maximum_overlap_condition=condition,
                evaluation_seconds=time.perf_counter() - started,
                coefficient_guard=screen, rpa=diagnostics,
                energy_quantity='frozen_body_RPA_correlation_not_PBE_total',
                rpa_correlation_energy_ev_per_cell=energy * HARTREE_TO_EV,
                rpa_correlation_energy_ev_per_c=energy * HARTREE_TO_EV / 2,
                reference_rpa_correlation_energy_ev_per_cell=reference_energy * HARTREE_TO_EV,
                reference_rpa_correlation_energy_ev_per_c=reference_energy * HARTREE_TO_EV / 2)

        first = evaluate(initial, 1 - 1e-12)
        floor = max(0., first['minimum_occupied_capture'] - 1e-4)
        tolerance = min(1 - 1e-15, max(1e-15, 1 - floor))
        return first, evaluate(candidate, tolerance)


def prepare_backoff(*, initial, initial_sha256, parent_result, parent_result_sha256,
                    parent_best_metadata_sha256, freeze, freeze_sha256, active_cache_index,
                    active_cache_index_sha256, output, alpha=.25):
    started = time.perf_counter()
    alpha = _alpha(alpha)
    output, parent_root = Path(output), Path(parent_result).parent
    if output.exists():
        raise FileExistsError(output)
    locked = {
        'ORIGINAL_COEFFICIENTS.txt': (Path(initial), initial_sha256),
        'INPUT_FREEZE.json': (Path(freeze), freeze_sha256),
        'ACTIVE_DATA_CACHE.json': (Path(active_cache_index), active_cache_index_sha256),
        'parent/RESULT.json': (Path(parent_result), parent_result_sha256),
        'parent/BEST_CHECKPOINT.json': (parent_root / 'BEST_CHECKPOINT.json', parent_best_metadata_sha256),
    }
    evidence = {name: _read_hashed(path, digest) for name, (path, digest) in locked.items()}
    parent = json.loads(evidence['parent/RESULT.json'])
    if parent.get('scope') != 'optimized_full_q_frozen_body_rpa_calibration':
        raise ValueError('parent must be an original optimized direction, not a backoff')
    parent, orbital, artifacts = _optimizer_artifacts(Path(parent_result), parent_result_sha256)
    if (parent.get('initial_sha256') != initial_sha256 or parent.get('freeze_sha256') != freeze_sha256
            or parent.get('active_cache_index_sha256') != active_cache_index_sha256):
        raise ValueError('parent initial/freeze/cache SHA256 binding mismatch')
    for name in ('ORBITAL_RESULTS.txt', 'BEST_ORBITAL_CHECKPOINT.txt'):
        evidence['parent/' + name] = artifacts[name]
        locked['parent/' + name] = (parent_root / name, parent['coefficient_sha256'])
    evidence['parent/C_3s3p2d_optimized.orb'] = orbital
    locked['parent/C_3s3p2d_optimized.orb'] = (parent_root / 'C_3s3p2d_optimized.orb', parent['orbital_sha256'])
    if artifacts['BEST_CHECKPOINT.json'] != evidence['parent/BEST_CHECKPOINT.json']:
        raise ValueError('parent best metadata changed during validation')
    original = read_periodic_optimizer_coefficients(initial, element='C', radial_rows=31,
                                                     expected_nu=(3, 3, 2, 0, 0))
    best = read_periodic_optimizer_coefficients(parent_root / 'BEST_ORBITAL_CHECKPOINT.txt',
                                                element='C', radial_rows=31, expected_nu=(3, 3, 2, 0, 0))
    initial_frame = retract_coefficients(original)
    candidate = interpolate_direction(original, best, alpha)
    configuration = parent['configuration']
    threads = configuration['threads']
    if type(threads) is not int or threads <= 0:
        raise ValueError('invalid parent thread configuration')
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        load_start = time.perf_counter()
        datasets, load_records, guard = load_frozen_c(freeze, freeze_sha256, original, output,
            active_cache_index=active_cache_index, active_cache_index_sha256=active_cache_index_sha256)
        load_seconds = time.perf_counter() - load_start
        evaluation_start = time.perf_counter()
        first, evaluated = evaluate_pair(datasets, initial_frame, candidate, guard=guard,
            frequency_batch_size=configuration['frequency_batch_size'], weights=parent['training_weights'])
        evaluation_seconds = time.perf_counter() - evaluation_start
    _match_parent_initial(first, parent['initial'])
    _require_backoff_improvement(first, evaluated)
    for path, digest in locked.values():
        _read_hashed(path, digest)
    coefficient_path = output / 'INTERPOLATED_COEFFICIENTS.txt'
    write_periodic_optimizer_coefficients(coefficient_path, candidate)
    orbital_path = output / 'C_3s3p2d_interpolated.orb'
    write_abacus_orbital(orbital_path, candidate, element='C', ecut_ry=100., rcut_bohr=10.,
                        dr_bohr=.01, smoothing_sigma_bohr=.1)
    initial_coefficient_path = output / 'INITIAL_RETRACTED_COEFFICIENTS.txt'
    write_periodic_optimizer_coefficients(initial_coefficient_path, initial_frame)
    initial_orbital_path = output / 'C_3s3p2d_initial_retracted.orb'
    write_abacus_orbital(initial_orbital_path, initial_frame, element='C', ecut_ry=100., rcut_bohr=10.,
                        dr_bohr=.01, smoothing_sigma_bohr=.1)
    for name, content in evidence.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
    provenance = dict(scope=SCOPE, alpha=alpha, retraction=RETRACTION,
        parent_result_sha256=parent_result_sha256, parent_best_metadata_sha256=parent_best_metadata_sha256,
        parent_best_coefficient_sha256=parent['coefficient_sha256'], initial_sha256=initial_sha256,
        freeze_sha256=freeze_sha256, active_cache_index_sha256=active_cache_index_sha256,
        coefficient_sha256=_sha256(coefficient_path.read_bytes()), orbital_sha256=_sha256(orbital_path.read_bytes()),
        initial_retracted_coefficient_sha256=_sha256(initial_coefficient_path.read_bytes()),
        initial_retracted_orbital_sha256=_sha256(initial_orbital_path.read_bytes()),
        evidence_sha256={name: _sha256(content) for name, content in evidence.items()})
    _write_json(output / 'BACKOFF_PROVENANCE.json', provenance)
    result = dict(status='success', scope=SCOPE, alpha=alpha, retraction=RETRACTION,
        nu=[3, 3, 2, 0, 0], fixed_nu=[0, 0, 0, 0, 0], ao_per_C=22,
        initial=first, candidate=evaluated, parent_result_sha256=parent_result_sha256,
        parent_best_metadata_sha256=parent_best_metadata_sha256, initial_sha256=initial_sha256,
        freeze_sha256=freeze_sha256, active_cache_index_sha256=active_cache_index_sha256,
        coefficient_sha256=provenance['coefficient_sha256'], orbital_sha256=provenance['orbital_sha256'],
        initial_retracted_coefficient_sha256=provenance['initial_retracted_coefficient_sha256'],
        initial_retracted_orbital_sha256=provenance['initial_retracted_orbital_sha256'],
        provenance_sha256=_sha256((output / 'BACKOFF_PROVENANCE.json').read_bytes()),
        configuration=configuration, training_weights=parent['training_weights'], load_records=load_records,
        cache_load_seconds=load_seconds, evaluation_seconds=evaluation_seconds,
        total_seconds=time.perf_counter() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform == 'darwin' else 1),
        capture_degradation=1e-4, actual_scf_pbe_gate='pending', physical_release_gate='hold')
    _write_json(output / 'RESULT.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('initial', 'initial-sha256', 'parent-result', 'parent-result-sha256',
                 'parent-best-metadata-sha256', 'freeze', 'freeze-sha256', 'active-cache-index',
                 'active-cache-index-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--alpha', type=float, default=.25)
    args = vars(parser.parse_args(argv))
    result = prepare_backoff(**args)
    print(json.dumps(dict(result=result, result_sha256=_sha256((Path(args['output']) / 'RESULT.json').read_bytes())),
                     allow_nan=False))


if __name__ == '__main__':
    main()
