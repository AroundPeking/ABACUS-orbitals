"""Read-only frozen-H response of the actually available primitive space.

Unlike the unreduced snapshot evaluator, this accepts an explicitly labelled
active cache and needs no Qref. It does not recover omitted primitive blocks,
claim a reference solution, or produce fitted NAO coefficients.
"""

import math

import numpy as np
import torch

from periodic_galerkin_sternheimer import _mother_overlap_orthogonalizer
from response_galerkin import galerkin_pi, occupied_frames
from response_metric import _hermitian


def _array(value):
    return value.detach().cpu().numpy()


def available_mother_response(dataset, *, relative_rank_tolerance=1e-12,
                              condition_limit=1e12, capture_tolerance=1e-6,
                              progress=None):
    """Sum the existing Hermitian per-k response convention, weights once.

    The metric is diagonally normalized before rank truncation, exactly as in
    the production mother evaluator. Negative metric modes are checked rather
    than silently discarded. The occupied normalizer only changes its row
    coordinates; it is never applied to source or occupation weights.
    """
    if (not 0 < relative_rank_tolerance < 1 or not math.isfinite(condition_limit)
            or condition_limit < 1 or not 0 <= capture_tolerance < 1):
        raise ValueError('invalid available-mother tolerances')
    reduction = dataset.active_primitive_reduction
    report = dict(
        space=('available_active_reduced_mother' if reduction is not None
               else 'available_unreduced_mother'),
        exact_reference_snapshot_fit=False, physical_release_gate='hold',
        available_primitive_count=dataset.primitive_count,
        original_primitive_count=(reduction.original_primitive_count if reduction
                                  else dataset.primitive_count),
        mapping_sha256=(reduction.mapping_sha256 if reduction else None),
        relative_rank_tolerance=relative_rank_tolerance,
        condition_limit=condition_limit, capture_tolerance=capture_tolerance,
        selected_iq=dataset.selected_iq, q_weight=dataset.q_weight,
        metric='diagonal_normalized_overlap',
        response_convention='sum_k_half_plus_adjoint_no_q_or_frequency_weight',
        reference_snapshot_count=sum(int(r.reference_projection.numel() > 0)
                                     for r in dataset.kpoints),
        k_records=[], k_weight_sum=0.)
    total = np.zeros(tuple(dataset.reference_response.shape), dtype=np.complex128)
    for record in dataset.kpoints:
        overlap, _ = _hermitian(_array(record.overlap))
        diagonal = overlap.diagonal().real
        if np.any(diagonal < 0) or not np.any(diagonal > 0):
            raise ValueError('overlap has no valid positive diagonal')
        active = diagonal > 0
        norms = np.sqrt(diagonal[active])
        normalized = overlap[np.ix_(active, active)] / norms[:, None] / norms[None, :]
        spectrum = np.linalg.eigvalsh(normalized)
        if spectrum[0] < -1e-10*max(spectrum[-1], 1e-300):
            raise ValueError('available mother overlap is materially indefinite')
        if np.any(~active) and np.linalg.norm(overlap[~active]) > 1e-12*np.linalg.norm(overlap):
            raise ValueError('null overlap diagonal has nonzero coupling')
        with torch.no_grad():
            lowdin, rank, condition = _mother_overlap_orthogonalizer(
                record.overlap, relative_rank_tolerance)
        if condition > condition_limit:
            raise ValueError('available mother overlap condition exceeds limit')
        lowdin = _array(lowdin)
        metric_residual = float(np.linalg.norm(lowdin.conj().T@overlap@lowdin-np.eye(rank)))
        if not math.isfinite(metric_residual) or metric_residual > 1e-3:
            raise ValueError('available mother orthogonalization residual exceeds limit')
        raw_occupied = _array(record.occupied_projection)@lowdin
        raw_capture = np.linalg.eigvalsh(raw_occupied@raw_occupied.conj().T)
        occupied = raw_occupied
        if record.occupied_projection_normalization is not None:
            occupied = _array(record.occupied_projection_normalization)@occupied
        _, virtual, captures = occupied_frames(occupied)
        if captures['minimum_capture'] < 1.-capture_tolerance:
            raise ValueError('available mother lost fixed occupied capture')
        hamiltonian, _ = _hermitian(_array(record.hamiltonian_ha))
        pi, spectral = galerkin_pi(
            lowdin.conj().T@hamiltonian@lowdin, _array(record.source)@lowdin,
            _array(record.source_eigenvalue_ha), _array(record.occupation),
            _array(dataset.frequency_ha), virtual, record.k_weight)
        if pi.shape != total.shape:
            raise ValueError('available mother response/reference dimensions differ')
        total += pi
        entry = dict(source_ik=record.source_ik, target_ik=record.target_ik,
                     reciprocal_shift=list(record.reciprocal_shift),
                     k_weight=record.k_weight, retained_rank=rank,
                     overlap_condition=condition, metric_residual=metric_residual,
                     normalized_overlap_minimum=float(spectrum[0]),
                     normalized_overlap_maximum=float(spectrum[-1]),
                     raw_minimum_capture=float(raw_capture[0]),
                     raw_maximum_capture=float(raw_capture[-1]), **captures)
        entry.update(spectral)
        report['k_records'].append(entry)
        report['k_weight_sum'] += record.k_weight
        if progress is not None:
            progress(entry)
    if not report['k_records'] or not np.isfinite(total).all():
        raise ValueError('empty or nonfinite available-mother response')
    return total, report
