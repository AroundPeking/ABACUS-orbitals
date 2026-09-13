"""Bounded shared-radial initialization on a fixed response target, not RPA.

All supplied radial subspaces are free. QR only removes within-l coefficient
gauge; it does not freeze a radial prefix or impose PBE. The first bounded
experiment deliberately keeps the effective virtual rank fixed.
"""
import math
import time

import torch

from response_radial_fit import shared_radial_fit_loss


def _retract(coefficients):
    result = {}
    for element, channels in coefficients.items():
        output = []
        for c in channels:
            c = c.detach().clone()
            if c.ndim != 2 or c.dtype != torch.float64 or not torch.isfinite(c).all():
                raise ValueError('finite real float64 radial coefficients required')
            if c.shape[1]:
                values = torch.linalg.svdvals(c)
                if c.shape[1] > c.shape[0] or values[-1] <= 1e-12*values[0]:
                    raise ValueError('initial radial rank is deficient')
                c = torch.linalg.qr(c, mode='reduced')[0]
            output.append(c.requires_grad_(True))
        result[element] = tuple(output)
    return result


def fit_shared_response_radials(initial, sectors, *, max_steps=5, max_evaluations=15,
                                radius=.05, progress=None):
    if (type(max_steps) is not int or max_steps <= 0
            or type(max_evaluations) is not int or max_evaluations <= 0
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError('invalid bounded response fit controls')
    start = time.perf_counter()
    current = _retract(initial)
    evaluations = 0
    history = []

    def evaluate(coefficients):
        nonlocal evaluations
        evaluations += 1
        value, details = shared_radial_fit_loss(coefficients, sectors)
        parameters = [c for channels in coefficients.values() for c in channels if c.shape[1]]
        gradients = torch.autograd.grad(value, parameters)
        horizontal = [(g-c@(c.T@g)).detach() for c, g in zip(parameters, gradients)]
        if not all(bool(torch.isfinite(g).all()) for g in horizontal):
            raise ValueError('nonfinite radial gradient')
        norm = math.sqrt(math.fsum(float(torch.sum(g*g)) for g in horizontal))
        return float(value.detach()), details, [g.detach() for g in horizontal], norm

    def record(value, accepted, step, step_radius, norm, reason):
        row = dict(evaluation=evaluations, step=step, loss=value, accepted=accepted,
            radius=step_radius, gradient_norm=norm, reason=reason,
            elapsed_seconds=time.perf_counter()-start)
        history.append(row)
        if progress is not None:
            progress(row)

    loss, details, gradient, norm = evaluate(current)
    initial_loss = loss
    rank_signature = details['virtual_rank_signature']
    record(loss, True, 0, 0., norm, 'initial')
    accepted_steps = 0
    stopping = 'evaluation_budget'
    while accepted_steps < max_steps and evaluations < max_evaluations:
        if norm <= 1e-10:
            stopping = 'snapshot_gradient_small_not_RPA_convergence'
            break
        step_radius = radius
        accepted = False
        while evaluations < max_evaluations and step_radius >= radius/256:
            direction = iter(gradient)
            trial = {element: tuple(c.detach()-step_radius*next(direction)/norm
                                    if c.shape[1] else c.detach() for c in channels)
                     for element, channels in current.items()}
            trial = _retract(trial)
            candidate_loss, candidate_details, candidate_gradient, candidate_norm = evaluate(trial)
            same_rank = candidate_details['virtual_rank_signature'] == rank_signature
            accepted = same_rank and candidate_loss <= loss-1e-4*step_radius*norm
            reason = 'accepted' if accepted else ('rank_changed' if not same_rank else 'insufficient_decrease')
            record(candidate_loss, accepted, accepted_steps+1, step_radius, candidate_norm, reason)
            if accepted:
                current, loss, details, gradient, norm = trial, candidate_loss, candidate_details, candidate_gradient, candidate_norm
                accepted_steps += 1
                break
            step_radius *= .5
        if not accepted:
            stopping = 'evaluation_budget' if evaluations >= max_evaluations else 'line_search_hold'
            break
    if accepted_steps == max_steps:
        stopping = 'max_steps_not_convergence'
    fitted = {e: tuple(c.detach().clone() for c in channels) for e, channels in current.items()}
    return fitted, dict(status='success', stage='bounded_snapshot_fit_not_RPA_acceptance',
        physical_release_gate='hold', initial_loss=initial_loss, final_loss=loss,
        accepted_steps=accepted_steps, evaluations=evaluations, stopping_reason=stopping,
        free_radial_count=sum(c.shape[1] for channels in current.values() for c in channels),
        old_radial_prefix_frozen=False, pbe_iteration_constraint=False,
        gradient_norm=norm, initial_radius=radius, virtual_rank_signature=rank_signature,
        final_dimensions=details, elapsed_seconds=time.perf_counter()-start, history=history)
