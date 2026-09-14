"""Bounded shared-radial initialization on a fixed response target, not RPA.

QR removes within-l coefficient gauge.  An optional occupied-embedding penalty
guides all-channel motion, while a hard capture floor rejects infeasible trial
steps.  A frozen prefix remains available only for explicit ablations.
"""
import math
import time

import torch

from response_radial_fit import shared_radial_fit_loss


def _validate_frozen_prefix(coefficients, frozen_prefix):
    if frozen_prefix is None:
        return {element: tuple(0 for _ in channels)
                for element, channels in coefficients.items()}
    if not isinstance(frozen_prefix, dict) or set(frozen_prefix) != set(coefficients):
        raise ValueError('frozen prefix contract must cover every element')
    result = {}
    for element, channels in coefficients.items():
        counts = frozen_prefix[element]
        if not isinstance(counts, (list, tuple)) or len(counts) != len(channels):
            raise ValueError('frozen prefix counts have the wrong shape')
        normalized = []
        for count, channel in zip(counts, channels):
            if (type(count) is not int or count < 0 or count > channel.shape[1]):
                raise ValueError('invalid frozen prefix count')
            if count:
                prefix = channel[:, :count].detach()
                singular = torch.linalg.svdvals(prefix)
                if singular.numel() == 0 or float(singular[-1]) <= 1e-12 * float(singular[0]):
                    raise ValueError('frozen prefix radial rank is deficient')
            normalized.append(count)
        result[element] = tuple(normalized)
    return result


def _retract(coefficients, frozen_counts=None):
    result = {}
    for element, channels in coefficients.items():
        output = []
        element_counts = ((0,) * len(channels) if frozen_counts is None
                          else frozen_counts[element])
        for c, frozen_count in zip(channels, element_counts):
            c = c.detach().clone()
            if c.ndim != 2 or c.dtype != torch.float64 or not torch.isfinite(c).all():
                raise ValueError('finite real float64 radial coefficients required')
            if c.shape[1]:
                if c.shape[1] > c.shape[0]:
                    raise ValueError('initial radial rank is deficient')
                if frozen_count:
                    prefix = torch.linalg.qr(c[:, :frozen_count], mode='reduced')[0]
                    identity = torch.eye(c.shape[0], dtype=torch.float64)
                    remainder = (identity-prefix@prefix.T)@c[:, frozen_count:]
                    if remainder.shape[1]:
                        values = torch.linalg.svdvals(remainder)
                        if values[-1] <= 1e-12*values[0]:
                            raise ValueError('free radial rank is deficient after prefix projection')
                        remainder = torch.linalg.qr(remainder, mode='reduced')[0]
                    c = torch.cat((prefix, remainder), dim=1)
                else:
                    values = torch.linalg.svdvals(c)
                    if values[-1] <= 1e-12*values[0]:
                        raise ValueError('initial radial rank is deficient')
                    c = torch.linalg.qr(c, mode='reduced')[0]
            output.append(c.requires_grad_(True))
        result[element] = tuple(output)
    return result


def fit_shared_response_radials(initial, sectors, *, max_steps=5, max_evaluations=15,
                                radius=.05, progress=None, frozen_prefix=None,
                                occupied_weight=0., occupied_capture_floor=None):
    if (type(max_steps) is not int or max_steps <= 0
            or type(max_evaluations) is not int or max_evaluations <= 0
            or not math.isfinite(radius) or radius <= 0
            or not math.isfinite(occupied_weight) or occupied_weight < 0
            or (occupied_capture_floor is not None
                and (not math.isfinite(occupied_capture_floor)
                     or not 0 <= occupied_capture_floor <= 1))):
        raise ValueError('invalid bounded response fit controls')
    start = time.perf_counter()
    frozen_counts = _validate_frozen_prefix(initial, frozen_prefix)
    current = _retract(initial, frozen_counts)
    evaluations = 0
    history = []

    def evaluate(coefficients):
        nonlocal evaluations
        evaluations += 1
        value, details = shared_radial_fit_loss(
            coefficients, sectors, occupied_weight=occupied_weight)
        if (occupied_capture_floor is not None
                and details['minimum_occupied_capture'] is None):
            raise ValueError('occupied capture constraint data are unavailable')
        parameter_meta = [(element, l, c)
                          for element, channels in coefficients.items()
                          for l, c in enumerate(channels) if c.shape[1]]
        parameters = [c for _, _, c in parameter_meta]
        gradients = torch.autograd.grad(value, parameters)
        horizontal = []
        for (element, l, c), g in zip(parameter_meta, gradients):
            frozen_count = frozen_counts[element][l]
            if frozen_count:
                direction = torch.zeros_like(g)
                if frozen_count < c.shape[1]:
                    free = g[:, frozen_count:]
                    prefix = c[:, :frozen_count]
                    direction[:, frozen_count:] = free-prefix@(prefix.T@free)
                horizontal.append(direction.detach())
            else:
                horizontal.append((g-c@(c.T@g)).detach())
        if not all(bool(torch.isfinite(g).all()) for g in horizontal):
            raise ValueError('nonfinite radial gradient')
        norm = math.sqrt(math.fsum(float(torch.sum(g*g)) for g in horizontal))
        return float(value.detach()), details, [g.detach() for g in horizontal], norm

    def record(value, accepted, step, step_radius, norm, reason, details):
        row = dict(evaluation=evaluations, step=step, loss=value, accepted=accepted,
            radius=step_radius, gradient_norm=norm, reason=reason,
            response_loss=details['response_loss'],
            occupied_residual=details['occupied_residual'],
            minimum_occupied_capture=details['minimum_occupied_capture'],
            elapsed_seconds=time.perf_counter()-start)
        history.append(row)
        if progress is not None:
            progress(row)

    loss, details, gradient, norm = evaluate(current)
    initial_loss = loss
    if (occupied_capture_floor is not None
            and details['minimum_occupied_capture'] < occupied_capture_floor):
        raise ValueError('initial candidate is below occupied capture floor')
    rank_signature = details['virtual_rank_signature']
    initial_details = details
    record(loss, True, 0, 0., norm, 'initial', details)
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
            trial = _retract(trial, frozen_counts)
            candidate_loss, candidate_details, candidate_gradient, candidate_norm = evaluate(trial)
            same_rank = candidate_details['virtual_rank_signature'] == rank_signature
            capture_ok = (occupied_capture_floor is None
                          or candidate_details['minimum_occupied_capture'] >= occupied_capture_floor)
            accepted = (same_rank and capture_ok
                        and candidate_loss <= loss-1e-4*step_radius*norm)
            reason = ('accepted' if accepted else
                      ('rank_changed' if not same_rank else
                       ('occupied_capture_below_floor' if not capture_ok else
                        'insufficient_decrease')))
            record(candidate_loss, accepted, accepted_steps+1, step_radius,
                   candidate_norm, reason, candidate_details)
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
        initial_response_loss=initial_details['response_loss'],
        final_response_loss=details['response_loss'],
        initial_occupied_residual=initial_details['occupied_residual'],
        final_occupied_residual=details['occupied_residual'],
        initial_minimum_occupied_capture=initial_details['minimum_occupied_capture'],
        final_minimum_occupied_capture=details['minimum_occupied_capture'],
        occupied_weight=occupied_weight, occupied_capture_floor=occupied_capture_floor,
        accepted_steps=accepted_steps, evaluations=evaluations, stopping_reason=stopping,
        free_radial_count=sum(c.shape[1]-frozen_counts[element][l]
                              for element, channels in current.items()
                              for l, c in enumerate(channels)),
        frozen_radial_count=sum(frozen_counts[element][l]
                               for element, channels in current.items()
                               for l, c in enumerate(channels)),
        old_radial_prefix_frozen=any(any(count > 0 for count in frozen_counts[element])
                                     for element in current), pbe_iteration_constraint=False,
        gradient_norm=norm, initial_radius=radius, virtual_rank_signature=rank_signature,
        final_dimensions=details, elapsed_seconds=time.perf_counter()-start, history=history)
