"""Differentiable body RPA objective on a frozen, Coulomb-whitened Pi space.

Pi already includes k weights and occupations, but not q weights. Frequencies
and quadrature weights are in Hartree. No head/wing, SCF, LRI or q extrapolation
is performed here; production acceptance remains a separate calculation.
"""

from dataclasses import dataclass
import math

import torch

from periodic_galerkin_data import PeriodicGalerkinDataset


@dataclass(frozen=True)
class PeriodicRpaQRecord:
    selected_iq: int
    q_weight: float
    frequency_ha: torch.Tensor
    frequency_weights_ha: torch.Tensor
    candidate_trace_pi: torch.Tensor
    candidate_logdet: torch.Tensor
    candidate_raw: torch.Tensor
    reference_raw: torch.Tensor
    candidate_contributions_ha: torch.Tensor
    reference_contributions_ha: torch.Tensor


@dataclass(frozen=True)
class PeriodicRpaObjective:
    loss: torch.Tensor
    pi_relative_squared_error: torch.Tensor
    trace_log_relative_squared_error: torch.Tensor
    energy_relative_squared_error: torch.Tensor
    candidate_energy_ha: torch.Tensor
    reference_energy_ha: torch.Tensor
    q_weight_coverage: float
    complete_q_weight: bool
    q_records: tuple


def rpa_trace_log(pi):
    """Return Tr[log(I-Pi)+Pi], Tr(Pi), logdet(I-Pi), per frequency.

    Require the imaginary-frequency, Hermitian negative-semidefinite response
    convention. Roundoff-sized positive eigenvalues are tolerated, not clipped.
    """
    if (
        not isinstance(pi, torch.Tensor)
        or pi.dtype not in (torch.float64, torch.complex128)
        or pi.device.type != "cpu"
        or pi.ndim != 3
        or pi.shape[0] == 0
        or pi.shape[1] == 0
        or pi.shape[1] != pi.shape[2]
        or not bool(torch.isfinite(pi).all())
    ):
        raise ValueError("Pi must be a finite CPU double frequency-by-square tensor")
    adjoint = pi.transpose(-2, -1).conj()
    scale = max(1.0, float(torch.abs(pi.detach()).max()))
    if float(torch.abs(pi.detach() - adjoint.detach()).max()) > 1e-10 * scale:
        raise ValueError("Pi must be Hermitian")
    eigenvalues = torch.linalg.eigvalsh((pi + adjoint) * 0.5)
    if bool(torch.any(eigenvalues >= 1.0)):
        raise ValueError("trace-log argument is not positive")
    if float(eigenvalues.detach().max()) > 1e-10 * scale:
        raise ValueError("imaginary-frequency Pi has a positive eigenmode")
    log_terms = torch.log1p(-eigenvalues)
    # Avoid losing the O(Pi^2) high-frequency tail in log1p(-x) + x.
    small_mode = eigenvalues.abs() < 1e-4
    small = torch.where(small_mode, eigenvalues, torch.zeros_like(eigenvalues))
    series = -small.square() * (
        0.5 + small * (1.0 / 3 + small * (0.25 + small * (0.2 + small / 6)))
    )
    raw = torch.where(small_mode, series, log_terms + eigenvalues)
    return raw.sum(dim=-1), eigenvalues.sum(dim=-1), log_terms.sum(dim=-1)


def _validate_inputs(datasets, responses):
    if not isinstance(datasets, tuple) or not datasets:
        raise ValueError("datasets must be a nonempty tuple")
    if not isinstance(responses, tuple) or len(responses) != len(datasets):
        raise ValueError("responses must match datasets")
    if any(not isinstance(d, PeriodicGalerkinDataset) for d in datasets):
        raise ValueError("invalid periodic dataset")
    first = datasets[0]
    shared = (
        "abacus_commit",
        "executable_sha256",
        "orbital_sha256",
        "pseudopotential_sha256",
        "auxiliary_basis_sha256",
        "primitive_blocks_sha256",
        "q_count",
    )
    seen = set()
    for dataset, response in zip(datasets, responses):
        if dataset.selected_iq in seen:
            raise ValueError("duplicate q representative")
        seen.add(dataset.selected_iq)
        if any(getattr(dataset, key) != getattr(first, key) for key in shared):
            raise ValueError("q datasets have mismatched frozen protocol")
        if (
            isinstance(dataset.q_weight, bool)
            or not isinstance(dataset.q_weight, (int, float))
            or not math.isfinite(dataset.q_weight)
            or not 0.0 < dataset.q_weight <= 1.0
        ):
            raise ValueError("q weight must be finite in (0, 1]")
        for field in ("frequency_ha", "frequency_weights_ha"):
            value = getattr(dataset, field)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or value.ndim != 1
                or value.numel() == 0
                or not bool(torch.isfinite(value).all())
                or not bool((value > 0).all())
                or not torch.equal(value, getattr(first, field))
            ):
                raise ValueError(
                    "q datasets require identical finite positive frequency grids"
                )
        count = dataset.frequency_ha.numel()
        if dataset.frequency_weights_ha.numel() != count:
            raise ValueError("frequency nodes and weights do not match")
        if not bool((dataset.frequency_ha[1:] > dataset.frequency_ha[:-1]).all()):
            raise ValueError("frequency nodes must be strictly increasing")
        shape = (
            count,
            dataset.whitened_auxiliary_rank,
            dataset.whitened_auxiliary_rank,
        )
        if (
            not isinstance(response, torch.Tensor)
            or response.shape != shape
            or dataset.reference_response.shape != shape
        ):
            raise ValueError(
                "candidate and reference Pi must share the declared auxiliary space"
            )
    coverage = math.fsum(d.q_weight for d in datasets)
    if coverage > 1.0 + 1e-12:
        raise ValueError("q weights exceed one")
    return coverage


def periodic_rpa_objective(
    datasets, responses, *, pi_weight=1.0, trace_log_weight=1.0, energy_weight=1.0
):
    """Compare frozen-reference Pi, local trace-log and integrated body Ec.

    Each error is normalized by its corresponding reference squared norm.
    Sum physical q weights without renormalizing incomplete star collections.
    Equal scalar weights are API defaults, not calibrated production weights.
    """
    for value in (pi_weight, trace_log_weight, energy_weight):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError("objective weights must be finite and nonnegative")
    if pi_weight <= 0 or trace_log_weight <= 0:
        raise ValueError("Pi and local trace-log guards must both remain active")
    coverage = _validate_inputs(datasets, responses)
    pi_numerators, pi_denominators = [], []
    raw_numerators, raw_denominators = [], []
    records = []
    for dataset, response in zip(datasets, responses):
        reference = dataset.reference_response.detach()
        reference_raw, _, _ = rpa_trace_log(reference)
        raw, trace, logdet = rpa_trace_log(response)
        weights = dataset.q_weight * dataset.frequency_weights_ha.detach()
        pi_numerators.append(
            torch.sum(weights[:, None, None] * (response - reference).abs().square())
        )
        pi_denominators.append(
            torch.sum(weights[:, None, None] * reference.abs().square())
        )
        raw_numerators.append(torch.dot(weights, (raw - reference_raw).square()))
        raw_denominators.append(torch.dot(weights, reference_raw.square()))
        records.append(
            PeriodicRpaQRecord(
                selected_iq=dataset.selected_iq,
                q_weight=dataset.q_weight,
                frequency_ha=dataset.frequency_ha.detach().clone(),
                frequency_weights_ha=dataset.frequency_weights_ha.detach().clone(),
                candidate_trace_pi=trace,
                candidate_logdet=logdet,
                candidate_raw=raw,
                reference_raw=reference_raw,
                candidate_contributions_ha=weights * raw / (2 * math.pi),
                reference_contributions_ha=weights * reference_raw / (2 * math.pi),
            )
        )
    candidate_energy = torch.stack(
        tuple(r.candidate_contributions_ha.sum() for r in records)
    ).sum()
    reference_energy = torch.stack(
        tuple(r.reference_contributions_ha.sum() for r in records)
    ).sum()
    pi_denominator = torch.stack(pi_denominators).sum()
    raw_denominator = torch.stack(raw_denominators).sum()
    denominators = (pi_denominator, raw_denominator, reference_energy.square())
    if any(
        not bool(torch.isfinite(value)) or float(value) <= 0.0 for value in denominators
    ):
        raise ValueError("reference RPA objective has zero or nonfinite norm")
    pi_loss = torch.stack(pi_numerators).sum() / pi_denominator
    raw_loss = torch.stack(raw_numerators).sum() / raw_denominator
    energy_loss = ((candidate_energy - reference_energy) / reference_energy).square()
    loss = (
        pi_weight * pi_loss + trace_log_weight * raw_loss + energy_weight * energy_loss
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("RPA objective is not finite")
    return PeriodicRpaObjective(
        loss=loss,
        pi_relative_squared_error=pi_loss,
        trace_log_relative_squared_error=raw_loss,
        energy_relative_squared_error=energy_loss,
        candidate_energy_ha=candidate_energy,
        reference_energy_ha=reference_energy,
        q_weight_coverage=coverage,
        complete_q_weight=abs(coverage - 1.0) <= 1e-12,
        q_records=tuple(records),
    )
