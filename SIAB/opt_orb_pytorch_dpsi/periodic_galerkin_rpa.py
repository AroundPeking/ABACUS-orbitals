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


_REFERENCE_INPUT_FIELDS = (
    "reference_response", "frequency_ha", "frequency_weights_ha"
)


@dataclass(frozen=True, eq=False)
class _PeriodicRpaReferenceQ:
    dataset: PeriodicGalerkinDataset
    inputs: tuple
    versions: tuple
    raw: tuple
    weights: tuple
    contributions_ha: tuple


@dataclass(frozen=True, eq=False)
class _PeriodicRpaReferenceCache:
    q_references: tuple
    pi_denominator: float
    raw_denominator: float
    reference_energy_ha: float


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


def _reference_input_version(value):
    try:
        return value._version
    except RuntimeError as error:
        raise ValueError("reference cache requires version-tracked input tensors") from error


def prepare_periodic_rpa_reference(datasets):
    """Prepare an opaque, immutable reference cache for these dataset objects.

    Reuse with ``periodic_rpa_objective(..., reference_cache=cache)``. A new
    tuple containing the same datasets in the same order is allowed. Replaced
    datasets/inputs and version-tracked in-place writes (including through
    detached aliases) invalidate the cache. Do not write through ``.data``,
    NumPy or raw storage, which bypass PyTorch's version counter, or mutate
    inputs concurrently with preparation/evaluation.

    Only the small derived vectors/scalars are copied into immutable Python
    values. Strong references retain input identity without copying full Pi;
    returned objective tensors never alias the cached derived values.
    """
    if (
        not isinstance(datasets, tuple)
        or not datasets
        or any(not isinstance(d, PeriodicGalerkinDataset) for d in datasets)
    ):
        raise ValueError("datasets must be a nonempty tuple of periodic datasets")
    _validate_inputs(datasets, tuple(d.reference_response for d in datasets))
    prepared = []
    pi_denominators, raw_denominators, contributions = [], [], []
    for dataset in datasets:
        inputs = tuple(getattr(dataset, name) for name in _REFERENCE_INPUT_FIELDS)
        versions = tuple(_reference_input_version(value) for value in inputs)
        reference = dataset.reference_response.detach()
        raw, _, _ = rpa_trace_log(reference)
        weights = dataset.q_weight * dataset.frequency_weights_ha.detach()
        pi_denominators.append(
            torch.sum(weights[:, None, None] * reference.abs().square())
        )
        raw_denominators.append(torch.dot(weights, raw.square()))
        contribution = weights * raw / (2 * math.pi)
        contributions.append(contribution)
        prepared.append(_PeriodicRpaReferenceQ(
            dataset=dataset,
            inputs=inputs,
            versions=versions,
            raw=tuple(raw.tolist()),
            weights=tuple(weights.tolist()),
            contributions_ha=tuple(contribution.tolist()),
        ))
    pi_denominator = torch.stack(pi_denominators).sum()
    raw_denominator = torch.stack(raw_denominators).sum()
    reference_energy = torch.stack(tuple(value.sum() for value in contributions)).sum()
    denominators = (pi_denominator, raw_denominator, reference_energy.square())
    if any(
        not bool(torch.isfinite(value)) or float(value) <= 0.0 for value in denominators
    ):
        raise ValueError("reference RPA objective has zero or nonfinite norm")
    cache = _PeriodicRpaReferenceCache(
        q_references=tuple(prepared),
        pi_denominator=float(pi_denominator),
        raw_denominator=float(raw_denominator),
        reference_energy_ha=float(reference_energy),
    )
    _validate_reference_cache(datasets, cache)
    return cache


def _validate_reference_cache(datasets, cache):
    if not isinstance(cache, _PeriodicRpaReferenceCache):
        raise ValueError("invalid RPA reference cache")
    if not isinstance(datasets, tuple) or len(datasets) != len(cache.q_references):
        raise ValueError("RPA reference cache datasets mismatch")
    for dataset, prepared in zip(datasets, cache.q_references):
        if dataset is not prepared.dataset:
            raise ValueError("RPA reference cache dataset identity/order mismatch")
        for name, original, version in zip(
            _REFERENCE_INPUT_FIELDS, prepared.inputs, prepared.versions
        ):
            if getattr(dataset, name) is not original:
                raise ValueError("RPA reference cache input identity mismatch: " + name)
            if _reference_input_version(original) != version:
                raise ValueError("RPA reference cache input was mutated: " + name)


def periodic_rpa_objective(
    datasets, responses, *, pi_weight=1.0, trace_log_weight=1.0, energy_weight=1.0,
    reference_cache=None
):
    """Compare frozen-reference Pi, local trace-log and integrated body Ec.

    Each error is normalized by its corresponding reference squared norm.
    Sum physical q weights without renormalizing incomplete star collections.
    Equal scalar weights are API defaults, not calibrated production weights.
    ``reference_cache`` is opt-in; prepare it with prepare_periodic_rpa_reference.
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
    if reference_cache is not None:
        _validate_reference_cache(datasets, reference_cache)
    coverage = _validate_inputs(datasets, responses)
    pi_numerators, pi_denominators = [], []
    raw_numerators, raw_denominators = [], []
    records = []
    for iq, (dataset, response) in enumerate(zip(datasets, responses)):
        reference = dataset.reference_response.detach()
        if reference_cache is None:
            reference_raw, _, _ = rpa_trace_log(reference)
        else:
            prepared = reference_cache.q_references[iq]
            reference_raw = torch.tensor(prepared.raw, dtype=torch.float64)
        raw, trace, logdet = rpa_trace_log(response)
        if reference_cache is None:
            weights = dataset.q_weight * dataset.frequency_weights_ha.detach()
        else:
            weights = torch.tensor(prepared.weights, dtype=torch.float64)
        pi_numerators.append(
            torch.sum(weights[:, None, None] * (response - reference).abs().square())
        )
        if reference_cache is None:
            pi_denominators.append(
                torch.sum(weights[:, None, None] * reference.abs().square())
            )
        raw_numerators.append(torch.dot(weights, (raw - reference_raw).square()))
        if reference_cache is None:
            raw_denominators.append(torch.dot(weights, reference_raw.square()))
            reference_contributions = weights * reference_raw / (2 * math.pi)
        else:
            reference_contributions = torch.tensor(
                prepared.contributions_ha, dtype=torch.float64
            )
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
                reference_contributions_ha=reference_contributions,
            )
        )
    candidate_energy = torch.stack(
        tuple(r.candidate_contributions_ha.sum() for r in records)
    ).sum()
    if reference_cache is None:
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
    else:
        reference_energy = torch.tensor(reference_cache.reference_energy_ha, dtype=torch.float64)
        pi_denominator = torch.tensor(reference_cache.pi_denominator, dtype=torch.float64)
        raw_denominator = torch.tensor(reference_cache.raw_denominator, dtype=torch.float64)
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
