from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers

import torch

from sternheimer_source_pair import SternheimerResponseSourcePair
from sternheimer_spillage import assemble_orbital_coefficients


@dataclass(frozen=True)
class ProjectedPiResult:
    loss: torch.Tensor
    frequency_ha: torch.Tensor
    frequency_weight: torch.Tensor
    frequency_loss: torch.Tensor
    candidate_a: torch.Tensor
    reference_a: torch.Tensor
    candidate_pi: torch.Tensor
    reference_pi: torch.Tensor
    reference_rank: int
    max_candidate_condition: float
    base_loss: torch.Tensor | None = None
    sensitivity_loss: torch.Tensor | None = None
    frequency_base_loss: torch.Tensor | None = None
    frequency_sensitivity_loss: torch.Tensor | None = None
    trace_log_difference: torch.Tensor | None = None
    minimum_reference_dielectric_eigenvalue: torch.Tensor | None = None
    minimum_candidate_dielectric_eigenvalue: torch.Tensor | None = None
    sensitivity_alpha: float | None = None


@dataclass(frozen=True)
class ProjectedPiFamilyResult:
    loss: torch.Tensor
    results: dict
    max_candidate_condition: float


def evaluate_rpa_sensitivity(
    reference_pi,
    candidate_pi,
    frequency_weight,
    relative_tolerance,
):
    """Return positive sensitivity losses and trace-log diagnostics."""
    try:
        relative_tolerance = float(relative_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "relative_tolerance must be finite and between zero and one"
        ) from exc
    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
        or relative_tolerance >= 1.0
    ):
        raise ValueError(
            "relative_tolerance must be finite and between zero and one"
        )
    if (
        reference_pi.ndim != 3
        or candidate_pi.shape != reference_pi.shape
        or reference_pi.shape[-2] != reference_pi.shape[-1]
    ):
        raise ValueError(
            "reference and candidate Pi must be matching frequency matrices"
        )
    if frequency_weight.ndim != 1 or frequency_weight.shape[0] != reference_pi.shape[0]:
        raise ValueError("frequency_weight must match the Pi frequency dimension")
    if not bool(torch.all(torch.isfinite(frequency_weight))) or not bool(
        torch.all(frequency_weight > 0.0)
    ):
        raise ValueError("frequency_weight must be positive and finite")

    hermitian_inputs = []
    for name, values in (
        ("reference", reference_pi),
        ("candidate", candidate_pi),
    ):
        if not bool(torch.all(torch.isfinite(values))):
            raise RuntimeError(f"{name} Pi must be finite")
        matrix_norm = torch.linalg.matrix_norm(values, dim=(-2, -1))
        hermitian_error = torch.linalg.matrix_norm(
            values - values.mH,
            dim=(-2, -1),
        )
        hermitian_threshold = 10.0 * relative_tolerance * torch.maximum(
            torch.ones_like(matrix_norm),
            matrix_norm,
        )
        if bool(torch.any(hermitian_error > hermitian_threshold)):
            raise RuntimeError(f"{name} Pi is materially non-Hermitian")
        hermitian_inputs.append((values + values.mH) / 2.0)

    reference_pi, candidate_pi = hermitian_inputs
    sensitivity_error = []
    sensitivity_reference_norm = []
    trace_log_difference = []
    minimum_reference_dielectric_eigenvalue = []
    minimum_candidate_dielectric_eigenvalue = []
    for reference, candidate in zip(reference_pi, candidate_pi):
        reference_eigenvalue, reference_eigenvector = torch.linalg.eigh(reference)
        candidate_eigenvalue = torch.linalg.eigvalsh(candidate)
        reference_dielectric = 1.0 - reference_eigenvalue
        candidate_dielectric = 1.0 - candidate_eigenvalue
        minimum_reference = torch.min(reference_dielectric)
        minimum_candidate = torch.min(candidate_dielectric)
        if float(minimum_reference) <= relative_tolerance:
            raise RuntimeError("reference I-Pi is not positive")
        if float(minimum_candidate) <= relative_tolerance:
            raise RuntimeError("candidate I-Pi is not positive")

        g = torch.abs(1.0 - 1.0 / reference_dielectric)
        maximum_g = torch.max(g)
        if float(maximum_g) <= relative_tolerance:
            raise RuntimeError("RPA sensitivity is numerically zero")
        weight_sqrt = (
            reference_eigenvector
            @ torch.diag(torch.sqrt(g / maximum_g)).to(
                reference_eigenvector.dtype
            )
            @ reference_eigenvector.mH
        )
        weighted_error = weight_sqrt @ (candidate - reference) @ weight_sqrt
        weighted_reference = weight_sqrt @ reference @ weight_sqrt
        sensitivity_error.append(
            torch.sum(torch.abs(weighted_error) ** 2).real
        )
        sensitivity_reference_norm.append(
            torch.sum(torch.abs(weighted_reference) ** 2).real
        )
        trace_log_difference.append(
            torch.sum(torch.log(candidate_dielectric) + candidate_eigenvalue)
            - torch.sum(torch.log(reference_dielectric) + reference_eigenvalue)
        )
        minimum_reference_dielectric_eigenvalue.append(minimum_reference)
        minimum_candidate_dielectric_eigenvalue.append(minimum_candidate)

    sensitivity_error = torch.stack(sensitivity_error)
    sensitivity_reference_norm = torch.stack(sensitivity_reference_norm)
    if not bool(torch.all(torch.isfinite(sensitivity_error))):
        raise RuntimeError("RPA sensitivity squared error must be finite")
    if not bool(
        torch.all(torch.isfinite(sensitivity_reference_norm))
    ) or not bool(torch.all(sensitivity_reference_norm > 0.0)):
        raise RuntimeError(
            "RPA sensitivity reference norm must be positive and finite"
        )
    frequency_sensitivity_loss = (
        sensitivity_error / sensitivity_reference_norm
    )
    sensitivity_loss = torch.sum(frequency_weight * sensitivity_error) / torch.sum(
        frequency_weight * sensitivity_reference_norm
    )
    trace_log_difference = torch.stack(trace_log_difference)
    if not bool(torch.isfinite(sensitivity_loss)) or not bool(
        torch.all(torch.isfinite(frequency_sensitivity_loss))
    ):
        raise RuntimeError("RPA sensitivity loss must be finite")
    if not bool(torch.all(torch.isfinite(trace_log_difference))):
        raise RuntimeError("RPA trace-log difference must be finite")

    return {
        "sensitivity_loss": sensitivity_loss,
        "frequency_sensitivity_loss": frequency_sensitivity_loss,
        "trace_log_difference": trace_log_difference,
        "minimum_reference_dielectric_eigenvalue": torch.stack(
            minimum_reference_dielectric_eigenvalue
        ),
        "minimum_candidate_dielectric_eigenvalue": torch.stack(
            minimum_candidate_dielectric_eigenvalue
        ),
    }


class ProjectedPiEvaluator:
    def __init__(
        self,
        pair,
        relative_rank_tolerance=1.0e-12,
        condition_limit=1.0e12,
        sensitivity_alpha=None,
    ):
        if not isinstance(pair, SternheimerResponseSourcePair):
            raise ValueError("pair must be SternheimerResponseSourcePair")
        self.pair = pair
        self.relative_rank_tolerance = _normalize_rank_tolerance(
            relative_rank_tolerance
        )
        self.condition_limit = _normalize_condition_limit(condition_limit)
        self.sensitivity_alpha = _normalize_sensitivity_alpha(sensitivity_alpha)

        (
            self._occupied_states,
            self._occupation,
            self._d,
        ) = _organize_source(pair)
        (
            self._frequency_ha,
            self._frequency_weight,
            self._q,
        ) = _organize_response(pair, self._occupied_states)
        (
            self._s_plus,
            self._reference_rank,
        ) = _primitive_pseudoinverse(
            pair.source.overlap,
            self.relative_rank_tolerance,
        )
        (
            self._reference_a,
            self._reference_pi,
            self._reference_norm,
        ) = self._build_reference()

    def _build_reference(self):
        d_s_plus = torch.matmul(self._d, self._s_plus)
        reference_a = []
        for frequency_index in range(self._q.shape[0]):
            value = torch.zeros(
                (self._d.shape[1], self._q.shape[2]),
                dtype=torch.complex128,
                device="cpu",
            )
            for occupied_index in range(self._d.shape[0]):
                value = value + self._occupation[occupied_index] * (
                    d_s_plus[occupied_index]
                    @ self._q[frequency_index, occupied_index].mH
                )
            reference_a.append(value)
        reference_a = torch.stack(reference_a)
        reference_pi = reference_a + reference_a.mH
        reference_norm = torch.sum(
            torch.abs(reference_pi) ** 2,
            dim=(1, 2),
        ).real
        if not bool(torch.all(torch.isfinite(reference_norm))):
            raise RuntimeError("primitive-reference norm must be finite")
        if not bool(torch.all(reference_norm > 0.0)):
            raise RuntimeError("primitive-reference norm must be positive")
        return reference_a, reference_pi, reference_norm

    def evaluate(self, coefficients):
        coefficient_matrix, _ = assemble_orbital_coefficients(
            self.pair.response,
            coefficients,
        )
        if coefficient_matrix.shape[1] == 0:
            raise RuntimeError("candidate orbital basis must be nonempty")
        candidate_overlap = (
            coefficient_matrix.mH
            @ self.pair.response.overlap
            @ coefficient_matrix
        )
        factor, condition = _factor_candidate_overlap(
            candidate_overlap,
            self.condition_limit,
        )

        d_projected = torch.matmul(self._d, coefficient_matrix)
        q_projected = torch.matmul(self._q, coefficient_matrix)
        candidate_a = []
        for frequency_index in range(q_projected.shape[0]):
            value = torch.zeros(
                (d_projected.shape[1], q_projected.shape[2]),
                dtype=torch.complex128,
                device="cpu",
            )
            for occupied_index in range(d_projected.shape[0]):
                solved = torch.cholesky_solve(
                    q_projected[frequency_index, occupied_index].mH,
                    factor,
                )
                value = value + self._occupation[occupied_index] * (
                    d_projected[occupied_index] @ solved
                )
            candidate_a.append(value)
        candidate_a = torch.stack(candidate_a)
        candidate_pi = candidate_a + candidate_a.mH
        error_norm = torch.sum(
            torch.abs(candidate_pi - self._reference_pi) ** 2,
            dim=(1, 2),
        ).real
        if not bool(torch.all(torch.isfinite(error_norm))):
            raise RuntimeError("projected-Pi squared error must be finite")
        frequency_loss = error_norm / self._reference_norm
        numerator = torch.sum(self._frequency_weight * error_norm)
        denominator = torch.sum(
            self._frequency_weight * self._reference_norm
        )
        if not bool(torch.isfinite(denominator)) or not bool(denominator > 0.0):
            raise RuntimeError(
                "weighted primitive-reference norm must be positive and finite"
            )
        loss = numerator / denominator
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("projected-Pi loss must be finite")

        if self.sensitivity_alpha is None:
            return ProjectedPiResult(
                loss=loss,
                frequency_ha=self._frequency_ha,
                frequency_weight=self._frequency_weight,
                frequency_loss=frequency_loss,
                candidate_a=candidate_a,
                reference_a=self._reference_a,
                candidate_pi=candidate_pi,
                reference_pi=self._reference_pi,
                reference_rank=self._reference_rank,
                max_candidate_condition=condition,
            )

        base_loss = loss
        frequency_base_loss = frequency_loss
        sensitivity = evaluate_rpa_sensitivity(
            self._reference_pi,
            candidate_pi,
            self._frequency_weight,
            self.relative_rank_tolerance,
        )
        sensitivity_loss = sensitivity["sensitivity_loss"]
        frequency_sensitivity_loss = sensitivity[
            "frequency_sensitivity_loss"
        ]
        alpha = self.sensitivity_alpha
        loss = alpha * base_loss + (1.0 - alpha) * sensitivity_loss
        frequency_loss = (
            alpha * frequency_base_loss
            + (1.0 - alpha) * frequency_sensitivity_loss
        )

        return ProjectedPiResult(
            loss=loss,
            frequency_ha=self._frequency_ha,
            frequency_weight=self._frequency_weight,
            frequency_loss=frequency_loss,
            candidate_a=candidate_a,
            reference_a=self._reference_a,
            candidate_pi=candidate_pi,
            reference_pi=self._reference_pi,
            reference_rank=self._reference_rank,
            max_candidate_condition=condition,
            base_loss=base_loss,
            sensitivity_loss=sensitivity_loss,
            frequency_base_loss=frequency_base_loss,
            frequency_sensitivity_loss=frequency_sensitivity_loss,
            trace_log_difference=sensitivity["trace_log_difference"],
            minimum_reference_dielectric_eigenvalue=sensitivity[
                "minimum_reference_dielectric_eigenvalue"
            ],
            minimum_candidate_dielectric_eigenvalue=sensitivity[
                "minimum_candidate_dielectric_eigenvalue"
            ],
            sensitivity_alpha=alpha,
        )


class NormalizedPhysicalFamilyProjectedPi:
    def __init__(
        self,
        named_pairs,
        relative_rank_tolerance=1.0e-12,
        condition_limit=1.0e12,
        sensitivity_alpha=None,
    ):
        if isinstance(named_pairs, Mapping):
            named_pairs = tuple(named_pairs.items())
        else:
            named_pairs = tuple(named_pairs)
        names = []
        evaluators = []
        for item in named_pairs:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    "physical family names and pairs must be (name, pair) values"
                )
            name, pair = item
            if not isinstance(name, str) or not name.strip():
                raise ValueError("physical family names must be nonempty")
            names.append(name)
            evaluator_options = {
                "relative_rank_tolerance": relative_rank_tolerance,
                "condition_limit": condition_limit,
            }
            if sensitivity_alpha is not None:
                evaluator_options["sensitivity_alpha"] = sensitivity_alpha
            evaluators.append(ProjectedPiEvaluator(pair, **evaluator_options))
        if not names or len(names) != len(set(names)):
            raise ValueError("physical family names must be nonempty and unique")
        self._names = tuple(names)
        self._evaluators = tuple(evaluators)

    def evaluate(self, coefficients):
        results = {}
        total = None
        max_condition = 0.0
        for name, evaluator in zip(self._names, self._evaluators):
            result = evaluator.evaluate(coefficients)
            results[name] = result
            total = result.loss if total is None else total + result.loss
            max_condition = max(
                max_condition,
                result.max_candidate_condition,
            )
        if total is None or not bool(torch.isfinite(total)):
            raise RuntimeError("physical-family projected-Pi loss must be finite")
        return ProjectedPiFamilyResult(
            loss=total,
            results=results,
            max_candidate_condition=max_condition,
        )


def _organize_source(pair):
    source = pair.source
    occupied_states = tuple(sorted(set(source.occupied_state.tolist())))
    channels = tuple(sorted(set(source.auxiliary_channel.tolist())))
    expected_keys = {
        (occupied_state, channel)
        for occupied_state in occupied_states
        for channel in channels
    }
    source_keys = set(pair.source_row_for_response_key)
    if source_keys != expected_keys:
        raise ValueError(
            "source rows do not form a complete occupied/channel rectangle"
        )

    occupied_index = {
        occupied_state: index
        for index, occupied_state in enumerate(occupied_states)
    }
    channel_index = {channel: index for index, channel in enumerate(channels)}
    d = torch.empty(
        (len(occupied_states), len(channels), source.d.shape[1]),
        dtype=torch.complex128,
        device="cpu",
    )
    occupation = torch.empty(
        len(occupied_states), dtype=torch.float64, device="cpu"
    )
    for occupied_state in occupied_states:
        values = []
        for channel in channels:
            row = pair.source_row_for_response_key[(occupied_state, channel)]
            d[occupied_index[occupied_state], channel_index[channel]] = source.d[row]
            values.append(source.occupation[row])
        stacked = torch.stack(values)
        if not bool(torch.all(stacked == stacked[0])):
            raise ValueError(
                f"source occupations differ across channels for occupied state "
                f"{occupied_state}"
            )
        occupation[occupied_index[occupied_state]] = stacked[0]
    return occupied_states, occupation, d


def _organize_response(pair, occupied_states):
    response = pair.response
    frequencies = torch.unique(response.frequency_ha, sorted=True)
    channels = tuple(sorted(set(response.auxiliary_channel.tolist())))
    frequency_values = tuple(float(value) for value in frequencies)
    expected_keys = {
        (frequency, occupied_state, channel)
        for frequency in frequency_values
        for occupied_state in occupied_states
        for channel in channels
    }
    rows = {}
    for row in range(response.q.shape[0]):
        key = (
            float(response.frequency_ha[row]),
            int(response.occupied_state[row]),
            int(response.auxiliary_channel[row]),
        )
        if key in rows:
            raise ValueError(f"duplicate response rectangle row: {key}")
        rows[key] = row
    if set(rows) != expected_keys:
        missing = sorted(expected_keys - set(rows))
        extra = sorted(set(rows) - expected_keys)
        raise ValueError(
            "response rows do not form a complete response rectangle: "
            f"missing={missing}, extra={extra}"
        )

    occupied_index = {
        occupied_state: index
        for index, occupied_state in enumerate(occupied_states)
    }
    channel_index = {channel: index for index, channel in enumerate(channels)}
    q = torch.empty(
        (
            len(frequency_values),
            len(occupied_states),
            len(channels),
            response.q.shape[1],
        ),
        dtype=torch.complex128,
        device="cpu",
    )
    frequency_weight = torch.empty(
        len(frequency_values), dtype=torch.float64, device="cpu"
    )
    for frequency_index, frequency in enumerate(frequency_values):
        weights = []
        for occupied_state in occupied_states:
            for channel in channels:
                row = rows[(frequency, occupied_state, channel)]
                q[
                    frequency_index,
                    occupied_index[occupied_state],
                    channel_index[channel],
                ] = response.q[row]
                weights.append(response.frequency_weight[row])
        weights = torch.stack(weights)
        if not bool(torch.all(weights == weights[0])):
            raise ValueError(
                f"frequency weights differ within frequency {frequency:.17g}"
            )
        if not bool(weights[0] > 0.0):
            raise ValueError("frequency weights must be positive")
        frequency_weight[frequency_index] = weights[0]
    return frequencies, frequency_weight, q


def _primitive_pseudoinverse(overlap, relative_rank_tolerance):
    hermitian = (overlap + overlap.mH) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
    largest = float(torch.max(eigenvalues))
    if not math.isfinite(largest) or largest <= 0.0:
        raise RuntimeError("primitive overlap has no positive eigenvalue")
    cutoff = relative_rank_tolerance * largest
    if float(torch.min(eigenvalues)) < -cutoff:
        raise RuntimeError("primitive overlap is materially indefinite")
    keep = eigenvalues > cutoff
    rank = int(torch.count_nonzero(keep))
    if rank == 0:
        raise RuntimeError("primitive overlap has numerical rank zero")
    retained_vectors = eigenvectors[:, keep]
    inverse = (
        retained_vectors
        @ torch.diag(1.0 / eigenvalues[keep]).to(torch.complex128)
        @ retained_vectors.mH
    )
    return inverse, rank


def _factor_candidate_overlap(overlap, condition_limit):
    hermitian = (overlap + overlap.mH) / 2.0
    factor, info = torch.linalg.cholesky_ex(hermitian)
    if int(info.item()) != 0:
        raise RuntimeError("candidate overlap is not positive definite")
    condition = float(torch.linalg.cond(hermitian))
    if not math.isfinite(condition):
        raise RuntimeError("candidate overlap condition number is not finite")
    if condition > condition_limit:
        raise RuntimeError(
            f"candidate overlap condition number {condition:.6g} exceeds "
            f"condition_limit {condition_limit:.6g}"
        )
    return factor, condition


def _normalize_rank_tolerance(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "relative_rank_tolerance must be finite and between zero and one"
        ) from exc
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(
            "relative_rank_tolerance must be finite and between zero and one"
        )
    return value


def _normalize_condition_limit(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("condition_limit must be finite and at least 1") from exc
    if not math.isfinite(value) or value < 1.0:
        raise ValueError("condition_limit must be finite and at least 1")
    return value


def _normalize_sensitivity_alpha(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("sensitivity_alpha must be a finite real number")
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            "sensitivity_alpha must be finite and between zero and one"
        )
    return float(value)
