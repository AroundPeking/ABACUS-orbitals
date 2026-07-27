from dataclasses import dataclass
import math
from collections.abc import Mapping

import torch

from response_basis import canonicalize_columns
from sternheimer_spillage import (
    RadialResidualSpectrum,
    evaluate_spillage_for_columns,
)


_TARGET_ROLES = frozenset({"physical", "ghost"})


@dataclass(frozen=True)
class CandidateGain:
    l: int
    mode: int
    atom: float
    multicenter: float

    @property
    def cost(self):
        return 2 * self.l + 1

    @property
    def key(self):
        return (self.l, self.mode)


@dataclass(frozen=True)
class CandidateEvaluation:
    gain: CandidateGain
    score: float
    admissible: bool
    rejection_reason: str = None


@dataclass(frozen=True)
class ResponseTargetFamily:
    name: str
    data: tuple
    role: str
    real_atom_index: int = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("target family name must be nonempty")
        data = tuple(self.data)
        if not data:
            raise ValueError("target family data must be nonempty")
        object.__setattr__(self, "data", data)
        if self.role not in _TARGET_ROLES:
            raise ValueError("target family role must be physical or ghost")
        if self.role == "physical":
            if self.real_atom_index is not None:
                raise ValueError(
                    "physical target family must not set real_atom_index"
                )
            return
        if type(self.real_atom_index) is not int or self.real_atom_index < 0:
            raise ValueError(
                "ghost target family requires a nonnegative real_atom_index"
            )
        for value in data:
            if not any(
                block.atom_index == self.real_atom_index
                for block in value.blocks
            ):
                raise ValueError("real_atom_index maps to no primitive blocks")


def _sum_residual(data_items, coefficients, include):
    total = 0.0
    for data in data_items:
        residual = float(
            evaluate_spillage_for_columns(
                data, coefficients, include
            ).weighted_residual.item()
        )
        if not math.isfinite(residual) or residual < 0.0:
            raise RuntimeError(
                "target-family residual must be finite and nonnegative"
            )
        total += residual
    return total


def _positive_dzp_residual(value):
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("fixed-DZP family residual must be positive")
    return value


def normalized_family_loss(family, current, fixed_dzp):
    if not isinstance(family, ResponseTargetFamily):
        raise TypeError("family must be a ResponseTargetFamily")
    if family.role != "physical":
        raise ValueError("normalized physical loss requires a physical family")
    include_all = lambda label: True
    numerator = _sum_residual(family.data, current, include_all)
    denominator = _positive_dzp_residual(
        _sum_residual(family.data, fixed_dzp, include_all)
    )
    return numerator / denominator


def borrowing_gap(family, current, fixed_dzp):
    if not isinstance(family, ResponseTargetFamily):
        raise TypeError("family must be a ResponseTargetFamily")
    if family.role != "ghost":
        raise ValueError("borrowing gap requires a ghost family")

    own = lambda label: label.atom_index == family.real_atom_index
    all_centers = lambda label: True
    denominator = _positive_dzp_residual(
        _sum_residual(family.data, fixed_dzp, own)
    )
    own_residual = _sum_residual(family.data, current, own)
    all_residual = _sum_residual(family.data, current, all_centers)
    difference = own_residual - all_residual
    tolerance = 1.0e-10 * max(
        abs(own_residual), abs(all_residual), denominator, 1.0
    )
    if difference < -tolerance:
        raise RuntimeError(
            "all-center residual exceeds own-center residual; "
            "check fragment/ghost target construction"
        )
    return max(difference, 0.0) / denominator


def score_candidate(value):
    physical = value.atom + value.multicenter
    if physical <= 1.0e-14:
        return float("-inf")
    return physical / value.cost


def select_best_candidate(values):
    admissible = [value for value in values if score_candidate(value) > 0.0]
    if not admissible:
        raise RuntimeError("no admissible positive-score response shell remains")
    return min(
        admissible,
        key=lambda value: (
            -score_candidate(value),
            value.cost,
            value.l,
            value.mode,
        ),
    )


def _clone_coefficients(coefficients):
    if not isinstance(coefficients, Mapping) or not coefficients:
        raise TypeError("coefficients must be a nonempty mapping")
    result = {}
    for element, by_l in coefficients.items():
        if isinstance(by_l, Mapping):
            result[element] = {
                l: value.detach().clone() for l, value in by_l.items()
            }
        else:
            try:
                result[element] = [value.detach().clone() for value in by_l]
            except (AttributeError, TypeError) as exc:
                raise TypeError(
                    f"coefficients[{element!r}] must contain tensor channels"
                ) from exc
    return result


def append_response_shell(coefficients, spectrum, mode=0):
    if not isinstance(spectrum, RadialResidualSpectrum):
        raise TypeError("spectrum must be a RadialResidualSpectrum")
    if type(mode) is not int or mode < 0:
        raise ValueError("mode must be a nonnegative integer")
    if mode >= spectrum.coefficients.shape[1]:
        raise ValueError("mode is outside the residual spectrum")

    result = _clone_coefficients(coefficients)
    try:
        channel = result[spectrum.element][spectrum.l]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"missing coefficient channel {spectrum.element}/{spectrum.l}"
        ) from exc
    if not isinstance(channel, torch.Tensor) or channel.ndim != 2:
        raise ValueError("response coefficient channel must be a rank-2 tensor")
    if channel.dtype != torch.float64 or channel.is_complex():
        raise ValueError("response coefficient channel must be real float64")
    if channel.shape[0] != spectrum.coefficients.shape[0]:
        raise ValueError(
            "candidate radial row count does not match coefficient channel"
        )

    candidate = canonicalize_columns(
        spectrum.coefficients[:, mode : mode + 1]
    )
    result[spectrum.element][spectrum.l] = torch.cat(
        (channel, candidate), dim=1
    )
    return result


def _rejected_candidate(l, mode, reason):
    gain = CandidateGain(l, mode, 0.0, 0.0)
    return CandidateEvaluation(
        gain=gain,
        score=float("-inf"),
        admissible=False,
        rejection_reason=reason,
    )


def evaluate_response_candidates(
    spectra,
    current,
    fixed_dzp,
    atom_family,
    multicenter_family,
):
    spectra = tuple(sorted(spectra, key=lambda value: value.l))
    if not spectra:
        raise ValueError("spectra must be nonempty")
    if any(not isinstance(value, RadialResidualSpectrum) for value in spectra):
        raise TypeError("spectra must contain RadialResidualSpectrum values")
    if len({value.l for value in spectra}) != len(spectra):
        raise ValueError("spectra must contain at most one candidate per l")

    atom_before = normalized_family_loss(atom_family, current, fixed_dzp)
    multicenter_before = normalized_family_loss(
        multicenter_family, current, fixed_dzp
    )
    values = []
    for spectrum in spectra:
        mode = 0
        if (
            spectrum.eigenvalues.numel() == 0
            or not math.isfinite(float(spectrum.eigenvalues[mode].item()))
        ):
            raise ValueError("residual spectrum eigenvalue must be finite")
        if float(spectrum.eigenvalues[mode].item()) <= 0.0:
            values.append(
                _rejected_candidate(
                    spectrum.l,
                    mode,
                    "residual eigenvalue is not positive",
                )
            )
            continue

        candidate = append_response_shell(current, spectrum, mode)
        try:
            gain = CandidateGain(
                l=spectrum.l,
                mode=mode,
                atom=atom_before
                - normalized_family_loss(atom_family, candidate, fixed_dzp),
                multicenter=multicenter_before
                - normalized_family_loss(
                    multicenter_family, candidate, fixed_dzp
                ),
            )
        except RuntimeError as exc:
            values.append(
                _rejected_candidate(
                    spectrum.l,
                    mode,
                    f"candidate evaluation failed: {exc}",
                )
            )
            continue

        score = score_candidate(gain)
        if gain.atom + gain.multicenter <= 1.0e-14:
            reason = "physical gain is not positive"
        elif not math.isfinite(score) or score <= 0.0:
            reason = "score is not positive"
        else:
            reason = None
        values.append(
            CandidateEvaluation(
                gain=gain,
                score=score,
                admissible=reason is None,
                rejection_reason=reason,
            )
        )
    return tuple(values)
