from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import spherical_jn
import torch


@dataclass(frozen=True)
class RadialChannelLocality:
    element: str
    l: int
    variable_columns: int
    tail_fraction: torch.Tensor
    condition: float


@dataclass(frozen=True)
class RadialLocalityResult:
    loss: torch.Tensor
    max_condition: float
    by_channel: dict


def _finite_real(value, name, *, positive=False):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite")
    return value


def _element_radial_value(radial, field, element):
    try:
        value = radial[field]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"radial configuration is missing {field}") from exc
    if isinstance(value, Mapping):
        try:
            value = value[element]
        except KeyError as exc:
            raise ValueError(
                f"radial {field} is missing element {element!r}"
            ) from exc
    return value


def _validate_fixed_specs(fixed_specs, info_element):
    try:
        fixed_specs = tuple(fixed_specs)
    except TypeError as exc:
        raise TypeError("fixed radial specs must be a sequence") from exc
    if not fixed_specs:
        raise ValueError("fixed radial specs must be nonempty")

    fixed = {}
    seen = set()
    for spec in fixed_specs:
        if not isinstance(spec, Mapping) or set(spec) != {
            "element",
            "l",
            "zeta",
        }:
            raise ValueError(
                "fixed radial spec requires element, l, and zeta"
            )
        element = spec["element"]
        l = spec["l"]
        zeta = spec["zeta"]
        if element not in info_element:
            raise ValueError(f"fixed radial spec element {element!r} is missing")
        if type(l) is not int or l < 0 or l >= info_element[element].Nl:
            raise ValueError(f"fixed radial spec l={l!r} is invalid")
        if type(zeta) is not int or zeta <= 0:
            raise ValueError(f"fixed radial spec zeta={zeta!r} is invalid")
        key = (element, l, zeta - 1)
        if key in seen:
            raise ValueError(f"duplicate fixed radial spec {key!r}")
        seen.add(key)
        fixed.setdefault((element, l), []).append(zeta - 1)
    return {
        key: tuple(sorted(indices)) for key, indices in fixed.items()
    }


def _factor_gram(matrix, condition_limit, label):
    matrix = (matrix + matrix.transpose(0, 1)) / 2.0
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info.item()) != 0:
        raise RuntimeError(f"{label} is not positive definite")
    condition = float(torch.linalg.cond(matrix).item())
    if not math.isfinite(condition):
        raise RuntimeError(f"{label} condition number is not finite")
    if condition > condition_limit:
        raise RuntimeError(
            f"{label} condition number {condition:.6g} exceeds "
            f"condition_limit {condition_limit:.6g}"
        )
    return factor, condition


class RadialSubspaceLocality:
    """Measure the outer radial weight of the nonfixed AO subspace."""

    def __init__(
        self,
        info_element,
        radial,
        eigenvalues,
        fixed_specs,
        local_radius,
        condition_limit=1.0e10,
    ):
        if not isinstance(info_element, Mapping) or not info_element:
            raise TypeError("info_element must be a nonempty mapping")
        if not isinstance(radial, Mapping):
            raise TypeError("radial must be a mapping")
        if not isinstance(eigenvalues, Mapping):
            raise TypeError("eigenvalues must be a mapping")
        self._condition_limit = _finite_real(
            condition_limit, "condition_limit", positive=True
        )
        if self._condition_limit < 1.0:
            raise ValueError("condition_limit must be at least 1")
        self._local_radius = _finite_real(
            local_radius, "local_radius", positive=True
        )
        self._fixed = _validate_fixed_specs(fixed_specs, info_element)
        self._metrics = {}

        for element, info in info_element.items():
            if not isinstance(element, str) or not element:
                raise ValueError("info_element keys must be nonempty strings")
            if type(info.Nl) is not int or info.Nl <= 0:
                raise ValueError(f"info_element[{element!r}].Nl must be positive")
            if type(info.Ne) is not int or info.Ne <= 0:
                raise ValueError(f"info_element[{element!r}].Ne must be positive")
            rcut = _finite_real(
                _element_radial_value(radial, "Rcut", element),
                f"radial Rcut[{element!r}]",
                positive=True,
            )
            dr = _finite_real(
                _element_radial_value(radial, "dr", element),
                f"radial dr[{element!r}]",
                positive=True,
            )
            sigma = _finite_real(
                _element_radial_value(
                    radial, "smearing_sigma", element
                ),
                f"radial smearing_sigma[{element!r}]",
            )
            if sigma < 0.0:
                raise ValueError("radial smearing_sigma must be nonnegative")
            if not 0.0 < self._local_radius < rcut:
                raise ValueError(
                    "local_radius must satisfy 0 < local_radius < every Rcut"
                )
            try:
                element_eigenvalues = eigenvalues[element]
            except KeyError as exc:
                raise ValueError(
                    f"eigenvalues are missing element {element!r}"
                ) from exc
            if (
                not isinstance(element_eigenvalues, torch.Tensor)
                or element_eigenvalues.dtype != torch.float64
                or element_eigenvalues.is_complex()
                or element_eigenvalues.device.type != "cpu"
                or element_eigenvalues.shape != (info.Nl, info.Ne)
                or not bool(torch.all(torch.isfinite(element_eigenvalues)))
            ):
                raise ValueError(
                    f"eigenvalues[{element!r}] must be finite CPU float64 "
                    f"with shape {(info.Nl, info.Ne)}"
                )

            nmesh = int(round(rcut / dr)) + 1
            radius = torch.arange(nmesh, dtype=torch.float64) * dr
            if abs(float(radius[-1].item()) - rcut) > 1.0e-10 * max(rcut, 1.0):
                raise ValueError("Rcut/dr must define an integer radial mesh")
            trapezoid_weight = torch.full(
                (nmesh,), dr, dtype=torch.float64
            )
            trapezoid_weight[0] *= 0.5
            trapezoid_weight[-1] *= 0.5
            radial_weight = trapezoid_weight * radius.square()
            tail_weight = radial_weight * (radius >= self._local_radius)

            for l in range(info.Nl):
                values = np.stack(
                    [
                        spherical_jn(
                            l,
                            float(wavenumber.item()) * radius.numpy(),
                        )
                        for wavenumber in element_eigenvalues[l]
                    ],
                    axis=1,
                )
                basis = torch.from_numpy(values).to(torch.float64)
                if sigma > 0.0:
                    smooth = 1.0 - torch.exp(
                        -(radius - rcut).square() / (2.0 * sigma * sigma)
                    )
                    basis = basis * smooth.unsqueeze(1)
                overlap = basis.transpose(0, 1) @ (
                    radial_weight.unsqueeze(1) * basis
                )
                tail = basis.transpose(0, 1) @ (
                    tail_weight.unsqueeze(1) * basis
                )
                self._metrics[(element, l)] = (
                    (overlap + overlap.transpose(0, 1)) / 2.0,
                    (tail + tail.transpose(0, 1)) / 2.0,
                )

    def evaluate(self, coefficients):
        if not isinstance(coefficients, Mapping) or not coefficients:
            raise TypeError("coefficients must be a nonempty mapping")
        first = None
        for element in sorted(coefficients):
            for channel in coefficients[element]:
                if isinstance(channel, torch.Tensor):
                    first = channel
                    break
            if first is not None:
                break
        if first is None:
            raise ValueError("coefficients contain no tensor channels")
        zero = first.sum() * 0.0

        weighted_tail = zero
        variable_ao = 0
        max_condition = 1.0
        by_channel = {}
        if set(coefficients) != {
            element for element, _ in self._metrics
        }:
            raise ValueError("coefficient elements do not match locality metrics")

        for element, l in sorted(self._metrics):
            try:
                channel = coefficients[element][l]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"coefficients are missing channel {element}/{l}"
                ) from exc
            overlap, tail = self._metrics[(element, l)]
            if (
                not isinstance(channel, torch.Tensor)
                or channel.dtype != torch.float64
                or channel.is_complex()
                or channel.device.type != "cpu"
                or channel.ndim != 2
                or channel.shape[0] != overlap.shape[0]
                or not bool(torch.all(torch.isfinite(channel)))
            ):
                raise ValueError(
                    f"coefficients[{element!r}][{l}] must be finite CPU "
                    f"float64 with {overlap.shape[0]} rows"
                )

            fixed_indices = self._fixed.get((element, l), ())
            if fixed_indices and max(fixed_indices) >= channel.shape[1]:
                raise ValueError(
                    f"fixed radial spec is missing from {element}/{l}"
                )
            fixed_set = set(fixed_indices)
            variable_indices = tuple(
                index
                for index in range(channel.shape[1])
                if index not in fixed_set
            )
            nvariable = len(variable_indices)
            if nvariable == 0:
                by_channel[(element, l)] = RadialChannelLocality(
                    element,
                    l,
                    0,
                    zero,
                    1.0,
                )
                continue

            variable = channel[:, variable_indices]
            if fixed_indices:
                fixed = channel[:, fixed_indices]
                fixed_gram = fixed.transpose(0, 1) @ overlap @ fixed
                fixed_factor, fixed_condition = _factor_gram(
                    fixed_gram,
                    self._condition_limit,
                    "fixed radial overlap",
                )
                projection = torch.cholesky_solve(
                    fixed.transpose(0, 1) @ overlap @ variable,
                    fixed_factor,
                )
                variable = variable - fixed @ projection
                max_condition = max(max_condition, fixed_condition)

            variable_gram = variable.transpose(0, 1) @ overlap @ variable
            variable_factor, condition = _factor_gram(
                variable_gram,
                self._condition_limit,
                "projected variable radial overlap",
            )
            variable_tail = variable.transpose(0, 1) @ tail @ variable
            tail_fraction = torch.trace(
                torch.cholesky_solve(variable_tail, variable_factor)
            ) / nvariable
            detached_fraction = float(tail_fraction.detach().item())
            tolerance = 1.0e-10
            if (
                not math.isfinite(detached_fraction)
                or detached_fraction < -tolerance
                or detached_fraction > 1.0 + tolerance
            ):
                raise RuntimeError(
                    "radial tail fraction must be finite and between zero and one"
                )
            tail_fraction = torch.clamp(tail_fraction, min=0.0, max=1.0)
            ao_cost = (2 * l + 1) * nvariable
            weighted_tail = weighted_tail + ao_cost * tail_fraction
            variable_ao += ao_cost
            max_condition = max(max_condition, condition)
            by_channel[(element, l)] = RadialChannelLocality(
                element,
                l,
                nvariable,
                tail_fraction,
                condition,
            )

        loss = weighted_tail / variable_ao if variable_ao else zero
        return RadialLocalityResult(loss, max_condition, by_channel)
