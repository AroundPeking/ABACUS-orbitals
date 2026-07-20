from dataclasses import dataclass
import math
from collections.abc import Mapping

import torch


@dataclass(frozen=True)
class OrbitalColumn:
    element: str
    atom_index: int
    l: int
    m: int
    zeta: int


@dataclass(frozen=True)
class SternheimerLossResult:
    loss: torch.Tensor
    weighted_residual: torch.Tensor
    weighted_norm: torch.Tensor
    max_condition: float


@dataclass(frozen=True)
class RadialResidualSpectrum:
    element: str
    atom_index: int
    l: int
    magnetic_channels: tuple
    numerical_rank: int
    eigenvalues: torch.Tensor
    cumulative_capture: torch.Tensor
    coefficients: torch.Tensor
    overlap_relative_deviation: float
    atom_indices: tuple


@dataclass(frozen=True)
class RadialResidualTerms:
    projected_overlap: torch.Tensor
    covariance: torch.Tensor
    magnetic_channels: tuple
    overlap_relative_deviation: float
    atom_index: int


def _orbital_label(column):
    return (
        f"{column.element}/{column.atom_index}/{column.l}/{column.m}/"
        f"zeta{column.zeta}"
    )


def _coefficient_matrix(c, element, l):
    try:
        by_l = c[element]
        coefficient = by_l[l]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"C is missing element/l {element}/{l}") from exc
    if not isinstance(coefficient, torch.Tensor):
        raise ValueError(f"C[{element!r}][{l}] must be a torch.Tensor")
    if coefficient.dtype != torch.float64 or coefficient.is_complex():
        raise ValueError(f"C[{element!r}][{l}] must be real float64")
    if coefficient.device.type != "cpu":
        raise ValueError(f"C[{element!r}][{l}] must be on CPU")
    if coefficient.ndim != 2:
        raise ValueError(f"C[{element!r}][{l}] must have rank 2")
    if not bool(torch.all(torch.isfinite(coefficient))):
        raise ValueError(
            f"C[{element!r}][{l}] must contain only finite values"
        )
    return coefficient


def assemble_orbital_coefficients(data, c):
    block_data = []
    labels = []
    seen_keys = set()
    represented_channels = set()
    group_m_values = {}
    atoms_by_element = {}
    n_column = 0

    for block in data.blocks:
        if block.key in seen_keys:
            raise ValueError(f"duplicate PrimitiveBlock key: {block.key}")
        seen_keys.add(block.key)
        represented_channels.add((block.element, block.l))
        group_key = (block.element, block.atom_index, block.l)
        group_m_values.setdefault(group_key, []).append(block.m)
        atoms_by_element.setdefault(block.element, set()).add(
            block.atom_index
        )

        coefficient = _coefficient_matrix(c, block.element, block.l)
        if coefficient.shape[0] != block.n_primitive:
            raise ValueError(
                f"C[{block.element!r}][{block.l}] radial row count "
                f"{coefficient.shape[0]} does not match block.n_primitive "
                f"{block.n_primitive}"
            )
        block_data.append((block, coefficient, n_column))
        for zeta in range(1, coefficient.shape[1] + 1):
            labels.append(
                OrbitalColumn(
                    block.element,
                    block.atom_index,
                    block.l,
                    block.m,
                    zeta,
                )
            )
        n_column += coefficient.shape[1]

    for (element, atom_index, l), m_values in sorted(group_m_values.items()):
        expected_m = tuple(range(-l, l + 1))
        actual_m = tuple(sorted(m_values))
        if actual_m != expected_m:
            raise ValueError(
                "incomplete PrimitiveBlock m group for "
                f"{element}/atom{atom_index}/l{l}: "
                f"expected {expected_m}, got {actual_m}"
            )

    missing_channels = []
    nonempty_channels = []
    for element, by_l in c.items():
        if isinstance(by_l, Mapping):
            channels = by_l.items()
        else:
            try:
                channels = enumerate(by_l)
            except TypeError as exc:
                raise ValueError(
                    f"C[{element!r}] must contain angular channels"
                ) from exc
        for l, coefficient in channels:
            coefficient = _coefficient_matrix(c, element, l)
            if (
                coefficient.shape[0] > 0
                and coefficient.shape[1] > 0
            ):
                nonempty_channels.append((element, l))
                if (element, l) not in represented_channels:
                    missing_channels.append((element, l))
    if missing_channels:
        formatted = ", ".join(
            f"{element}/{l}" for element, l in sorted(missing_channels)
        )
        raise ValueError(
            "Sternheimer data is missing primitive blocks for C channels: "
            + formatted
        )

    missing_groups = []
    for element, l in nonempty_channels:
        for atom_index in sorted(atoms_by_element.get(element, ())):
            if (element, atom_index, l) not in group_m_values:
                missing_groups.append((element, atom_index, l))
    if missing_groups:
        element, atom_index, l = sorted(missing_groups)[0]
        raise ValueError(
            "Sternheimer data is missing PrimitiveBlock group for "
            f"{element}/atom{atom_index}/l{l}"
        )

    assembled = torch.zeros(
        (data.q.shape[1], n_column),
        dtype=torch.complex128,
        device=data.q.device,
    )
    for block, coefficient, column_offset in block_data:
        row_slice = slice(block.offset, block.offset + block.n_primitive)
        column_slice = slice(column_offset, column_offset + coefficient.shape[1])
        assembled[row_slice, column_slice] = coefficient.to(torch.complex128)

    return assembled, tuple(labels)


def _factor_hermitian(matrix, condition_limit, name, positive_error):
    hermitian = (matrix + matrix.conj().transpose(0, 1)) / 2.0
    factor, info = torch.linalg.cholesky_ex(hermitian)
    if int(info.item()) != 0:
        raise RuntimeError(positive_error)

    condition = float(torch.linalg.cond(hermitian).item())
    if not math.isfinite(condition):
        raise RuntimeError(f"{name} condition number is not finite")
    if condition > condition_limit:
        raise RuntimeError(
            f"{name} condition number {condition:.6g} exceeds "
            f"condition_limit {condition_limit:.6g}"
        )
    return factor, condition


def _normalize_condition_limit(condition_limit):
    try:
        condition_limit = float(condition_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("condition_limit must be finite and at least 1") from exc
    if not math.isfinite(condition_limit) or condition_limit < 1.0:
        raise ValueError("condition_limit must be finite and at least 1")
    return condition_limit


def _row_diagonal(q, solve):
    return torch.sum(q * solve.transpose(0, 1), dim=1)


def _clamp_roundoff_negative(values, local_scale, name, detail):
    tolerance = 1.0e-10 * local_scale
    negative = values < -tolerance
    if bool(torch.any(negative)):
        row = int(torch.nonzero(negative, as_tuple=False)[0].item())
        raise RuntimeError(
            f"materially negative {name} at reference {row} "
            f"({float(values[row].item()):.6g}, "
            f"tolerance {float(tolerance[row].item()):.6g}); {detail}"
        )
    return torch.clamp(values, min=0.0)


def _normalize_spectrum_tolerances(
    relative_rank_tolerance, magnetic_overlap_tolerance
):
    normalized = []
    for name, value in (
        ("relative_rank_tolerance", relative_rank_tolerance),
        ("magnetic_overlap_tolerance", magnetic_overlap_tolerance),
    ):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and positive") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        normalized.append(value)
    return tuple(normalized)


def _fixed_projection(data, c0, fixed_orbitals, condition_limit):
    assembled, labels = assemble_orbital_coefficients(data, c0)
    fixed_orbitals = tuple(fixed_orbitals)
    if not fixed_orbitals:
        raise ValueError("fixed orbital space must be nonempty")
    if len(set(fixed_orbitals)) != len(fixed_orbitals):
        raise ValueError("a fixed orbital was requested more than once")
    for value in fixed_orbitals:
        if not isinstance(value, OrbitalColumn):
            raise ValueError("fixed orbitals must be OrbitalColumn values")
        if labels.count(value) != 1:
            raise ValueError(
                f"fixed orbital {_orbital_label(value)} not found exactly once"
            )

    fixed_set = set(fixed_orbitals)
    fixed_indices = tuple(
        index for index, label in enumerate(labels) if label in fixed_set
    )
    a0 = assembled[:, fixed_indices]
    overlap = data.overlap
    s00 = a0.conj().transpose(0, 1) @ overlap @ a0
    s00_cholesky, _ = _factor_hermitian(
        s00,
        condition_limit,
        "fixed overlap",
        "fixed overlap is not positive definite",
    )
    q0 = data.q @ a0
    return assembled, labels, a0, s00_cholesky, q0


def _radial_residual_terms(
    data,
    c0,
    fixed_orbitals,
    element,
    atom_index,
    l,
    magnetic_overlap_tolerance,
    condition_limit,
):
    _, _, a0, s00_cholesky, q0 = _fixed_projection(
        data, c0, fixed_orbitals, condition_limit
    )
    overlap = data.overlap

    blocks = sorted(
        (
            block
            for block in data.blocks
            if block.element == element
            and block.atom_index == atom_index
            and block.l == l
        ),
        key=lambda block: block.m,
    )
    expected_m = tuple(range(-l, l + 1))
    actual_m = tuple(block.m for block in blocks)
    if actual_m != expected_m:
        raise ValueError(
            "incomplete PrimitiveBlock m group for "
            f"{element}/atom{atom_index}/l{l}: "
            f"expected {expected_m}, got {actual_m}"
        )
    primitive_counts = {block.n_primitive for block in blocks}
    if len(primitive_counts) != 1:
        raise ValueError(
            "magnetic PrimitiveBlock channels must share one radial count"
        )

    projected_overlaps = []
    covariance = None
    weight = data.effective_weight.to(torch.complex128).unsqueeze(1)
    for block in blocks:
        block_slice = slice(block.offset, block.offset + block.n_primitive)
        s0m = a0.conj().transpose(0, 1) @ overlap[:, block_slice]
        s00_inverse_s0m = torch.cholesky_solve(s0m, s00_cholesky)
        qbar = data.q[:, block_slice] - q0 @ s00_inverse_s0m
        sbar = (
            overlap[block_slice, block_slice]
            - s0m.conj().transpose(0, 1) @ s00_inverse_s0m
        )
        sbar = (sbar + sbar.conj().transpose(0, 1)) / 2.0
        imaginary_scale = max(float(torch.linalg.norm(sbar.real).item()), 1.0)
        if (
            float(torch.linalg.norm(sbar.imag).item())
            > 1.0e-10 * imaginary_scale
        ):
            raise RuntimeError(
                "projected primitive overlap has a material imaginary part"
            )
        projected_overlaps.append(sbar.real)

        block_covariance = qbar.conj().transpose(0, 1) @ (weight * qbar)
        covariance = (
            block_covariance
            if covariance is None
            else covariance + block_covariance
        )

    average_overlap = sum(projected_overlaps) / len(projected_overlaps)
    overlap_scale = max(float(torch.linalg.norm(average_overlap).item()), 1.0)
    overlap_relative_deviation = max(
        float(torch.linalg.norm(value - average_overlap).item()) / overlap_scale
        for value in projected_overlaps
    )
    if overlap_relative_deviation > magnetic_overlap_tolerance:
        raise RuntimeError(
            "magnetic-channel projected overlaps disagree: "
            f"relative deviation {overlap_relative_deviation:.6g} exceeds "
            f"{magnetic_overlap_tolerance:.6g}"
        )

    covariance = (covariance + covariance.conj().transpose(0, 1)) / 2.0
    return RadialResidualTerms(
        projected_overlap=average_overlap,
        covariance=covariance,
        magnetic_channels=actual_m,
        overlap_relative_deviation=overlap_relative_deviation,
        atom_index=atom_index,
    )


def _diagonalize_radial_terms(
    terms,
    element,
    atom_index,
    l,
    atom_indices,
    relative_rank_tolerance,
    magnetic_overlap_tolerance,
):
    terms = tuple(terms)
    if not terms:
        raise ValueError("radial residual terms must be nonempty")
    expected_channels = terms[0].magnetic_channels
    expected_shape = terms[0].projected_overlap.shape
    for value in terms[1:]:
        if value.magnetic_channels != expected_channels:
            raise ValueError("radial residual terms have incompatible m channels")
        if value.projected_overlap.shape != expected_shape:
            raise ValueError(
                "radial residual terms have incompatible primitive counts"
            )

    average_overlap = sum(value.projected_overlap for value in terms) / len(terms)
    overlap_scale = max(float(torch.linalg.norm(average_overlap).item()), 1.0)
    cross_term_deviation = max(
        float(torch.linalg.norm(value.projected_overlap - average_overlap).item())
        / overlap_scale
        for value in terms
    )
    overlap_relative_deviation = max(
        cross_term_deviation,
        max(value.overlap_relative_deviation for value in terms),
    )
    if overlap_relative_deviation > magnetic_overlap_tolerance:
        raise RuntimeError(
            "target/atom projected overlaps disagree: "
            f"relative deviation {overlap_relative_deviation:.6g} exceeds "
            f"{magnetic_overlap_tolerance:.6g}"
        )

    average_overlap = (average_overlap + average_overlap.transpose(0, 1)) / 2.0
    overlap_eigenvalues, overlap_eigenvectors = torch.linalg.eigh(
        average_overlap
    )
    largest_overlap = float(torch.max(overlap_eigenvalues).item())
    if not math.isfinite(largest_overlap) or largest_overlap <= 0.0:
        raise RuntimeError("projected primitive overlap has no positive modes")
    rank_cutoff = relative_rank_tolerance * largest_overlap
    if float(torch.min(overlap_eigenvalues).item()) < -rank_cutoff:
        raise RuntimeError("projected primitive overlap is materially indefinite")
    keep = overlap_eigenvalues > rank_cutoff
    numerical_rank = int(torch.count_nonzero(keep).item())
    if numerical_rank == 0:
        raise RuntimeError("projected primitive overlap has numerical rank zero")
    whitener = overlap_eigenvectors[:, keep] / torch.sqrt(
        overlap_eigenvalues[keep]
    ).unsqueeze(0)

    covariance = sum(value.covariance for value in terms)
    covariance = (covariance + covariance.conj().transpose(0, 1)) / 2.0
    real_covariance = covariance.real
    whitened_covariance = (
        whitener.transpose(0, 1) @ real_covariance @ whitener
    )
    whitened_covariance = (
        whitened_covariance + whitened_covariance.transpose(0, 1)
    ) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened_covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvalue_scale = max(float(torch.max(torch.abs(eigenvalues)).item()), 1.0)
    if float(torch.min(eigenvalues).item()) < -1.0e-10 * eigenvalue_scale:
        raise RuntimeError("radial residual covariance is materially indefinite")
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    coefficients = whitener @ eigenvectors
    total_capture = torch.sum(eigenvalues)
    if bool(total_capture > 0.0):
        cumulative_capture = torch.cumsum(eigenvalues, dim=0) / total_capture
    else:
        cumulative_capture = torch.zeros_like(eigenvalues)

    return RadialResidualSpectrum(
        element=element,
        atom_index=atom_index,
        l=l,
        magnetic_channels=expected_channels,
        numerical_rank=numerical_rank,
        eigenvalues=eigenvalues,
        cumulative_capture=cumulative_capture,
        coefficients=coefficients,
        overlap_relative_deviation=overlap_relative_deviation,
        atom_indices=tuple(atom_indices),
    )


def radial_residual_spectrum(
    data,
    c0,
    fixed_orbitals,
    element,
    atom_index,
    l,
    relative_rank_tolerance=1.0e-10,
    magnetic_overlap_tolerance=1.0e-8,
    condition_limit=1.0e12,
):
    """Return the optimal shared radial response spectrum for one l channel."""
    relative_rank_tolerance, magnetic_overlap_tolerance = (
        _normalize_spectrum_tolerances(
            relative_rank_tolerance, magnetic_overlap_tolerance
        )
    )
    terms = _radial_residual_terms(
        data,
        c0,
        fixed_orbitals,
        element,
        atom_index,
        l,
        magnetic_overlap_tolerance,
        condition_limit,
    )
    return _diagonalize_radial_terms(
        (terms,),
        element,
        atom_index,
        l,
        (atom_index,),
        relative_rank_tolerance,
        magnetic_overlap_tolerance,
    )


def _expand_fixed_radial_specs(labels, fixed_specs):
    fixed_specs = tuple(fixed_specs)
    if not fixed_specs:
        raise ValueError("fixed radial specs must be nonempty")

    selected = []
    for spec in fixed_specs:
        if not isinstance(spec, Mapping) or set(spec) != {
            "element",
            "l",
            "zeta",
        }:
            raise ValueError("fixed radial spec requires element, l, and zeta")
        element = spec["element"]
        l = spec["l"]
        zeta = spec["zeta"]
        if not isinstance(element, str) or not element:
            raise ValueError("fixed radial spec element must be nonempty")
        if type(l) is not int or l < 0:
            raise ValueError("fixed radial spec l must be a nonnegative integer")
        if type(zeta) is not int or zeta <= 0:
            raise ValueError("fixed radial spec zeta must be a positive integer")
        matching = tuple(
            label
            for label in labels
            if label.element == element
            and label.l == l
            and label.zeta == zeta
        )
        if not matching:
            raise ValueError(
                f"fixed radial spec {(element, l, zeta)!r} maps to no columns"
            )
        selected.extend(matching)

    if len(set(selected)) != len(selected):
        raise ValueError("a fixed radial spec was requested more than once")
    selected_set = set(selected)
    return tuple(label for label in labels if label in selected_set)


def radial_residual_spectrum_many(
    data_items,
    c0,
    fixed_specs,
    element,
    l,
    relative_rank_tolerance=1.0e-4,
    magnetic_overlap_tolerance=1.0e-4,
    condition_limit=1.0e12,
):
    """Aggregate one shared radial spectrum over targets and atom centers."""
    data_items = tuple(data_items)
    if not data_items:
        raise ValueError("data_items must be nonempty")
    relative_rank_tolerance, magnetic_overlap_tolerance = (
        _normalize_spectrum_tolerances(
            relative_rank_tolerance, magnetic_overlap_tolerance
        )
    )

    terms = []
    atom_indices = set()
    for data in data_items:
        _, labels = assemble_orbital_coefficients(data, c0)
        fixed_orbitals = _expand_fixed_radial_specs(labels, fixed_specs)
        target_atoms = sorted(
            {
                block.atom_index
                for block in data.blocks
                if block.element == element and block.l == l
            }
        )
        if not target_atoms:
            raise ValueError(
                f"Sternheimer target has no {element}/l{l} primitive blocks"
            )
        for atom_index in target_atoms:
            atom_indices.add(atom_index)
            terms.append(
                _radial_residual_terms(
                    data,
                    c0,
                    fixed_orbitals,
                    element,
                    atom_index,
                    l,
                    magnetic_overlap_tolerance,
                    condition_limit,
                )
            )

    return _diagonalize_radial_terms(
        terms,
        element,
        None,
        l,
        tuple(sorted(atom_indices)),
        relative_rank_tolerance,
        magnetic_overlap_tolerance,
    )


def shell_count_for_capture(spectrum, threshold):
    if not isinstance(spectrum, RadialResidualSpectrum):
        raise TypeError("spectrum must be a RadialResidualSpectrum")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must satisfy 0 < threshold <= 1") from exc
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must satisfy 0 < threshold <= 1")
    if not bool(torch.any(spectrum.eigenvalues > 0.0)):
        raise RuntimeError("radial residual spectrum has zero captured weight")
    for index, value in enumerate(spectrum.cumulative_capture):
        if float(value.item()) >= threshold:
            return index + 1
    return spectrum.numerical_rank


def evaluate_spillage_for_columns(
    data, c, include, condition_limit=1.0e12
):
    """Evaluate the full reference residual for an explicit AO projector."""
    if not callable(include):
        raise TypeError("include must be callable")
    condition_limit = _normalize_condition_limit(condition_limit)
    assembled, labels = assemble_orbital_coefficients(data, c)
    indices = tuple(
        index for index, label in enumerate(labels) if bool(include(label))
    )
    if not indices:
        raise ValueError("selected projector contains no orbital columns")

    selected = assembled[:, indices]
    overlap = selected.conj().transpose(0, 1) @ data.overlap @ selected
    names = ", ".join(_orbital_label(labels[index]) for index in indices)
    cholesky, condition = _factor_hermitian(
        overlap,
        condition_limit,
        "selected projector overlap",
        "selected projector overlap is not positive definite; "
        f"selected columns: {names}",
    )
    q = data.q @ selected
    represented = _row_diagonal(
        q, torch.cholesky_solve(q.conj().transpose(0, 1), cholesky)
    ).real
    residual_scale = torch.maximum(
        torch.maximum(torch.abs(data.norm), torch.abs(represented)),
        torch.ones_like(data.norm),
    )
    residual = _clamp_roundoff_negative(
        data.norm - represented,
        residual_scale,
        "selected-projector residual",
        "the selected projector represents more than the reference norm; "
        "check q and overlap",
    )

    weight = data.effective_weight
    weighted_norm = torch.sum(weight * data.norm)
    if not bool(torch.isfinite(weighted_norm)) or not bool(weighted_norm > 0.0):
        raise RuntimeError("weighted reference norm must be positive and finite")
    weighted_residual = torch.sum(weight * residual)
    if not bool(torch.isfinite(weighted_residual)):
        raise RuntimeError("weighted selected-projector residual must be finite")
    return SternheimerLossResult(
        loss=weighted_residual / weighted_norm,
        weighted_residual=weighted_residual,
        weighted_norm=weighted_norm,
        max_condition=condition,
    )


class SternheimerSpillage:
    def __init__(
        self,
        data,
        c0,
        fixed_orbitals,
        condition_limit=1.0e12,
    ):
        condition_limit = _normalize_condition_limit(condition_limit)

        assembled, labels = assemble_orbital_coefficients(data, c0)
        fixed_orbitals = tuple(fixed_orbitals)
        if not fixed_orbitals:
            raise ValueError("fixed orbital space must be nonempty")
        for orbital in fixed_orbitals:
            if not isinstance(orbital, OrbitalColumn):
                raise ValueError("fixed orbitals must be OrbitalColumn values")
            count = labels.count(orbital)
            if count != 1:
                raise ValueError(
                    f"fixed orbital {_orbital_label(orbital)} not found exactly once"
                )
        if len(set(fixed_orbitals)) != len(fixed_orbitals):
            raise ValueError("a fixed orbital was requested more than once")

        fixed_set = set(fixed_orbitals)
        fixed_indices = tuple(
            index for index, label in enumerate(labels) if label in fixed_set
        )
        variable_indices = tuple(
            index for index, label in enumerate(labels) if label not in fixed_set
        )
        if not variable_indices:
            raise ValueError("variable orbital space must be nonempty")

        self._data = data
        self._condition_limit = condition_limit
        self._labels = labels
        self._variable_indices = variable_indices
        self._variable_labels = tuple(labels[index] for index in variable_indices)

        self._a0 = assembled[:, fixed_indices].detach().clone()
        s00 = self._a0.conj().transpose(0, 1) @ data.overlap @ self._a0
        self._s00_cholesky, self._fixed_condition = _factor_hermitian(
            s00,
            condition_limit,
            "fixed overlap",
            "fixed overlap is not positive definite",
        )
        self._s00_cholesky = self._s00_cholesky.detach().clone()

        self._q0 = (data.q @ self._a0).detach().clone()
        self._s00_inverse_q0_h = torch.cholesky_solve(
            self._q0.conj().transpose(0, 1), self._s00_cholesky
        ).detach().clone()
        fixed_represented = _row_diagonal(
            self._q0, self._s00_inverse_q0_h
        ).real
        nbar_scale = torch.maximum(
            torch.maximum(torch.abs(data.norm), torch.abs(fixed_represented)),
            torch.ones_like(data.norm),
        )
        self._nbar = _clamp_roundoff_negative(
            data.norm - fixed_represented,
            nbar_scale,
            "projected reference norm nbar",
            "check norm, q, overlap, and the requested fixed orbitals",
        ).detach().clone()

    def evaluate(self, c):
        assembled, labels = assemble_orbital_coefficients(self._data, c)
        if labels != self._labels:
            raise ValueError("orbital labels changed from the constructor assembly")

        a1 = assembled[:, self._variable_indices]
        overlap = self._data.overlap
        s01 = self._a0.conj().transpose(0, 1) @ overlap @ a1
        s11 = a1.conj().transpose(0, 1) @ overlap @ a1
        s00_inverse_s01 = torch.cholesky_solve(s01, self._s00_cholesky)
        q1 = self._data.q @ a1
        qbar = q1 - self._q0 @ s00_inverse_s01
        sbar = (
            s11
            - s01.conj().transpose(0, 1) @ s00_inverse_s01
        )

        variable_names = ", ".join(
            _orbital_label(label) for label in self._variable_labels
        )
        sbar_cholesky, variable_condition = _factor_hermitian(
            sbar,
            self._condition_limit,
            "variable overlap",
            "variable overlap is not positive definite; "
            f"variable columns: {variable_names}",
        )
        sbar_inverse_qbar_h = torch.cholesky_solve(
            qbar.conj().transpose(0, 1), sbar_cholesky
        )
        represented = _row_diagonal(qbar, sbar_inverse_qbar_h).real
        residual_scale = torch.maximum(
            torch.maximum(torch.abs(self._nbar), torch.abs(represented)),
            torch.ones_like(self._nbar),
        )
        residual = _clamp_roundoff_negative(
            self._nbar - represented,
            residual_scale,
            "projected residual",
            "the variable projector represents more than nbar; check q and overlap",
        )

        weight = self._data.effective_weight
        weighted_norm = torch.sum(weight * self._nbar)
        if not bool(torch.isfinite(weighted_norm)) or not bool(weighted_norm > 0.0):
            raise RuntimeError("weighted projected norm must be positive and finite")
        weighted_residual = torch.sum(weight * residual)
        if not bool(torch.isfinite(weighted_residual)):
            raise RuntimeError("weighted projected residual must be finite")
        loss = weighted_residual / weighted_norm

        return SternheimerLossResult(
            loss=loss,
            weighted_residual=weighted_residual,
            weighted_norm=weighted_norm,
            max_condition=max(self._fixed_condition, variable_condition),
        )
