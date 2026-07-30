#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    coefficients: np.ndarray
    represented_norm: float
    rank: int


@dataclass(frozen=True)
class TargetArrays:
    blocks: tuple
    q: np.ndarray
    overlap: np.ndarray
    norm: np.ndarray
    occupation: np.ndarray
    frequency_weight: np.ndarray
    frequency_ha: np.ndarray
    auxiliary_channel: np.ndarray

    @property
    def weight(self):
        return self.occupation * self.frequency_weight


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_orbitals(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    mesh = None
    dr = None
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] == "Mesh":
            mesh = int(fields[1])
        if len(fields) == 2 and fields[0] == "dr":
            dr = float(fields[1])
    if mesh is None or mesh <= 0 or dr is None or dr <= 0.0:
        raise ValueError(f"invalid orbital mesh in {path}")

    orbitals = {}
    index = 0
    while index < len(lines):
        if lines[index].split() != ["Type", "L", "N"]:
            index += 1
            continue
        index += 1
        if index >= len(lines):
            raise ValueError(f"missing orbital label in {path}")
        fields = lines[index].split()
        if len(fields) != 3:
            raise ValueError(f"invalid orbital label in {path}: {lines[index]}")
        key = (int(fields[1]), int(fields[2]))
        index += 1
        values = []
        while index < len(lines):
            fields = lines[index].split()
            if fields == ["Type", "L", "N"]:
                break
            values.extend(float(value) for value in fields)
            index += 1
        if len(values) != mesh:
            raise ValueError(
                f"orbital {key} in {path} has {len(values)} points, "
                f"expected {mesh}"
            )
        if key in orbitals:
            raise ValueError(f"duplicate orbital {key} in {path}")
        orbitals[key] = np.asarray(values, dtype=np.float64)

    if not orbitals:
        raise ValueError(f"no radial orbitals found in {path}")
    return np.arange(mesh, dtype=np.float64) * dr, orbitals


def read_coefficients(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    columns = {}
    index = 0
    while index < len(lines):
        if lines[index].split() != ["Type", "L", "Zeta-Orbital"]:
            index += 1
            continue
        index += 1
        if index >= len(lines):
            raise ValueError(f"missing coefficient label in {path}")
        fields = lines[index].split()
        if len(fields) != 3:
            raise ValueError(f"invalid coefficient label in {path}")
        element, l, zeta = fields[0], int(fields[1]), int(fields[2])
        index += 1
        values = []
        while index < len(lines):
            fields = lines[index].split()
            if fields == ["Type", "L", "Zeta-Orbital"]:
                break
            if fields == ["</Coefficient>"]:
                break
            if len(fields) != 1:
                raise ValueError(
                    f"invalid coefficient value in {path}: {lines[index]}"
                )
            values.append(float(fields[0]))
            index += 1
        key = (element, l)
        columns.setdefault(key, []).append((zeta, values))

    result = {}
    for key, values in columns.items():
        ordered = sorted(values)
        if [zeta for zeta, _ in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError(f"non-contiguous zeta columns for {key} in {path}")
        lengths = {len(column) for _, column in ordered}
        if len(lengths) != 1:
            raise ValueError(f"inconsistent coefficient lengths for {key}")
        result[key] = np.asarray(
            [column for _, column in ordered], dtype=np.float64
        ).T
    if not result:
        raise ValueError(f"no coefficients found in {path}")
    return result


def rank_revealing_projection(overlap, q, relative_rank_tolerance=1.0e-4):
    overlap = np.asarray(overlap, dtype=np.complex128)
    q = np.asarray(q, dtype=np.complex128)
    hermitian = (overlap + overlap.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    largest = float(eigenvalues[-1])
    if not math.isfinite(largest) or largest <= 0.0:
        raise ValueError("overlap has no positive eigenvalue")
    cutoff = float(relative_rank_tolerance) * largest
    if float(eigenvalues[0]) < -cutoff:
        raise ValueError("overlap is materially indefinite")
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise ValueError("overlap has numerical rank zero")
    vectors = eigenvectors[:, keep]
    coefficients = vectors @ (
        (vectors.conj().T @ q.conj()) / eigenvalues[keep]
    )
    represented = float(np.real(q @ coefficients))
    if represented < -1.0e-10 * max(1.0, abs(represented)):
        raise ValueError("projection has negative represented norm")
    return ProjectionResult(
        coefficients=coefficients,
        represented_norm=max(0.0, represented),
        rank=int(np.count_nonzero(keep)),
    )


def classify_reference(
    block_overlaps, q_by_l, relative_rank_tolerance=1.0e-4
):
    represented = {}
    for l, q in q_by_l.items():
        represented[l] = rank_revealing_projection(
            block_overlaps[l], q, relative_rank_tolerance
        ).represented_norm
    if not represented:
        raise ValueError("reference contains no angular blocks")
    dominant_l = max(represented, key=represented.get)
    return dominant_l, represented


def load_target(path, optimizer_dir):
    optimizer_dir = str(Path(optimizer_dir).resolve())
    if optimizer_dir not in sys.path:
        sys.path.insert(0, optimizer_dir)
    from IO.read_sternheimer import read_sternheimer

    data = read_sternheimer(path)
    return TargetArrays(
        blocks=tuple(data.blocks),
        q=data.q.numpy(),
        overlap=data.overlap.numpy(),
        norm=data.norm.numpy(),
        occupation=data.occupation.numpy(),
        frequency_weight=data.frequency_weight.numpy(),
        frequency_ha=data.frequency_ha.numpy(),
        auxiliary_channel=data.auxiliary_channel.numpy(),
    )


def _block_slice(block):
    return slice(block.offset, block.offset + block.n_primitive)


def _coefficient_for_block(coefficients, block):
    key = (block.element, block.l)
    if key not in coefficients:
        raise ValueError(f"coefficients are missing {key}")
    value = coefficients[key]
    if value.shape[0] != block.n_primitive:
        raise ValueError(
            f"coefficient row count for {key} is {value.shape[0]}, "
            f"expected {block.n_primitive}"
        )
    return value


def assemble_projector(target, coefficients, select):
    selections = []
    for block in target.blocks:
        value = _coefficient_for_block(coefficients, block)
        indices = tuple(index for index in range(value.shape[1]) if select(block, index))
        selections.append((block, value, indices))
    column_count = sum(len(indices) for _, _, indices in selections)
    if column_count == 0:
        raise ValueError("projector contains no columns")
    assembled = np.zeros(
        (target.overlap.shape[0], column_count), dtype=np.complex128
    )
    column = 0
    for block, value, indices in selections:
        for index in indices:
            assembled[_block_slice(block), column] = value[:, index]
            column += 1
    return assembled


def fixed_context(target, coefficients):
    fixed_counts = {0: 2, 1: 1}
    fixed = assemble_projector(
        target,
        coefficients,
        lambda block, index: index < fixed_counts.get(block.l, 0),
    )
    s00 = fixed.conj().T @ target.overlap @ fixed
    q0 = target.q @ fixed
    s00_inverse = np.linalg.inv((s00 + s00.conj().T) / 2.0)
    represented = np.real(
        np.einsum("ij,ji->i", q0, s00_inverse @ q0.conj().T)
    )
    nbar = target.norm - represented
    if float(np.min(nbar)) < -1.0e-9:
        raise ValueError("fixed DZP projector exceeds a reference norm")
    nbar = np.maximum(nbar, 0.0)
    weighted_norm = float(np.sum(target.weight * nbar))
    return {
        "fixed": fixed,
        "s00_inverse": s00_inverse,
        "q0": q0,
        "nbar": nbar,
        "weighted_norm": weighted_norm,
    }


def evaluate_extra_orbitals(target, coefficients, context):
    fixed_counts = {0: 2, 1: 1}
    variable = assemble_projector(
        target,
        coefficients,
        lambda block, index: index >= fixed_counts.get(block.l, 0),
    )
    fixed = context["fixed"]
    s01 = fixed.conj().T @ target.overlap @ variable
    sbar = (
        variable.conj().T @ target.overlap @ variable
        - s01.conj().T @ context["s00_inverse"] @ s01
    )
    qbar = target.q @ variable - context["q0"] @ context["s00_inverse"] @ s01
    solve = np.linalg.solve((sbar + sbar.conj().T) / 2.0, qbar.conj().T)
    represented = np.real(np.einsum("ij,ji->i", qbar, solve))
    residual = np.maximum(context["nbar"] - represented, 0.0)
    weighted_residual = float(np.sum(target.weight * residual))
    return weighted_residual / context["weighted_norm"]


def radial_residual_spectrum(
    target,
    coefficients,
    context,
    l,
    relative_rank_tolerance=1.0e-4,
):
    fixed = context["fixed"]
    overlaps = []
    covariance = None
    for block in target.blocks:
        if block.l != l:
            continue
        block_slice = _block_slice(block)
        s0m = fixed.conj().T @ target.overlap[:, block_slice]
        qbar = (
            target.q[:, block_slice]
            - context["q0"] @ context["s00_inverse"] @ s0m
        )
        sbar = (
            target.overlap[block_slice, block_slice]
            - s0m.conj().T @ context["s00_inverse"] @ s0m
        )
        overlaps.append(np.real((sbar + sbar.conj().T) / 2.0))
        value = qbar.conj().T @ (target.weight[:, None] * qbar)
        covariance = value if covariance is None else covariance + value
    if not overlaps:
        raise ValueError(f"target contains no l={l} primitive blocks")

    overlap = sum(overlaps) / len(overlaps)
    eigenvalues_s, eigenvectors_s = np.linalg.eigh(overlap)
    largest = float(eigenvalues_s[-1])
    keep = eigenvalues_s > relative_rank_tolerance * largest
    if not np.any(keep):
        raise ValueError(f"projected l={l} primitive overlap has rank zero")
    whitener = eigenvectors_s[:, keep] / np.sqrt(eigenvalues_s[keep])[None, :]
    covariance = np.real((covariance + covariance.conj().T) / 2.0)
    whitened = whitener.T @ covariance @ whitener
    eigenvalues, eigenvectors = np.linalg.eigh((whitened + whitened.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vectors = whitener @ eigenvectors[:, order]
    return {
        "l": l,
        "overlap": overlap,
        "rank": int(np.count_nonzero(keep)),
        "eigenvalues": eigenvalues,
        "coefficients": vectors,
    }


def _spherical_bessel_zeros(l, count):
    from scipy.optimize import brentq
    from scipy.special import spherical_jn

    if l == 0:
        return np.arange(1, count + 1, dtype=np.float64) * np.pi
    roots = []
    step = np.pi / 16.0
    left = 1.0e-8
    fleft = float(spherical_jn(l, left))
    right = left + step
    while len(roots) < count:
        fright = float(spherical_jn(l, right))
        if fleft * fright < 0.0:
            roots.append(brentq(lambda value: spherical_jn(l, value), left, right))
        left, fleft = right, fright
        right += step
        if right > (count + l + 4) * np.pi:
            raise RuntimeError(f"failed to find {count} spherical-Bessel roots")
    return np.asarray(roots, dtype=np.float64)


def primitive_radials(l, count, radius, rcut=8.0):
    from scipy.special import spherical_jn

    wave_numbers = _spherical_bessel_zeros(l, count) / float(rcut)
    return np.column_stack(
        [spherical_jn(l, wave_number * radius) for wave_number in wave_numbers]
    )


def radial_inner(left, right, radius):
    return float(np.trapz(left * right * radius * radius, radius))


def normalize_radial(radial, radius):
    norm = radial_inner(radial, radial, radius)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("radial function has nonpositive norm")
    return radial / math.sqrt(norm)


def align_radial(reference, value, radius):
    if radial_inner(reference, value, radius) < 0.0:
        return -value
    return value


def _nodes(radial):
    threshold = 1.0e-2 * float(np.max(np.abs(radial)))
    values = radial[np.abs(radial) >= threshold]
    if len(values) < 2:
        return 0
    return int(np.count_nonzero(values[:-1] * values[1:] < 0.0))


def output_like_mode(mode, l, radius, fixed_orbitals, sigma=0.1):
    rcut = float(radius[-1])
    smoothing = 1.0 - np.exp(-((radius - rcut) ** 2) / (2.0 * sigma * sigma))
    value = np.real(mode) * smoothing
    for zeta in range({0: 2, 1: 1}[l]):
        fixed = fixed_orbitals[(l, zeta)]
        value = value - fixed * radial_inner(fixed, value, radius)
    return normalize_radial(value, radius)


def represented_by_block(target, row, relative_rank_tolerance):
    represented = []
    for block in target.blocks:
        block_slice = _block_slice(block)
        result = rank_revealing_projection(
            target.overlap[block_slice, block_slice],
            target.q[row, block_slice],
            relative_rank_tolerance,
        )
        represented.append((block, result))
    return represented


def select_representative_channels(target, relative_rank_tolerance):
    first_frequency = float(np.min(target.frequency_ha))
    indices = np.flatnonzero(np.isclose(target.frequency_ha, first_frequency))
    selected = {}
    for row in indices:
        represented = represented_by_block(target, row, relative_rank_tolerance)
        by_l = {}
        for block, result in represented:
            by_l[block.l] = by_l.get(block.l, 0.0) + result.represented_norm
        dominant_l = max(by_l, key=by_l.get)
        if by_l[dominant_l] <= 1.0e-14 * max(1.0, float(target.norm[row])):
            continue
        score = float(target.norm[row])
        if dominant_l not in selected or score > selected[dominant_l][0]:
            selected[dominant_l] = (score, int(target.auxiliary_channel[row]))
    for l in (0, 1):
        if l not in selected:
            raise RuntimeError(f"no representative l={l} reference found")
    return {l: value[1] for l, value in selected.items()}


def reference_radial(
    target,
    row,
    radius,
    relative_rank_tolerance,
):
    represented = represented_by_block(target, row, relative_rank_tolerance)
    block, projection = max(represented, key=lambda value: value[1].represented_norm)
    primitives = primitive_radials(block.l, block.n_primitive, radius)
    radial = primitives @ projection.coefficients
    capture = projection.represented_norm / float(target.norm[row])
    return block, radial / math.sqrt(float(target.norm[row])), capture


def plot_orbitals(radius, initial, optimized, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True)
    panels = (
        (axes[0, 0], ((0, 0), (0, 1)), "Fixed DZP s orbitals"),
        (axes[0, 1], ((1, 0),), "Fixed DZP p orbital"),
        (axes[1, 0], ((0, 2),), "Trainable TZDP-extra 3s"),
        (axes[1, 1], ((1, 1),), "Trainable TZDP-extra 2p"),
    )
    labels = {(0, 0): "1s", (0, 1): "2s", (0, 2): "3s", (1, 0): "1p", (1, 1): "2p"}
    colors = {0: "#2369a8", 1: "#d17c0f", 2: "#2f8f5b"}
    for axis, keys, title in panels:
        for key in keys:
            old = initial[key]
            new = align_radial(old, optimized[key], radius)
            color = colors[key[1]]
            axis.plot(radius, old, color=color, linewidth=2.0, label=f"{labels[key]} initial")
            axis.plot(radius, new, color=color, linewidth=1.8, linestyle="--", label=f"{labels[key]} optimized")
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_title(title)
        axis.set_xlim(radius[0], radius[-1])
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.18)
    axes[0, 0].set_ylabel(r"Radial function $R_l(r)$")
    axes[1, 0].set_ylabel(r"Radial function $R_l(r)$")
    axes[1, 0].set_xlabel(r"Radius $r$ (bohr)")
    axes[1, 1].set_xlabel(r"Radius $r$ (bohr)")
    figure.suptitle("H TZDP: fixed DZP and optimized extra orbitals", fontsize=14)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_reference_and_modes(
    target,
    radius,
    initial_orbitals,
    optimized_orbitals,
    spectra,
    mode_overlaps,
    rank1_floor,
    output,
    relative_rank_tolerance,
):
    import matplotlib.pyplot as plt

    channels = select_representative_channels(target, relative_rank_tolerance)
    frequencies = np.unique(target.frequency_ha)
    frequency_indices = (0, 5, 10)
    colors = ("#2369a8", "#d17c0f", "#2f8f5b")
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharex="col")

    reference_rows = {}
    for l, axis in enumerate(axes[0]):
        channel = channels[l]
        rows_for_plot = []
        for frequency_index, color in zip(frequency_indices, colors):
            frequency = float(frequencies[frequency_index])
            candidates = np.flatnonzero(
                (target.auxiliary_channel == channel)
                & np.isclose(target.frequency_ha, frequency)
            )
            if len(candidates) != 1:
                raise RuntimeError("reference row lookup is not unique")
            row = int(candidates[0])
            block, radial, capture = reference_radial(
                target, row, radius, relative_rank_tolerance
            )
            if block.l != l:
                raise RuntimeError("representative channel changed angular momentum")
            label = rf"$\omega={frequency:.3g}$ Ha, {100.0 * capture:.1f}%"
            axis.plot(radius, radial.real, color=color, linewidth=1.9, label=label + " Re")
            axis.plot(radius, radial.imag, color=color, linewidth=1.4, linestyle="--", label=label + " Im")
            rows_for_plot.append(
                {
                    "row": row,
                    "frequency_ha": frequency,
                    "capture": capture,
                    "m": block.m,
                }
            )
        reference_rows[l] = rows_for_plot
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_title(f"Projected reference first-order wavefunction: {'s' if l == 0 else 'p'}")
        axis.set_ylabel(r"$\widetilde{R}_{\delta\psi}(r)/\|\delta\psi\|$")
        axis.legend(frameon=False, fontsize=7, ncol=2)
        axis.grid(alpha=0.18)

    for l, axis in enumerate(axes[1]):
        key = (l, {0: 2, 1: 1}[l])
        old = normalize_radial(initial_orbitals[key], radius)
        new = normalize_radial(optimized_orbitals[key], radius)
        new = align_radial(old, new, radius)
        primitives = primitive_radials(l, spectra[l]["coefficients"].shape[0], radius)
        raw_mode = primitives @ spectra[l]["coefficients"][:, 0]
        mode = output_like_mode(raw_mode, l, radius, optimized_orbitals)
        mode = align_radial(new, mode, radius)
        axis.plot(radius, old, color="#2369a8", linewidth=1.9, label="initial extra orbital")
        axis.plot(radius, new, color="#d17c0f", linewidth=1.9, label="optimized extra orbital")
        axis.plot(radius, mode, color="#2f8f5b", linewidth=2.0, linestyle="--", label="leading residual ST mode")
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        old_overlap = mode_overlaps[l]["initial"]
        new_overlap = mode_overlaps[l]["optimized"]
        axis.set_title(
            f"{'3s' if l == 0 else '2p'} vs leading residual mode: "
            f"{old_overlap:.3f} -> {new_overlap:.3f}"
        )
        axis.set_xlabel(r"Radius $r$ (bohr)")
        axis.set_ylabel(r"Normalized radial function $R_l(r)$")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.18)

    figure.suptitle(
        "What the Sternheimer target asks for\n"
        f"rank-1 s+p ST-only lower bound = {rank1_floor:.6f}",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return reference_rows


def projected_mode_overlap(coefficient, spectrum):
    mode = spectrum["coefficients"][:, 0]
    overlap = spectrum["overlap"]
    numerator = abs(coefficient.T @ overlap @ mode)
    denominator = math.sqrt(
        float(coefficient.T @ overlap @ coefficient)
        * float(mode.T @ overlap @ mode)
    )
    return float(numerator / denominator)


def parse_args():
    script = Path(__file__).resolve()
    repository = script.parents[3]
    project = script.parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-orbital",
        type=Path,
        default=repository
        / "Dojo-NC-SR/Orbitals_v2.0/H_TZDP/H_gga_8au_100Ry_3s2p.orb",
    )
    parser.add_argument(
        "--initial-coefficients",
        type=Path,
        default=repository
        / "Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/ORBITAL_RESULTS.txt",
    )
    parser.add_argument(
        "--optimized-orbital",
        type=Path,
        default=project / "results/siab_h_joint_d41f975e_21315288/ORBITAL_1U.dat",
    )
    parser.add_argument(
        "--optimized-coefficients",
        type=Path,
        default=project
        / "results/siab_h_joint_d41f975e_21315288/ORBITAL_RESULTS.txt",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=project
        / "results/siab_h_option1_20260719/producer_21311439/sternheimer_matrix.dat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "results/fixed_dzp_tzdp_orbital_analysis",
    )
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (
        args.initial_orbital,
        args.initial_coefficients,
        args.optimized_orbital,
        args.optimized_coefficients,
        args.target,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()
    optimizer_dir = script.parents[2] / "opt_orb_pytorch_dpsi"
    target = load_target(args.target, optimizer_dir)
    radius_initial, initial_orbitals = read_orbitals(args.initial_orbital)
    radius_optimized, optimized_orbitals = read_orbitals(args.optimized_orbital)
    np.testing.assert_allclose(radius_initial, radius_optimized, rtol=0.0, atol=1.0e-14)
    radius = radius_initial
    initial_coefficients = read_coefficients(args.initial_coefficients)
    optimized_coefficients = read_coefficients(args.optimized_coefficients)

    context = fixed_context(target, initial_coefficients)
    initial_loss = evaluate_extra_orbitals(target, initial_coefficients, context)
    optimized_loss = evaluate_extra_orbitals(target, optimized_coefficients, context)
    spectra = {
        l: radial_residual_spectrum(
            target,
            initial_coefficients,
            context,
            l,
            args.relative_rank_tolerance,
        )
        for l in (0, 1)
    }
    rank1_floor = 1.0 - sum(
        float(spectra[l]["eigenvalues"][0]) for l in (0, 1)
    ) / context["weighted_norm"]
    full_sp_primitive_floor = 1.0 - sum(
        float(np.sum(spectra[l]["eigenvalues"])) for l in (0, 1)
    ) / context["weighted_norm"]

    fixed_max_difference = {}
    for key in ((0, 0), (0, 1), (1, 0)):
        fixed_max_difference[f"l{key[0]}_zeta{key[1] + 1}"] = float(
            np.max(np.abs(initial_orbitals[key] - optimized_orbitals[key]))
        )

    mode_overlaps = {}
    for l, zeta in ((0, 2), (1, 1)):
        mode_overlaps[l] = {
            "initial": projected_mode_overlap(
                initial_coefficients[("H", l)][:, zeta], spectra[l]
            ),
            "optimized": projected_mode_overlap(
                optimized_coefficients[("H", l)][:, zeta], spectra[l]
            ),
        }

    orbital_stem = args.output / "fixed_dzp_tzdp_orbitals"
    response_stem = args.output / "sternheimer_reference_and_residual_modes"
    plot_orbitals(radius, initial_orbitals, optimized_orbitals, orbital_stem)
    reference_rows = plot_reference_and_modes(
        target,
        radius,
        initial_orbitals,
        optimized_orbitals,
        spectra,
        mode_overlaps,
        rank1_floor,
        response_stem,
        args.relative_rank_tolerance,
    )

    summary = {
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in (
                ("initial_orbital", args.initial_orbital),
                ("initial_coefficients", args.initial_coefficients),
                ("optimized_orbital", args.optimized_orbital),
                ("optimized_coefficients", args.optimized_coefficients),
                ("sternheimer_target", args.target),
            )
        },
        "relative_rank_tolerance": args.relative_rank_tolerance,
        "sternheimer_loss": {
            "initial_3s2p": initial_loss,
            "optimized_3s2p": optimized_loss,
            "same_size_st_only_rank1_floor": rank1_floor,
            "remaining_same_size_st_only_headroom": optimized_loss - rank1_floor,
            "full_sp_primitive_floor": full_sp_primitive_floor,
        },
        "mode_overlap": {
            "s": mode_overlaps[0],
            "p": mode_overlaps[1],
        },
        "fixed_dzp_max_abs_difference": fixed_max_difference,
        "radial_nodes": {
            "initial_3s": _nodes(initial_orbitals[(0, 2)]),
            "optimized_3s": _nodes(optimized_orbitals[(0, 2)]),
            "initial_2p": _nodes(initial_orbitals[(1, 1)]),
            "optimized_2p": _nodes(optimized_orbitals[(1, 1)]),
        },
        "representative_reference_rows": {
            "s": reference_rows[0],
            "p": reference_rows[1],
        },
        "figures": {
            "orbitals_png": str(orbital_stem.with_suffix(".png").resolve()),
            "response_png": str(response_stem.with_suffix(".png").resolve()),
        },
    }
    summary_path = args.output / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["sternheimer_loss"], indent=2, sort_keys=True))
    print(json.dumps(summary["mode_overlap"], indent=2, sort_keys=True))
    print(summary_path)


if __name__ == "__main__":
    main()
