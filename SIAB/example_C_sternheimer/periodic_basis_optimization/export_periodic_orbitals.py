#!/usr/bin/env python3
"""Export periodic Galerkin Bessel coefficients as an ABACUS orbital file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from periodic_galerkin_basis import read_periodic_optimizer_coefficients  # noqa: E402


def parse_nu(value):
    try:
        counts = tuple(int(field.strip()) for field in value.split(","))
    except ValueError as error:
        raise ValueError("nu must be a comma-separated integer list") from error
    if not counts or any(count < 0 for count in counts) or not any(counts):
        raise ValueError("nu must contain nonnegative counts and be nonempty")
    return counts


def validate_bessel_contract(*, radial_rows, ecut_ry, rcut_bohr):
    if type(radial_rows) is not int or radial_rows <= 0:
        raise ValueError("radial_rows must be a positive integer")
    if not math.isfinite(ecut_ry) or not math.isfinite(rcut_bohr):
        raise ValueError("Bessel cutoff parameters must be finite")
    if ecut_ry <= 0.0 or rcut_bohr <= 0.0:
        raise ValueError("Bessel cutoff parameters must be positive")
    expected = int(math.sqrt(ecut_ry) * rcut_bohr / math.pi)
    if expected != radial_rows:
        raise ValueError(
            "coefficient primitive count {} does not match the ABACUS Bessel "
            "contract {} from ecut={} Ry and rcut={} Bohr".format(
                radial_rows, expected, ecut_ry, rcut_bohr
            )
        )
    return expected


def _odd_double_factorial(value):
    result = 1.0
    for factor in range(1, value + 1, 2):
        result *= factor
    return result


def spherical_bessel_j(l, x):
    if type(l) is not int or l < 0:
        raise ValueError("angular momentum must be a nonnegative integer")
    values = np.asarray(x, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("spherical Bessel arguments must be finite")
    flat = values.reshape(-1)
    result = np.empty_like(flat)
    small = np.abs(flat) < 1.0e-7
    if np.any(small):
        xs = flat[small]
        leading = np.power(xs, l) / _odd_double_factorial(2 * l + 1)
        result[small] = leading * (1.0 - xs * xs / (2.0 * (2 * l + 3)))
    if np.any(~small):
        xn = flat[~small]
        j0 = np.sin(xn) / xn
        if l == 0:
            result[~small] = j0
        else:
            j1 = np.sin(xn) / (xn * xn) - np.cos(xn) / xn
            if l == 1:
                result[~small] = j1
            else:
                previous, current = j0, j1
                for order in range(1, l):
                    following = (2 * order + 1) / xn * current - previous
                    previous, current = current, following
                result[~small] = current
    reshaped = result.reshape(values.shape)
    return float(reshaped) if reshaped.ndim == 0 else reshaped


def spherical_bessel_roots(l, count):
    if type(count) is not int or count <= 0:
        raise ValueError("root count must be a positive integer")
    roots = []
    step = math.pi / 8.0
    left = 1.0e-7
    fleft = spherical_bessel_j(l, left)
    maximum = (count + 0.5 * l + 3.0) * math.pi
    while left < maximum and len(roots) < count:
        right = left + step
        fright = spherical_bessel_j(l, right)
        if fleft * fright < 0.0:
            lower, upper = left, right
            flower = fleft
            for _ in range(80):
                middle = 0.5 * (lower + upper)
                fmiddle = spherical_bessel_j(l, middle)
                if flower * fmiddle <= 0.0:
                    upper = middle
                else:
                    lower = middle
                    flower = fmiddle
            roots.append(0.5 * (lower + upper))
        left, fleft = right, fright
    if len(roots) != count:
        raise RuntimeError("failed to locate the requested spherical Bessel roots")
    return np.asarray(roots, dtype=float)


def _simpson(values, dx):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 3 or values.size % 2 != 1:
        raise ValueError("radial integration requires an odd mesh with at least three points")
    return dx / 3.0 * (
        values[0]
        + values[-1]
        + 4.0 * np.sum(values[1:-1:2])
        + 2.0 * np.sum(values[2:-1:2])
    )


def _orthonormalize_radial(radial, radius, dr_bohr):
    output = np.asarray(radial, dtype=float).copy()
    if output.ndim != 2 or output.shape[0] != radius.size:
        raise ValueError("radial orbitals have invalid dimensions")
    for column in range(output.shape[1]):
        for previous in range(column):
            overlap = _simpson(
                output[:, column] * output[:, previous] * radius * radius,
                dr_bohr,
            )
            output[:, column] -= overlap * output[:, previous]
        norm2 = _simpson(output[:, column] ** 2 * radius * radius, dr_bohr)
        if not math.isfinite(norm2) or norm2 <= 1.0e-20:
            raise ValueError("candidate radial orbitals are linearly dependent")
        output[:, column] /= math.sqrt(norm2)
    return output


def build_radial_orbitals(
    coefficients,
    *,
    element,
    ecut_ry,
    rcut_bohr,
    dr_bohr,
    smoothing_sigma_bohr,
):
    if set(coefficients) != {element}:
        raise ValueError("coefficient elements do not match the requested element")
    channels = list(coefficients[element])
    while channels and channels[-1].shape[1] == 0:
        channels.pop()
    if not channels:
        raise ValueError("coefficient file defines no active angular channel")
    radial_rows = int(channels[0].shape[0])
    validate_bessel_contract(
        radial_rows=radial_rows,
        ecut_ry=ecut_ry,
        rcut_bohr=rcut_bohr,
    )
    if not math.isfinite(dr_bohr) or dr_bohr <= 0.0:
        raise ValueError("dr must be finite and positive")
    if not math.isfinite(smoothing_sigma_bohr) or smoothing_sigma_bohr <= 0.0:
        raise ValueError("smoothing sigma must be finite and positive")
    intervals = rcut_bohr / dr_bohr
    rounded_intervals = round(intervals)
    if abs(intervals - rounded_intervals) > 1.0e-10 or rounded_intervals % 2 != 0:
        raise ValueError("rcut/dr must be an even integer for Simpson integration")
    radius = np.arange(rounded_intervals + 1, dtype=float) * dr_bohr
    smoothing = 1.0 - np.exp(
        -((radius - rcut_bohr) ** 2)
        / (2.0 * smoothing_sigma_bohr * smoothing_sigma_bohr)
    )
    orbitals = []
    for l, channel in enumerate(channels):
        if channel.ndim != 2 or channel.shape[0] != radial_rows:
            raise ValueError("coefficient channels do not share one primitive count")
        values = channel.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("coefficient channels must be finite")
        roots = spherical_bessel_roots(l, radial_rows)
        primitives = spherical_bessel_j(
            l,
            radius[:, None] * roots[None, :] / rcut_bohr,
        ) * smoothing[:, None]
        radial = primitives @ values
        orbitals.append(_orthonormalize_radial(radial, radius, dr_bohr))
    return radius, tuple(orbitals)


def write_abacus_orbital(
    path,
    coefficients,
    *,
    element,
    ecut_ry,
    rcut_bohr,
    dr_bohr,
    smoothing_sigma_bohr,
):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise ValueError("orbital output parent directory does not exist")
    radius, orbitals = build_radial_orbitals(
        coefficients,
        element=element,
        ecut_ry=ecut_ry,
        rcut_bohr=rcut_bohr,
        dr_bohr=dr_bohr,
        smoothing_sigma_bohr=smoothing_sigma_bohr,
    )
    names = ["S", "P", "D"] + [chr(code) for code in range(ord("F"), ord("Z") + 1)]
    lines = [
        "---------------------------------------------------------------------------",
        "Element                     {}".format(element),
        "Energy Cutoff(Ry)           {}".format(ecut_ry),
        "Radius Cutoff(a.u.)         {}".format(rcut_bohr),
        "Lmax                        {}".format(len(orbitals) - 1),
    ]
    for l, radial in enumerate(orbitals):
        lines.append("Number of {}orbital-->       {}".format(names[l], radial.shape[1]))
    lines.extend(
        (
            "---------------------------------------------------------------------------",
            "SUMMARY  END",
            "",
            "Mesh                        {}".format(radius.size),
            "dr                          {}".format(dr_bohr),
        )
    )
    for l, radial in enumerate(orbitals):
        for zeta in range(radial.shape[1]):
            lines.extend(
                (
                    "                Type                   L                   N",
                    "                   0                   {}                   {}".format(l, zeta),
                )
            )
            values = radial[:, zeta]
            for start in range(0, values.size, 4):
                lines.append("  ".join("{:.14e}".format(value) for value in values[start:start + 4]))
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "element": element,
        "ecut_ry": ecut_ry,
        "rcut_bohr": rcut_bohr,
        "dr_bohr": dr_bohr,
        "smoothing_sigma_bohr": smoothing_sigma_bohr,
        "mesh": int(radius.size),
        "nu": [int(radial.shape[1]) for radial in orbitals],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--nu", required=True)
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--ecut-ry", type=float, default=100.0)
    parser.add_argument("--rcut-bohr", type=float, default=10.0)
    parser.add_argument("--dr-bohr", type=float, default=0.01)
    parser.add_argument("--smoothing-sigma-bohr", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    nu = parse_nu(args.nu)
    coefficients = read_periodic_optimizer_coefficients(
        args.input,
        element=args.element,
        radial_rows=args.radial_rows,
        expected_nu=nu,
    )
    metadata = write_abacus_orbital(
        args.output,
        coefficients,
        element=args.element,
        ecut_ry=args.ecut_ry,
        rcut_bohr=args.rcut_bohr,
        dr_bohr=args.dr_bohr,
        smoothing_sigma_bohr=args.smoothing_sigma_bohr,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
