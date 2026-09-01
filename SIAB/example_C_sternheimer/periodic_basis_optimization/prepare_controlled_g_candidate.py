#!/usr/bin/env python3
"""Append one lowest-kinetic g shell to an immutable periodic C DZP candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SIAB_ROOT = HERE.parents[1]
sys.path.insert(0, str(SIAB_ROOT / "opt_orb_pytorch_dpsi"))

from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)
from export_periodic_orbitals import (  # noqa: E402
    _simpson,
    build_radial_orbitals,
    spherical_bessel_roots,
    write_abacus_orbital,
)


BASE_NU = (3, 3, 2, 0, 0)
TARGET_NU = (3, 3, 2, 0, 1)
RADIAL_ROWS = 31
ECUT_RY = 100.0
RCUT_BOHR = 10.0
DR_BOHR = 0.01
SMOOTHING_SIGMA_BOHR = 0.1
AO_COUNT_ATOM = 31
ANGULAR_MOMENTUM = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _count_interior_nodes(radius: np.ndarray, values: np.ndarray) -> int:
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    if radius.shape != values.shape:
        raise ValueError("radius and radial values must have the same shape")
    threshold = max(float(np.max(np.abs(values))) * 1.0e-6, 1.0e-14)
    resolved = (radius >= 0.1) & (radius < radius[-1]) & (np.abs(values) > threshold)
    signs = np.sign(values[resolved])
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _radial_diagnostics(radius: np.ndarray, radial: np.ndarray) -> dict:
    probability = radial * radial * radius * radius
    norm = float(_simpson(probability, DR_BOHR))
    roots = spherical_bessel_roots(ANGULAR_MOMENTUM, RADIAL_ROWS)
    derivative = np.gradient(radial, DR_BOHR)
    numerical_kinetic_ry = float(
        _simpson(
            derivative * derivative * radius * radius
            + ANGULAR_MOMENTUM * (ANGULAR_MOMENTUM + 1.0) * radial * radial,
            DR_BOHR,
        )
    )
    tail = probability.copy()
    tail[radius < 9.0] = 0.0
    return {
        "angular_momentum": ANGULAR_MOMENTUM,
        "first_bessel_root": float(roots[0]),
        "interior_node_count": _count_interior_nodes(radius, radial),
        "kinetic_energy_ry": float((roots[0] / RCUT_BOHR) ** 2),
        "mean_radius_bohr": float(_simpson(probability * radius, DR_BOHR)),
        "mean_square_radius_bohr2": float(
            _simpson(probability * radius * radius, DR_BOHR)
        ),
        "numerical_kinetic_energy_ry": numerical_kinetic_ry,
        "radial_norm": norm,
        "tail_probability_r_ge_9_bohr": float(_simpson(tail, DR_BOHR)),
    }


def _amplitude_token(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    magnitude = f"{abs(value):.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return sign + (magnitude or "0")


def prepare_candidate(
    *,
    source: Path,
    root: Path,
    second_primitive_amplitude: float = 0.0,
    optimized_g_source: Path | None = None,
    optimized_g_max_primitives: int | None = None,
) -> dict:
    source = Path(source).resolve(strict=True)
    root = Path(root).resolve()
    second_primitive_amplitude = float(second_primitive_amplitude)
    if not math.isfinite(second_primitive_amplitude):
        raise ValueError("second primitive amplitude must be finite")
    if optimized_g_source is not None:
        optimized_g_source = Path(optimized_g_source).resolve(strict=True)
        if second_primitive_amplitude != 0.0:
            raise ValueError(
                "optimized g source and second primitive amplitude are mutually exclusive"
            )
    if optimized_g_max_primitives is not None:
        if optimized_g_source is None:
            raise ValueError("optimized g lowpass requires an optimized g source")
        if not 1 <= optimized_g_max_primitives <= RADIAL_ROWS:
            raise ValueError("optimized g max primitives must be within [1, 31]")
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if not root.parent.is_dir():
        raise ValueError("candidate parent directory does not exist")

    base = read_periodic_optimizer_coefficients(
        source,
        element="C",
        radial_rows=RADIAL_ROWS,
        expected_nu=BASE_NU,
    )
    coefficients = {"C": [channel.detach().clone() for channel in base["C"]]}
    if optimized_g_source is None:
        g_seed = torch.zeros(RADIAL_ROWS, 1, dtype=torch.float64)
        g_seed[0, 0] = 1.0
        g_seed[1, 0] = second_primitive_amplitude
    else:
        optimized = read_periodic_optimizer_coefficients(
            optimized_g_source,
            element="C",
            radial_rows=RADIAL_ROWS,
            expected_nu=(3, 3, 2, 1, 1),
        )
        g_seed = optimized["C"][4].detach().clone()
        if tuple(g_seed.shape) != (RADIAL_ROWS, 1):
            raise RuntimeError("optimized g source does not contain exactly one g radial")
        if optimized_g_max_primitives is not None:
            g_seed[optimized_g_max_primitives:, 0] = 0.0
            g_seed /= torch.linalg.norm(g_seed)
    coefficients["C"][4] = g_seed

    radius, orbitals = build_radial_orbitals(
        coefficients,
        element="C",
        ecut_ry=ECUT_RY,
        rcut_bohr=RCUT_BOHR,
        dr_bohr=DR_BOHR,
        smoothing_sigma_bohr=SMOOTHING_SIGMA_BOHR,
    )
    diagnostics = _radial_diagnostics(radius, orbitals[4][:, 0])
    if diagnostics["interior_node_count"] != 0:
        raise RuntimeError("controlled g radial contains an interior radial node")
    if abs(diagnostics["radial_norm"] - 1.0) > 1.0e-10:
        raise RuntimeError("controlled g radial normalization failed")
    if not all(
        math.isfinite(value)
        for value in diagnostics.values()
        if isinstance(value, float)
    ):
        raise RuntimeError("controlled g radial diagnostics are not finite")

    temporary = Path(tempfile.mkdtemp(prefix=root.name + ".tmp-", dir=root.parent))
    try:
        contracted = second_primitive_amplitude != 0.0
        optimized_profile = optimized_g_source is not None
        lowpass_profile = optimized_g_max_primitives is not None
        token = _amplitude_token(second_primitive_amplitude)
        if lowpass_profile:
            suffix = f"joint_atom_solid_optimized_g_lowpass_n{optimized_g_max_primitives}"
        elif optimized_profile:
            suffix = "joint_atom_solid_optimized_g"
        else:
            suffix = f"contracted_l4_bessel_{token}" if contracted else "lowest_l4_bessel"
        coefficient_name = f"C_3s3p2d1g_{suffix}.txt"
        orbital_name = f"C_gga_10au_100Ry_3s3p2d1g_{suffix}.orb"
        coefficient_path = temporary / coefficient_name
        orbital_path = temporary / orbital_name
        write_periodic_optimizer_coefficients(coefficient_path, coefficients)
        write_abacus_orbital(
            orbital_path,
            coefficients,
            element="C",
            ecut_ry=ECUT_RY,
            rcut_bohr=RCUT_BOHR,
            dr_bohr=DR_BOHR,
            smoothing_sigma_bohr=SMOOTHING_SIGMA_BOHR,
        )
        restored = read_periodic_optimizer_coefficients(
            coefficient_path,
            element="C",
            radial_rows=RADIAL_ROWS,
            expected_nu=TARGET_NU,
        )
        for actual, expected in zip(restored["C"][:3], base["C"][:3]):
            if not torch.equal(actual, expected):
                raise RuntimeError("controlled g export changed the DZP prefix")

        payload = {
            "status": "success",
            "profile": (
                "controlled_optimized_g_lowpass"
                if lowpass_profile
                else "controlled_optimized_g"
                if optimized_profile
                else "controlled_contracted_g" if contracted else "controlled_lowest_g"
            ),
            "nu": list(TARGET_NU),
            "ao_count_atom": AO_COUNT_ATOM,
            "seed_definition": (
                "joint_atom_solid_optimized_g_lowpass"
                if lowpass_profile
                else "joint_atom_solid_optimized_g_only"
                if optimized_profile
                else "lowest_l4_bessel_plus_scaled_second_primitive"
                if contracted
                else "lowest_l4_spherical_bessel_primitive"
            ),
            "second_primitive_amplitude": second_primitive_amplitude,
            "base_coefficients": str(source),
            "base_coefficients_sha256": sha256(source),
            "coefficients_filename": coefficient_name,
            "coefficients_sha256": sha256(coefficient_path),
            "orbital_filename": orbital_name,
            "orbital_sha256": sha256(orbital_path),
            "g_diagnostics": diagnostics,
        }
        if optimized_profile:
            payload.update(
                {
                    "optimized_g_source": str(optimized_g_source),
                    "optimized_g_source_sha256": sha256(optimized_g_source),
                    "optimized_g_max_primitives": optimized_g_max_primitives,
                }
            )
        manifest = temporary / "CANDIDATE.json"
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        (temporary / "provenance.txt").write_text(
            "status=success\n"
            f"purpose={payload['profile']}_candidate\n"
            f"base_coefficients_sha256={payload['base_coefficients_sha256']}\n"
            + (
                f"optimized_g_source_sha256={payload['optimized_g_source_sha256']}\n"
                if optimized_profile
                else ""
            )
            + f"candidate_orbital_sha256={payload['orbital_sha256']}\n",
            encoding="ascii",
        )
        (temporary / "STATUS").write_text("success\n", encoding="ascii")
        os.replace(temporary, root)
    except Exception:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--second-primitive-amplitude", type=float, default=0.0)
    parser.add_argument("--optimized-g-source", type=Path)
    parser.add_argument("--optimized-g-max-primitives", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_candidate(
                source=args.source,
                root=args.root,
                second_primitive_amplitude=args.second_primitive_amplitude,
                optimized_g_source=args.optimized_g_source,
                optimized_g_max_primitives=args.optimized_g_max_primitives,
            ),
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
