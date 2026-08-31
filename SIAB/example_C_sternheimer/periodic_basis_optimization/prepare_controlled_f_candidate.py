#!/usr/bin/env python3
"""Append one lowest-kinetic f shell to an immutable periodic C DZP candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
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
TARGET_NU = (3, 3, 2, 1, 0)
RADIAL_ROWS = 31
ECUT_RY = 100.0
RCUT_BOHR = 10.0
DR_BOHR = 0.01
SMOOTHING_SIGMA_BOHR = 0.1
AO_COUNT_ATOM = 29


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _count_interior_nodes(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float)
    # The upward recurrence used by the common exporter has roundoff at the
    # first two l=3 mesh points.  Count physical nodes only above one ppm of
    # the radial maximum.
    threshold = max(float(np.max(np.abs(values))) * 1.0e-6, 1.0e-14)
    signs = np.sign(values[1:-1][np.abs(values[1:-1]) > threshold])
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _radial_diagnostics(
    radius: np.ndarray,
    radial: np.ndarray,
    seed_coefficients: np.ndarray,
) -> dict:
    probability = radial * radial * radius * radius
    norm = float(_simpson(probability, DR_BOHR))
    mean_radius = float(_simpson(probability * radius, DR_BOHR))
    mean_radius2 = float(_simpson(probability * radius * radius, DR_BOHR))
    tail = probability.copy()
    tail[radius < 9.0] = 0.0
    tail_probability = float(_simpson(tail, DR_BOHR))
    roots = spherical_bessel_roots(3, RADIAL_ROWS)
    seed_coefficients = np.asarray(seed_coefficients, dtype=float)
    coefficient_norm2 = float(np.dot(seed_coefficients, seed_coefficients))
    if not math.isfinite(coefficient_norm2) or abs(coefficient_norm2 - 1.0) > 1.0e-12:
        raise ValueError("controlled f seed coefficients must be normalized")
    kinetic_ry = float(
        np.sum(seed_coefficients * seed_coefficients * (roots / RCUT_BOHR) ** 2)
    )
    derivative = np.gradient(radial, DR_BOHR)
    numerical_kinetic_ry = float(
        _simpson(derivative * derivative * radius * radius + 12.0 * radial * radial, DR_BOHR)
    )
    return {
        "angular_momentum": 3,
        "first_bessel_root": float(roots[0]),
        "interior_node_count": _count_interior_nodes(radial),
        "kinetic_energy_hartree": 0.5 * kinetic_ry,
        "kinetic_energy_ry": kinetic_ry,
        "mean_radius_bohr": mean_radius,
        "mean_square_radius_bohr2": mean_radius2,
        "numerical_kinetic_energy_ry": numerical_kinetic_ry,
        "radial_norm": norm,
        "tail_probability_r_ge_9_bohr": tail_probability,
    }


def prepare_candidate(
    *,
    source: Path,
    root: Path,
    second_primitive_amplitude: float = 0.0,
) -> dict:
    source = Path(source).resolve(strict=True)
    root = Path(root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if not root.parent.is_dir():
        raise ValueError("candidate parent directory does not exist")

    second_primitive_amplitude = float(second_primitive_amplitude)
    if not math.isfinite(second_primitive_amplitude) or abs(second_primitive_amplitude) > 0.5:
        raise ValueError("second primitive amplitude must be finite and within [-0.5, 0.5]")

    base = read_periodic_optimizer_coefficients(
        source,
        element="C",
        radial_rows=RADIAL_ROWS,
        expected_nu=BASE_NU,
    )
    coefficients = {"C": [channel.detach().clone() for channel in base["C"]]}
    f_seed = torch.zeros(RADIAL_ROWS, 1, dtype=torch.float64)
    f_seed[0, 0] = 1.0
    f_seed[1, 0] = second_primitive_amplitude
    f_seed /= torch.linalg.norm(f_seed)
    coefficients["C"][3] = f_seed

    radius, orbitals = build_radial_orbitals(
        coefficients,
        element="C",
        ecut_ry=ECUT_RY,
        rcut_bohr=RCUT_BOHR,
        dr_bohr=DR_BOHR,
        smoothing_sigma_bohr=SMOOTHING_SIGMA_BOHR,
    )
    if len(orbitals) != 4 or orbitals[3].shape[1] != 1:
        raise RuntimeError("controlled f candidate has an unexpected channel layout")
    diagnostics = _radial_diagnostics(
        radius,
        orbitals[3][:, 0],
        f_seed[:, 0].detach().cpu().numpy(),
    )
    if diagnostics["interior_node_count"] != 0:
        raise RuntimeError("lowest f seed unexpectedly contains an interior radial node")
    if abs(diagnostics["radial_norm"] - 1.0) > 1.0e-10:
        raise RuntimeError("controlled f radial normalization failed")
    if not all(math.isfinite(value) for value in diagnostics.values() if isinstance(value, float)):
        raise RuntimeError("controlled f radial diagnostics are not finite")

    temporary = Path(tempfile.mkdtemp(prefix=root.name + ".tmp-", dir=root.parent))
    try:
        if second_primitive_amplitude == 0.0:
            profile = "controlled_lowest_f"
            seed_definition = "lowest_l3_spherical_bessel_primitive"
            suffix = "lowest_l3_bessel"
        else:
            profile = "controlled_contracted_f"
            seed_definition = "normalized_l3_primitives_0_plus_amplitude_times_1"
            suffix = "contracted_l3_a_{:+.3f}".format(second_primitive_amplitude)
        coefficient_name = "C_3s3p2d1f_{}.txt".format(suffix)
        orbital_name = "C_gga_10au_100Ry_3s3p2d1f_{}.orb".format(suffix)
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
        if any(
            not torch.equal(actual, expected)
            for actual, expected in zip(restored["C"][:3], base["C"][:3])
        ):
            raise RuntimeError("controlled f export changed an existing DZP coefficient")

        payload = {
            "ao_count_atom": AO_COUNT_ATOM,
            "base_coefficients": str(source),
            "base_coefficients_sha256": sha256(source),
            "coefficients_filename": coefficient_name,
            "coefficients_sha256": sha256(coefficient_path),
            "f_diagnostics": diagnostics,
            "nu": list(TARGET_NU),
            "orbital_filename": orbital_name,
            "orbital_sha256": sha256(orbital_path),
            "profile": profile,
            "second_primitive_amplitude": second_primitive_amplitude,
            "seed_definition": seed_definition,
            "status": "success",
        }
        manifest = temporary / "CANDIDATE.json"
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        (temporary / "provenance.txt").write_text(
            "status=success\n"
            "purpose=controlled_single_f_physics_candidate\n"
            "base_layout=3s3p2d\n"
            "target_layout=3s3p2d1f\n"
            f"seed_definition={seed_definition}\n"
            f"second_primitive_amplitude={second_primitive_amplitude:.16g}\n"
            "existing_dzp_coefficients_unchanged=yes\n"
            f"base_coefficients_sha256={payload['base_coefficients_sha256']}\n"
            f"selected_coefficients_sha256={payload['coefficients_sha256']}\n"
            f"selected_orbital_sha256={payload['orbital_sha256']}\n"
            f"candidate_manifest_sha256={sha256(manifest)}\n",
            encoding="ascii",
        )
        (temporary / "STATUS").write_text("success\n", encoding="ascii")
        os.replace(temporary, root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--second-primitive-amplitude", type=float, default=0.0)
    args = parser.parse_args()
    result = prepare_candidate(
        source=args.source,
        root=args.root,
        second_primitive_amplitude=args.second_primitive_amplitude,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
