#!/usr/bin/env python3
"""Export optimized H response coefficients as an ABACUS numerical orbital."""

import argparse
import hashlib
import json
import pathlib
import sys
import types

import numpy as np


SIAB_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPT_DIR = SIAB_ROOT / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

import orbital as siab_orbital
from IO.func_C import read_C_init
from IO.print_orbital import print_orbital_head


def _sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _radial_norm(radial, dr):
    return float(siab_orbital.inner_product(radial, radial, dr))


def _channel_validation(raw_channel, orthogonal_channel, serialized_channel, dr, l):
    count = len(raw_channel)
    overlap = np.empty((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(count):
            overlap[left, right] = siab_orbital.inner_product(
                orthogonal_channel[left], orthogonal_channel[right], dr
            )

    relative_residuals = []
    for raw in raw_channel:
        reconstructed = np.zeros_like(raw)
        for orthogonal in orthogonal_channel:
            coefficient = siab_orbital.inner_product(orthogonal, raw, dr)
            reconstructed += coefficient * orthogonal
        residual = raw - reconstructed
        relative_residuals.append(
            np.sqrt(max(_radial_norm(residual, dr), 0.0) / _radial_norm(raw, dr))
        )

    return {
        "l": int(l),
        "radial_count": count,
        "orthonormality_max_abs_error": float(
            np.max(np.abs(overlap - np.eye(count)))
        ),
        "maximum_relative_span_residual": float(max(relative_residuals)),
        "serialized_max_abs_error": float(
            max(
                np.max(np.abs(expected - actual))
                for expected, actual in zip(
                    orthogonal_channel, serialized_channel, strict=True
                )
            )
        ),
    }


def _write_orbital(path, orbitals, info_radial):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        print_orbital_head(stream, orbitals["H"], info_radial, "H")
        for l, channel in enumerate(orbitals["H"]):
            for zeta, radial in enumerate(channel):
                print("                Type                   L                   N", file=stream)
                print(f"                   0                   {l}                   {zeta}", file=stream)
                for index, value in enumerate(radial):
                    print(f"{value:.14e}", end="  ", file=stream)
                    if index % 4 == 3:
                        print(file=stream)
                print(file=stream)


def _read_serialized_orbitals(path, nu, mesh):
    lines = pathlib.Path(path).read_text(encoding="ascii").splitlines()
    orbitals = [[] for _ in nu]
    index = 0
    while index < len(lines):
        if lines[index].strip() != "Type                   L                   N":
            index += 1
            continue
        if index + 1 >= len(lines):
            raise ValueError("serialized orbital has an incomplete Type block")
        label = lines[index + 1].split()
        if len(label) != 3:
            raise ValueError("serialized orbital has an invalid Type label")
        _, l_text, zeta_text = label
        l = int(l_text)
        zeta = int(zeta_text)
        index += 2
        values = []
        while index < len(lines) and len(values) < mesh:
            values.extend(float(value) for value in lines[index].split())
            index += 1
        if len(values) != mesh:
            raise ValueError("serialized orbital radial block has the wrong mesh")
        if l < 0 or l >= len(nu) or zeta != len(orbitals[l]):
            raise ValueError("serialized orbital Type blocks are out of order")
        orbitals[l].append(np.asarray(values, dtype=np.float64))
    if [len(channel) for channel in orbitals] != list(nu):
        raise ValueError("serialized orbital radial counts differ from Nu")
    return orbitals


def export_h_response_orbital(
    coefficient_file,
    orbital_path,
    manifest_path,
    *,
    nu=(3, 3, 2),
    siab_commit,
    optimization_json=None,
    rcut=8.0,
    dr=0.01,
    ecut=100,
    primitive_count=25,
):
    """Export one H basis while preserving every per-l primitive span."""
    nu = tuple(int(value) for value in nu)
    if not nu or any(value <= 0 for value in nu):
        raise ValueError("Nu must contain positive radial counts")
    if not siab_commit:
        raise ValueError("siab_commit must be nonempty")

    info_element = {
        "H": types.SimpleNamespace(
            Nl=len(nu),
            Ne=int(primitive_count),
            Nu=list(nu),
        )
    }
    info_radial = {
        "Rcut": {"H": float(rcut)},
        "dr": {"H": float(dr)},
        "Ecut": {"H": int(ecut)},
        "smearing_sigma": {"H": 0.0},
    }
    coefficients, initialization = read_C_init(
        coefficient_file, info_element, return_metadata=True
    )
    if initialization.appended_indices:
        raise ValueError("coefficient file does not define every requested AO")

    eigenvalues = siab_orbital.set_E(info_element, info_radial["Rcut"])
    raw = siab_orbital.generate_orbital(
        info_element, info_radial, coefficients, eigenvalues
    )
    exported = {
        "H": [
            [radial.copy() for radial in channel]
            for channel in raw["H"]
        ]
    }
    siab_orbital.orth(exported["H"], info_radial["dr"]["H"])
    _write_orbital(orbital_path, exported, info_radial)

    mesh = int(rcut / dr) + 1
    serialized = _read_serialized_orbitals(orbital_path, nu, mesh)
    validation = [
        _channel_validation(
            raw["H"][l], exported["H"][l], serialized[l], dr, l
        )
        for l in range(len(nu))
    ]
    if any(
        channel["orthonormality_max_abs_error"] > 1.0e-10
        or channel["maximum_relative_span_residual"] > 1.0e-10
        or channel["serialized_max_abs_error"] > 1.0e-12
        for channel in validation
    ):
        raise ValueError("exported orbital failed the radial subspace validation")

    optimization_path = (
        pathlib.Path(optimization_json) if optimization_json is not None else None
    )
    manifest = {
        "schema_version": 1,
        "method": "raw_bessel_then_per_l_gram_schmidt",
        "element": "H",
        "source_siab_commit": siab_commit,
        "source_coefficients": str(pathlib.Path(coefficient_file).resolve()),
        "source_coefficients_sha256": _sha256(coefficient_file),
        "source_optimization": (
            str(optimization_path.resolve()) if optimization_path is not None else None
        ),
        "source_optimization_sha256": (
            _sha256(optimization_path) if optimization_path is not None else None
        ),
        "radial_orbitals_by_l": list(nu),
        "candidate_ao_dimension": int(
            sum((2 * l + 1) * count for l, count in enumerate(nu))
        ),
        "primitive_count_per_l": int(primitive_count),
        "radial": {
            "rcut_bohr": float(rcut),
            "dr_bohr": float(dr),
            "mesh": mesh,
            "ecut_ry": int(ecut),
            "smearing_applied": False,
        },
        "subspace_validation": validation,
        "orbital_path": str(pathlib.Path(orbital_path).resolve()),
        "orbital_sha256": _sha256(orbital_path),
    }
    manifest_path = pathlib.Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coefficient_file", type=pathlib.Path)
    parser.add_argument("orbital_path", type=pathlib.Path)
    parser.add_argument("manifest_path", type=pathlib.Path)
    parser.add_argument("--nu", type=int, nargs="+", default=(3, 3, 2))
    parser.add_argument("--siab-commit", required=True)
    parser.add_argument("--optimization-json", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = export_h_response_orbital(
        args.coefficient_file,
        args.orbital_path,
        args.manifest_path,
        nu=args.nu,
        siab_commit=args.siab_commit,
        optimization_json=args.optimization_json,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
