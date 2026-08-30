#!/usr/bin/env python3
"""Stage a full-DZP C candidate along a verified reverse optimization direction."""

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


HERE = Path(__file__).resolve().parent
SIAB_ROOT = HERE.parents[1]
sys.path.insert(0, str(SIAB_ROOT / "opt_orb_pytorch_dpsi"))

from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)
from export_periodic_orbitals import write_abacus_orbital  # noqa: E402


NU = (3, 3, 2, 0, 0)
AO_COUNT_ATOM = 22


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path):
    return read_periodic_optimizer_coefficients(
        Path(path).resolve(strict=True),
        element="C",
        radial_rows=31,
        expected_nu=NU,
    )


def _validated_channel_alphas(alpha, channel_alphas):
    if (alpha is None) == (channel_alphas is None):
        raise ValueError("provide exactly one scalar alpha or channel_alphas")
    if channel_alphas is None:
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha >= 0.0:
            raise ValueError("reverse-search alpha must be finite and negative")
        return alpha, (alpha,) * len(NU)
    values = tuple(float(value) for value in channel_alphas)
    if len(values) != len(NU):
        raise ValueError("channel_alphas must define s,p,d,f,g")
    if any(not math.isfinite(value) or value > 0.0 for value in values):
        raise ValueError("channel reverse-search alphas must be finite and nonpositive")
    if not any(value < 0.0 for value in values):
        raise ValueError("channel_alphas must contain a reverse component")
    return None, values


def prepare_candidate(
    *,
    original: Path,
    optimized: Path,
    root: Path,
    alpha: float | None = None,
    channel_alphas=None,
) -> dict:
    alpha, resolved_channel_alphas = _validated_channel_alphas(
        alpha,
        channel_alphas,
    )
    original = Path(original).resolve(strict=True)
    optimized = Path(optimized).resolve(strict=True)
    root = Path(root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if not root.parent.is_dir():
        raise ValueError("candidate parent directory does not exist")

    initial = _read(original)
    selected = _read(optimized)
    coefficients = {"C": []}
    for channel_alpha, initial_channel, selected_channel in zip(
        resolved_channel_alphas,
        initial["C"],
        selected["C"],
    ):
        if initial_channel.shape != selected_channel.shape:
            raise ValueError("coefficient channel layouts differ")
        coefficients["C"].append(
            initial_channel + channel_alpha * (selected_channel - initial_channel)
        )

    temporary = Path(tempfile.mkdtemp(prefix=root.name + ".tmp-", dir=root.parent))
    try:
        if alpha is None:
            coefficient_name = "C_3s3p2d_reverse_channel_resolved.txt"
            orbital_name = "C_gga_10au_100Ry_3s3p2d_reverse_channel_resolved.orb"
        else:
            coefficient_name = "C_3s3p2d_reverse_alpha_{:+.3f}.txt".format(alpha)
            orbital_name = "C_gga_10au_100Ry_3s3p2d_reverse_alpha_{:+.3f}.orb".format(alpha)
        coefficient_path = temporary / coefficient_name
        orbital_path = temporary / orbital_name
        write_periodic_optimizer_coefficients(coefficient_path, coefficients)
        write_abacus_orbital(
            orbital_path,
            coefficients,
            element="C",
            ecut_ry=100.0,
            rcut_bohr=10.0,
            dr_bohr=0.01,
            smoothing_sigma_bohr=0.1,
        )
        payload = {
            "alpha": alpha,
            "ao_count_atom": AO_COUNT_ATOM,
            "channel_alphas": list(resolved_channel_alphas),
            "coefficients_filename": coefficient_name,
            "coefficients_sha256": sha256(coefficient_path),
            "direction": "original_plus_channel_alpha_times_optimized_minus_original",
            "nu": list(NU),
            "optimized_coefficients": str(optimized),
            "optimized_coefficients_sha256": sha256(optimized),
            "orbital_filename": orbital_name,
            "orbital_sha256": sha256(orbital_path),
            "original_coefficients": str(original),
            "original_coefficients_sha256": sha256(original),
            "profile": "interpolated_dzp",
            "status": "success",
        }
        manifest = temporary / "CANDIDATE.json"
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        (temporary / "provenance.txt").write_text(
            "status=success\n"
            "purpose=full_dzp_reverse_relaxed_direction_sos_line_search\n"
            f"alpha={alpha if alpha is not None else 'channel_resolved'}\n"
            f"channel_alphas={','.join(f'{value:.16g}' for value in resolved_channel_alphas)}\n"
            f"original_coefficients_sha256={payload['original_coefficients_sha256']}\n"
            f"optimized_coefficients_sha256={payload['optimized_coefficients_sha256']}\n"
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
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--alpha", type=float)
    direction.add_argument(
        "--channel-alphas",
        help="comma-separated reverse coefficients for s,p,d,f,g",
    )
    args = parser.parse_args()
    result = prepare_candidate(
        original=args.original,
        optimized=args.optimized,
        root=args.root,
        alpha=args.alpha,
        channel_alphas=(
            tuple(field.strip() for field in args.channel_alphas.split(","))
            if args.channel_alphas is not None
            else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
