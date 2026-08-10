#!/usr/bin/env python3
"""Evaluate the held-out H2 response-orbital Galerkin sidecar."""

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch


SIAB_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPT_DIR = SIAB_ROOT / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from IO.read_sternheimer_fixed_ao import read_sternheimer_fixed_ao
from IO.read_sternheimer_primitive_galerkin import (
    read_sternheimer_primitive_galerkin,
)
from response_orbital_galerkin_gate import (
    evaluate_response_orbital_galerkin_gate,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=pathlib.Path)
    parser.add_argument("output_json", type=pathlib.Path)
    parser.add_argument("output_npz", type=pathlib.Path)
    parser.add_argument("--sidecar-commit", required=True)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--condition-limit", type=float, default=1.0e12)
    parser.add_argument(
        "--spectral-direct-relative-tolerance", type=float, default=1.0e-10
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_json = args.output_json.resolve()
    output_npz = args.output_npz.resolve()
    for path in (output_json, output_npz):
        if path.exists():
            raise ValueError(f"output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    case_dir = args.case_dir.resolve()
    response_path = case_dir / "sternheimer_galerkin_response.dat"
    fixed_path = case_dir / "sternheimer_galerkin_fixed_ao.dat"
    primitive = read_sternheimer_primitive_galerkin(response_path)
    fixed_ao = read_sternheimer_fixed_ao(fixed_path)
    if primitive.provenance["abacus_commit"] != args.sidecar_commit:
        raise ValueError("response sidecar commit differs from --sidecar-commit")
    if fixed_ao.provenance["abacus_commit"] != args.sidecar_commit:
        raise ValueError("fixed-AO sidecar commit differs from --sidecar-commit")

    result = evaluate_response_orbital_galerkin_gate(
        primitive,
        fixed_ao,
        relative_rank_tolerance=args.relative_rank_tolerance,
        condition_limit=args.condition_limit,
    )
    np.savez(
        output_npz,
        response_m=result.direct.response.numpy(),
        response_half_m=result.direct.response_half.numpy(),
        frequency_ha=primitive.frequency_ha.numpy(),
        frequency_weight_ha=primitive.frequency_weight_ha.numpy(),
    )
    eigenvalues = torch.linalg.eigvalsh(result.direct.response)
    payload = {
        "method": "h2_response_orbital_galerkin_gate_v1",
        "sidecar_abacus_commit": args.sidecar_commit,
        "representation": primitive.representation,
        "relative_rank_tolerance": args.relative_rank_tolerance,
        "condition_limit": args.condition_limit,
        "dimensions": {
            "response_ao": result.response_dimension,
            "fixed_ao": result.fixed_dimension,
            "union": result.response_dimension + result.fixed_dimension,
            "auxiliary": len(primitive.channels),
            "frequency": primitive.frequency_ha.shape[0],
        },
        "direct_solver": {
            "retained_rank_by_spin": list(
                result.direct.retained_parent_rank_by_spin
            ),
            "dropped_rank_by_spin": list(result.direct.dropped_parent_rank_by_spin),
            "projected_overlap_condition_by_spin": list(
                result.direct.projected_overlap_condition_by_spin
            ),
            "fixed_ao_overlap_condition": (
                result.direct.fixed_ao_overlap_condition
            ),
            "fixed_ao_eigenvalue_max_abs_error_ha": (
                result.direct.fixed_ao_eigenvalue_max_abs_error_ha
            ),
        },
        "spectral_solver": list(result.spectral_diagnostics),
        "spectral_direct_response_relative_frobenius": (
            result.spectral_direct_relative_frobenius
        ),
        "spectral_direct_response_max_abs_difference": (
            result.spectral_direct_max_abs_difference
        ),
        "spectral_direct_relative_tolerance": (
            args.spectral_direct_relative_tolerance
        ),
        "spectral_direct_gate_passed": (
            result.spectral_direct_relative_frobenius
            <= args.spectral_direct_relative_tolerance
        ),
        "response_m_frobenius": float(
            torch.linalg.vector_norm(result.direct.response)
        ),
        "response_m_max_abs": float(torch.max(torch.abs(result.direct.response))),
        "response_m_minimum_eigenvalue_by_frequency": [
            float(value) for value in eigenvalues[:, 0]
        ],
        "response_m_maximum_eigenvalue_by_frequency": [
            float(value) for value in eigenvalues[:, -1]
        ],
        "input_sha256": {
            str(path): _sha256(path) for path in (response_path, fixed_path)
        },
        "output_npz": str(output_npz),
        "output_npz_sha256": _sha256(output_npz),
        "provenance": primitive.provenance,
    }
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["spectral_direct_gate_passed"]:
        raise RuntimeError("spectral and direct projected responses differ")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
