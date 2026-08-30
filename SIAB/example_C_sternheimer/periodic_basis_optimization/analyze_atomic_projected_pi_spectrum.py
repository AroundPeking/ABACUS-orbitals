#!/usr/bin/env python3
"""Diagnose atomic projected-Pi causality without evaluating a trace-log."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from IO.read_sternheimer import read_sternheimer  # noqa: E402
from IO.read_sternheimer_source import read_sternheimer_source  # noqa: E402
from periodic_galerkin_basis import read_periodic_optimizer_coefficients  # noqa: E402
from projected_pi import ProjectedPiEvaluator  # noqa: E402
from sternheimer_source_pair import pair_response_and_source  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_spectrum(matrices):
    if (
        not isinstance(matrices, torch.Tensor)
        or matrices.ndim != 3
        or matrices.shape[-2] != matrices.shape[-1]
    ):
        raise ValueError("Pi must be a stack of square frequency matrices")
    if not bool(torch.all(torch.isfinite(matrices))):
        raise ValueError("Pi frequency matrices must be finite")
    reports = []
    for matrix in matrices:
        hermitian = 0.5 * (matrix + matrix.transpose(-2, -1).conj())
        eigenvalues = torch.linalg.eigvalsh(hermitian)
        minimum = float(torch.min(eigenvalues).detach())
        maximum = float(torch.max(eigenvalues).detach())
        minimum_argument = 1.0 - maximum
        reports.append(
            {
                "minimum_eigenvalue": minimum,
                "maximum_eigenvalue": maximum,
                "minimum_i_minus_pi_eigenvalue": minimum_argument,
                "i_minus_pi_positive": math.isfinite(minimum_argument)
                and minimum_argument > 0.0,
            }
        )
    return reports


def parse_nu(value, max_l):
    try:
        nu = tuple(int(field.strip()) for field in value.split(","))
    except ValueError as error:
        raise ValueError("nu must be comma-separated integers") from error
    if len(nu) != max_l + 1 or any(count < 0 for count in nu) or not any(nu):
        raise ValueError("nu must define one nonnegative count per channel")
    return nu


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--element", default="C")
    parser.add_argument("--nu", default="3,3,2,0,0")
    parser.add_argument("--max-l", type=int, default=4)
    parser.add_argument("--radial-rows", type=int, default=31)
    args = parser.parse_args(argv)
    nu = parse_nu(args.nu, args.max_l)
    for path in (args.response, args.source, args.candidate):
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError("all inputs must be nonempty regular files")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output already exists")

    pair = pair_response_and_source(
        read_sternheimer(args.response),
        read_sternheimer_source(args.source),
    )
    coefficients = read_periodic_optimizer_coefficients(
        args.candidate,
        element=args.element,
        radial_rows=args.radial_rows,
        expected_nu=nu,
    )
    evaluator = ProjectedPiEvaluator(pair)
    with torch.no_grad():
        result = evaluator.evaluate(coefficients)
    reference = matrix_spectrum(result.reference_pi)
    candidate = matrix_spectrum(result.candidate_pi)
    report = {
        "format_version": 1,
        "status": "success",
        "scope": "atomic projected-Pi spectrum diagnostic; not an RPA energy",
        "response": str(args.response.resolve()),
        "response_sha256": sha256(args.response),
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "nu": list(nu),
        "frequency_ha": [float(value) for value in result.frequency_ha],
        "frequency_weight": [float(value) for value in result.frequency_weight],
        "projected_pi_loss": float(result.loss),
        "maximum_candidate_overlap_condition": float(
            result.max_candidate_condition
        ),
        "reference_spectrum": reference,
        "candidate_spectrum": candidate,
        "reference_i_minus_pi_positive": all(
            item["i_minus_pi_positive"] for item in reference
        ),
        "candidate_i_minus_pi_positive": all(
            item["i_minus_pi_positive"] for item in candidate
        ),
        "response_provenance": pair.response.provenance,
        "source_provenance": pair.source.provenance,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
