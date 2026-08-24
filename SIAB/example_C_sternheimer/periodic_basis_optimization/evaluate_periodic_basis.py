#!/usr/bin/env python3
"""Evaluate a C SIAB basis and its Bessel mother space against periodic Pi."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
H_SELECTION_DIR = SIAB_DIR / "example_H_sternheimer/greedy_response_selection"
sys.path.insert(0, str(OPT_DIR))
sys.path.insert(0, str(H_SELECTION_DIR))

from periodic_galerkin_campaign import evaluate_periodic_basis_capacity  # noqa: E402
from periodic_galerkin_data import read_periodic_galerkin_dataset  # noqa: E402
from response_selection_campaign import read_optimizer_coefficients  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_nu(value, *, max_l):
    try:
        nu = tuple(int(field.strip()) for field in value.split(","))
    except ValueError as exc:
        raise ValueError("nu must be a comma-separated integer list") from exc
    required = max_l + 1
    if len(nu) != required:
        names = {5: "five"}
        count = names.get(required, str(required))
        raise ValueError(f"nu must contain exactly {count} angular channels")
    if any(count < 0 for count in nu):
        raise ValueError("nu counts must be nonnegative")
    if not any(nu):
        raise ValueError("nu must define a nonempty basis")
    return nu


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--nu", default="3,3,2,0,0")
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--max-l", type=int, default=4)
    parser.add_argument("--mother-response-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--condition-limit", type=float, default=1.0e12)
    parser.add_argument("--occupied-capture-tolerance", type=float, default=1.0e-6)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    nu = parse_nu(args.nu, max_l=args.max_l)
    dataset_path = args.dataset.resolve()
    coefficient_path = args.coefficients.resolve()
    output_path = args.output.resolve()
    if not dataset_path.is_dir() or dataset_path.is_symlink():
        raise ValueError("dataset must be a real directory")
    if (
        not coefficient_path.is_file()
        or coefficient_path.is_symlink()
        or coefficient_path.stat().st_size == 0
    ):
        raise ValueError("coefficients must be a nonempty regular file")
    if output_path.exists():
        raise FileExistsError(output_path)
    if (
        args.radial_rows <= 0
        or args.max_l < 0
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                args.mother_response_tolerance,
                args.relative_rank_tolerance,
                args.condition_limit,
                args.occupied_capture_tolerance,
            )
        )
    ):
        raise ValueError("evaluation dimensions and tolerances must be positive")

    dataset = read_periodic_galerkin_dataset(dataset_path)
    coefficients = read_optimizer_coefficients(
        coefficient_path,
        element=args.element,
        radial_rows=args.radial_rows,
        max_l=args.max_l,
        expected_nu=nu,
    )
    report = evaluate_periodic_basis_capacity(
        dataset,
        coefficients,
        mother_response_tolerance=args.mother_response_tolerance,
        relative_rank_tolerance=args.relative_rank_tolerance,
        condition_limit=args.condition_limit,
        occupied_capture_tolerance=args.occupied_capture_tolerance,
    )
    report.update(
        {
            "format_version": 1,
            "inputs": {
                "dataset": str(dataset_path),
                "coefficients": str(coefficient_path),
                "coefficients_sha256": sha256(coefficient_path),
                "abacus_commit": dataset.abacus_commit,
                "executable_sha256": dataset.executable_sha256,
                "orbital_sha256": dataset.orbital_sha256,
                "pseudopotential_sha256": dataset.pseudopotential_sha256,
                "auxiliary_basis_sha256": dataset.auxiliary_basis_sha256,
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
