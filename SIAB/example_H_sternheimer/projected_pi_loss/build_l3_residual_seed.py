#!/usr/bin/env python3
"""Append the leading atomic l=3 residual mode to the selected H basis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent
SIAB_DIR = EXAMPLE_DIR.parent
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
SELECTION_DIR = EXAMPLE_DIR / "greedy_response_selection"
sys.path.insert(0, str(OPT_DIR))
sys.path.insert(0, str(SELECTION_DIR))

from IO.read_sternheimer import read_sternheimer  # noqa: E402
from response_selection_campaign import (  # noqa: E402
    read_optimizer_coefficients,
    write_optimizer_coefficients,
)
from sternheimer_spillage import radial_residual_spectrum_many  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefficients", required=True)
    parser.add_argument("--atom-target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--magnetic-overlap-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--condition-limit", type=float, default=1.0e12)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    coefficient_path = Path(args.coefficients).resolve()
    target_path = Path(args.atom_target).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    for path in (coefficient_path, target_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    expected_nu=(3, 2, 2, 0, 0)
    coefficients = read_optimizer_coefficients(
        coefficient_path,
        element="H",
        radial_rows=25,
        max_l=4,
        expected_nu=expected_nu,
    )
    current_specs = tuple(
        {"element": "H", "l": l_value, "zeta": zeta + 1}
        for l_value, channel in enumerate(coefficients["H"])
        for zeta in range(channel.shape[1])
    )
    atom = read_sternheimer(target_path)
    spectrum = radial_residual_spectrum_many(
        (atom,),
        coefficients,
        current_specs,
        "H",
        3,
        relative_rank_tolerance=args.relative_rank_tolerance,
        magnetic_overlap_tolerance=args.magnetic_overlap_tolerance,
        condition_limit=args.condition_limit,
    )
    leading_eigenvalue = float(spectrum.eigenvalues[0])
    if leading_eigenvalue <= 0.0:
        raise RuntimeError("the leading l=3 residual eigenvalue is not positive")

    candidate = {
        element: [channel.detach().clone() for channel in channels]
        for element, channels in coefficients.items()
    }
    leading_mode = spectrum.coefficients[:, 0].detach().clone().reshape(-1, 1)
    candidate["H"][3] = torch.cat((candidate["H"][3], leading_mode), dim=1)
    if tuple(channel.shape[1] for channel in candidate["H"]) != (3, 2, 2, 1, 0):
        raise RuntimeError("the l=3 seed changed an unexpected orbital channel")
    write_optimizer_coefficients(output_path, candidate)

    payload = {
        "format_version": 1,
        "source_family": "H",
        "source_coefficients": str(coefficient_path),
        "source_coefficients_sha256": sha256(coefficient_path),
        "source_target": str(target_path),
        "source_target_sha256": sha256(target_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "l": 3,
        "radial_mode": 0,
        "added_ao": 7,
        "leading_eigenvalue": leading_eigenvalue,
        "gain_per_added_ao": leading_eigenvalue / 7.0,
        "leading_cumulative_capture": float(spectrum.cumulative_capture[0]),
        "numerical_rank": spectrum.numerical_rank,
        "overlap_relative_deviation": spectrum.overlap_relative_deviation,
        "relative_rank_tolerance": args.relative_rank_tolerance,
        "magnetic_overlap_tolerance": args.magnetic_overlap_tolerance,
        "condition_limit": args.condition_limit,
        "eigenvalues": [float(value) for value in spectrum.eigenvalues],
        "cumulative_capture": [
            float(value) for value in spectrum.cumulative_capture
        ],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
