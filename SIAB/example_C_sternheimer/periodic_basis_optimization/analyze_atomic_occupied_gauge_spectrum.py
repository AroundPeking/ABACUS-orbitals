#!/usr/bin/env python3
"""Diagnose atomic projected-Pi before and after occupied-gauge alignment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from analyze_atomic_projected_pi_spectrum import matrix_spectrum, sha256  # noqa: E402
from atomic_occupied_gauge import (  # noqa: E402
    derive_occupied_gauge,
    rotate_source_rows_to_response_gauge,
)
from IO.read_sternheimer import read_sternheimer  # noqa: E402
from IO.read_sternheimer_source import read_sternheimer_source  # noqa: E402
from projected_pi import ProjectedPiEvaluator  # noqa: E402
from sternheimer_source_pair import pair_response_and_source  # noqa: E402


def reference_pi(response, source):
    evaluator = ProjectedPiEvaluator(pair_response_and_source(response, source))
    return evaluator._reference_pi


def complex_matrix_as_pairs(matrix):
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in matrix
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--response-wfs", required=True, type=Path, action="append"
    )
    parser.add_argument("--source-wfs", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = (
        args.response,
        args.source,
        *args.response_wfs,
        *args.source_wfs,
    )
    for path in inputs:
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError("all inputs must be nonempty regular files")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output already exists")

    response = read_sternheimer(args.response)
    source = read_sternheimer_source(args.source)
    gauge = derive_occupied_gauge(args.response_wfs, args.source_wfs)
    aligned_d = rotate_source_rows_to_response_gauge(
        source.d,
        source.occupied_state,
        source.auxiliary_channel,
        gauge.transform,
    )
    aligned_source = replace(source, d=aligned_d)
    with torch.no_grad():
        before = matrix_spectrum(reference_pi(response, source))
        after = matrix_spectrum(reference_pi(response, aligned_source))

    report = {
        "format_version": 1,
        "status": "success",
        "scope": (
            "atomic occupied-gauge diagnostic; corrected matrices are not "
            "production RPA data"
        ),
        "response": str(args.response.resolve()),
        "response_sha256": sha256(args.response),
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "response_wfs": [str(path.resolve()) for path in args.response_wfs],
        "response_wfs_sha256": [sha256(path) for path in args.response_wfs],
        "source_wfs": [str(path.resolve()) for path in args.source_wfs],
        "source_wfs_sha256": [sha256(path) for path in args.source_wfs],
        "occupied_counts_by_spin": list(gauge.occupied_counts),
        "subspace_residuals_by_spin": list(gauge.subspace_residuals),
        "unitarity_errors_by_spin": list(gauge.unitarity_errors),
        "maximum_subspace_residual": gauge.maximum_subspace_residual,
        "maximum_unitarity_error": gauge.maximum_unitarity_error,
        "maximum_occupied_eigenvalue_difference_ry": (
            gauge.maximum_eigenvalue_difference_ry
        ),
        "response_to_source_gauge": complex_matrix_as_pairs(gauge.transform),
        "uncorrected_reference_spectrum": before,
        "gauge_corrected_reference_spectrum": after,
        "uncorrected_i_minus_pi_positive": all(
            item["i_minus_pi_positive"] for item in before
        ),
        "gauge_corrected_i_minus_pi_positive": all(
            item["i_minus_pi_positive"] for item in after
        ),
        "response_provenance": response.provenance,
        "source_provenance": source.provenance,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
