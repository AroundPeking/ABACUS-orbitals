#!/usr/bin/env python3
"""Numerically compare two SIAB Sternheimer response-target files."""

import argparse
import json
from pathlib import Path

import torch

from IO.read_sternheimer import read_sternheimer


def _max_abs(tensor):
    return float(torch.max(torch.abs(tensor)).item()) if tensor.numel() else 0.0


def _relative_frobenius(reference, candidate):
    difference = torch.linalg.vector_norm(candidate - reference).item()
    norm = torch.linalg.vector_norm(reference).item()
    if norm == 0.0:
        return float(difference)
    return float(difference / norm)


def _compare_tensor(reference, candidate, name):
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"{name} shape mismatch: {tuple(reference.shape)} != "
            f"{tuple(candidate.shape)}"
        )
    return {
        "max_abs": _max_abs(candidate - reference),
        "relative_frobenius": _relative_frobenius(reference, candidate),
        "shape": list(reference.shape),
    }


def _compare_provenance(reference, candidate):
    reference = dict(reference)
    candidate = dict(candidate)
    reference_threads = reference.pop("omp_threads", None)
    candidate_threads = candidate.pop("omp_threads", None)

    if (reference_threads is None) != (candidate_threads is None):
        raise ValueError("target provenance differs: omp_threads")
    for name, value in (
        ("reference omp_threads", reference_threads),
        ("candidate omp_threads", candidate_threads),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")

    differing_fields = sorted(
        key
        for key in set(reference) | set(candidate)
        if reference.get(key) != candidate.get(key)
    )
    if differing_fields:
        raise ValueError(
            "target provenance differs: " + ", ".join(differing_fields)
        )
    return {
        "reference_omp_threads": reference_threads,
        "candidate_omp_threads": candidate_threads,
    }


def compare(reference_path, candidate_path):
    reference = read_sternheimer(reference_path)
    candidate = read_sternheimer(candidate_path)

    if reference.blocks != candidate.blocks:
        raise ValueError("primitive block layout differs")
    provenance = _compare_provenance(
        reference.provenance, candidate.provenance
    )

    integer_metadata = {
        "occupied_state": bool(
            torch.equal(reference.occupied_state, candidate.occupied_state)
        ),
        "auxiliary_channel": bool(
            torch.equal(reference.auxiliary_channel, candidate.auxiliary_channel)
        ),
    }
    if not all(integer_metadata.values()):
        raise ValueError("integer reference metadata differs")

    return {
        "reference": str(Path(reference_path)),
        "candidate": str(Path(candidate_path)),
        "provenance": provenance,
        "q": _compare_tensor(reference.q, candidate.q, "OVERLAP_Q"),
        "overlap": _compare_tensor(
            reference.overlap, candidate.overlap, "OVERLAP_S"
        ),
        "frequency_ha": _compare_tensor(
            reference.frequency_ha, candidate.frequency_ha, "frequency_ha"
        ),
        "occupation": _compare_tensor(
            reference.occupation, candidate.occupation, "occupation"
        ),
        "frequency_weight": _compare_tensor(
            reference.frequency_weight, candidate.frequency_weight, "frequency_weight"
        ),
        "norm": _compare_tensor(reference.norm, candidate.norm, "norm"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-abs-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--relative-frobenius-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()

    if args.max_abs_tolerance <= 0.0 or args.relative_frobenius_tolerance <= 0.0:
        raise ValueError("comparison tolerances must be positive")

    result = compare(args.reference, args.candidate)
    checks = {}
    for name, metrics in result.items():
        if not isinstance(metrics, dict) or "max_abs" not in metrics:
            continue
        checks[name] = (
            metrics["max_abs"] <= args.max_abs_tolerance
            and metrics["relative_frobenius"]
            <= args.relative_frobenius_tolerance
        )
    result["tolerances"] = {
        "max_abs": args.max_abs_tolerance,
        "relative_frobenius": args.relative_frobenius_tolerance,
    }
    result["passed"] = all(checks.values())
    result["checks"] = checks
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit("Sternheimer target comparison exceeds tolerance")


if __name__ == "__main__":
    main()
