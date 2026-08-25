#!/usr/bin/env python3
"""Select a periodic C basis from an independent Galerkin comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SELECTION_ORDER = (
    "global_weighted_relative_trace_log_error",
    "global_weighted_relative_pi_error",
    "ao_count_cell",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_candidate(report, *, allowed_labels, occupied_capture_floor):
    if not isinstance(report, dict) or report.get("format_version") != 1:
        raise ValueError("comparison report must use format version 1")
    datasets = report.get("datasets")
    if (
        not isinstance(datasets, list)
        or len(datasets) != 1
        or datasets[0].get("selected_iq") != 43
    ):
        raise ValueError("selection requires the independent q3 dataset")
    allowed_labels = tuple(allowed_labels)
    if not allowed_labels or len(set(allowed_labels)) != len(allowed_labels):
        raise ValueError("allowed candidate labels must be unique and nonempty")
    if (
        not isinstance(occupied_capture_floor, (int, float))
        or isinstance(occupied_capture_floor, bool)
        or not math.isfinite(occupied_capture_floor)
        or occupied_capture_floor <= 0.0
        or occupied_capture_floor > 1.0
    ):
        raise ValueError("occupied capture floor must be finite in (0, 1]")

    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("comparison report candidates must be a list")
    selected_by_label = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("label") not in allowed_labels:
            continue
        label = candidate["label"]
        if label in selected_by_label:
            raise ValueError("comparison report contains a duplicate allowed candidate")
        selected_by_label[label] = candidate
    if set(selected_by_label) != set(allowed_labels):
        raise ValueError("comparison report must contain exactly the allowed candidates")

    validated = []
    for label in allowed_labels:
        candidate = selected_by_label[label]
        capture = candidate.get("minimum_occupied_capture")
        if (
            not isinstance(capture, (int, float))
            or isinstance(capture, bool)
            or not math.isfinite(capture)
            or capture < occupied_capture_floor
        ):
            raise ValueError("allowed candidate does not pass the occupied capture floor")
        values = []
        for field in SELECTION_ORDER[:2]:
            value = candidate.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError("candidate accuracy metric must be finite and nonnegative")
            values.append(float(value))
        ao_count = candidate.get("ao_count_cell")
        if type(ao_count) is not int or ao_count <= 0:
            raise ValueError("candidate AO count must be a positive integer")
        validated.append((values[0], values[1], ao_count, candidate))

    winner = min(validated, key=lambda item: item[:3])[3]
    return {
        **winner,
        "selection_order": list(SELECTION_ORDER),
        "occupied_capture_floor": float(occupied_capture_floor),
        "allowed_labels": list(allowed_labels),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--occupied-capture-floor", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    comparison = args.comparison.resolve()
    output = args.output.resolve()
    if not comparison.is_file() or comparison.is_symlink():
        raise ValueError("comparison must be a regular file")
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(comparison.read_text(encoding="ascii"))
    selection = select_candidate(
        report,
        allowed_labels=("joint-two-g", "joint-three-g"),
        occupied_capture_floor=args.occupied_capture_floor,
    )
    for candidate in report["candidates"]:
        if candidate.get("label") not in selection["allowed_labels"]:
            continue
        coefficient_path = Path(candidate["coefficients"])
        if not coefficient_path.is_file() or coefficient_path.is_symlink():
            raise ValueError("candidate coefficients must be a regular file")
        if sha256(coefficient_path) != candidate["coefficients_sha256"]:
            raise ValueError("candidate coefficient SHA256 does not match the report")
    selection.update(
        {
            "selection_format_version": 1,
            "comparison": str(comparison),
            "comparison_sha256": sha256(comparison),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
