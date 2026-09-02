#!/usr/bin/env python3
"""Validate one standard FD8 diamond-C basis-optimization q dataset."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frequency-grid-output", type=Path, required=True)
    parser.add_argument(
        "--frequency-grid-source",
        choices=("greenx_minimax",),
        required=True,
    )
    parser.add_argument("--expected-abacus-commit", required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--expected-orbital-sha256", required=True)
    parser.add_argument("--expected-pseudopotential-sha256", required=True)
    parser.add_argument("--expected-auxiliary-basis-sha256", required=True)
    parser.add_argument("--expected-iq", type=int, required=True)
    parser.add_argument("--expected-q-weight", type=float, required=True)
    parser.add_argument("--maximum-solver-relative-residual", type=float, default=1.01e-6)
    return parser.parse_args(argv)


def read_records(path):
    records = {}
    with Path(path).open(encoding="ascii") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2:
                records.setdefault(fields[0], []).append(fields[1:])
    return records


def scalar(records, key):
    values = records.get(key, [])
    if len(values) != 1 or len(values[0]) != 1:
        raise ValueError(f"missing or non-scalar field: {key}")
    return values[0][0]


def read_input(path):
    values = {}
    with Path(path).open(encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "INPUT_PARAMETERS")):
                continue
            fields = stripped.split()
            if len(fields) >= 2:
                values[fields[0]] = fields[1:]
    return values


def parse_frequency_rows(records):
    rows = []
    for fields in records.get("frequency", []):
        if len(fields) != 3:
            raise ValueError("frequency record has invalid width")
        rows.append((int(fields[0]), float(fields[1]), float(fields[2])))
    if len(rows) != 12 or [row[0] for row in rows] != list(range(12)):
        raise ValueError("standard FD8 dataset requires indexed twelve-point frequencies")
    previous = -math.inf
    for _, omega, weight in rows:
        if (
            not math.isfinite(omega)
            or not math.isfinite(weight)
            or omega <= previous
            or weight <= 0.0
        ):
            raise ValueError("frequency grid must be finite, positive, and increasing")
        previous = omega
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    manifest = read_records(args.manifest)
    status = read_records(args.status)
    input_values = read_input(args.input)

    expected = {
        "abacus_commit": args.expected_abacus_commit,
        "executable_sha256": args.expected_executable_sha256,
        "orbital_sha256": args.expected_orbital_sha256,
        "pseudopotential_sha256": args.expected_pseudopotential_sha256,
        "auxiliary_basis_sha256": args.expected_auxiliary_basis_sha256,
        "auxiliary_basis_source": "product_pca",
        "kernel": "full_coulomb",
        "q_count": "64",
        "selected_iq": str(args.expected_iq),
        "frequency_count": "12",
    }
    for key, value in expected.items():
        if scalar(manifest, key) != value:
            raise ValueError(f"manifest field differs: {key}")
    if scalar(status, "status") != "success" or scalar(status, "all_converged") != "yes":
        raise ValueError("response status is not successfully converged")
    residual = float(scalar(status, "max_solver_relative_residual"))
    if not math.isfinite(residual) or residual > args.maximum_solver_relative_residual:
        raise ValueError("solver residual exceeds the standard gate")
    q_weight = float(scalar(manifest, "q_weight"))
    if not math.isclose(q_weight, args.expected_q_weight, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("q weight differs")

    raw_dimension = int(scalar(manifest, "raw_auxiliary_dimension"))
    retained_rank = int(scalar(manifest, "whitened_auxiliary_rank"))
    if raw_dimension <= 0 or not 0 < retained_rank <= raw_dimension:
        raise ValueError("auxiliary dimensions are invalid")

    if "sternheimer_frequency_grid_file" in input_values:
        raise ValueError("GreenX minimax q1 must generate its basis-dependent grid")

    rows = parse_frequency_rows(manifest)

    args.frequency_grid_output.write_text(
        "\n".join(
            f"{index + 1}\t{omega:.17e}\t{weight:.17e}"
            for index, omega, weight in rows
        )
        + "\n",
        encoding="ascii",
    )
    payload = {
        "status": "success",
        "frequency_grid_source": args.frequency_grid_source,
        "frequency_count": len(rows),
        "frequency_grid_sha256": sha256(args.frequency_grid_output),
        "raw_auxiliary_dimension": raw_dimension,
        "whitened_auxiliary_rank": retained_rank,
        "q_index": args.expected_iq,
        "q_weight": q_weight,
        "max_solver_relative_residual": residual,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return payload


if __name__ == "__main__":
    main()
