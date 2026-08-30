#!/usr/bin/env python3
"""Remove complete high-angular-momentum blocks from SIAB coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)


def parse_counts(value):
    try:
        counts = tuple(int(field.strip()) for field in value.split(","))
    except ValueError as error:
        raise ValueError("orbital counts must be comma-separated integers") from error
    if not counts or any(count < 0 for count in counts) or not any(counts):
        raise ValueError("orbital counts must be nonnegative and nonempty")
    return counts


def truncate_angular_channels(coefficients, *, target_lmax, element):
    if set(coefficients) != {element}:
        raise ValueError("coefficient elements do not match the requested element")
    channels = coefficients[element]
    if (
        type(target_lmax) is not int
        or target_lmax < 0
        or target_lmax >= len(channels) - 1
    ):
        raise ValueError(
            "target_lmax must be non-negative and smaller than the source Lmax"
        )
    return {
        element: [
            channel.detach().clone() for channel in channels[: target_lmax + 1]
        ]
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--input-nu", required=True)
    parser.add_argument("--target-lmax", type=int, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise ValueError("coefficient output parent directory does not exist")
    input_nu = parse_counts(args.input_nu)
    coefficients = read_periodic_optimizer_coefficients(
        args.input,
        element=args.element,
        radial_rows=args.radial_rows,
        expected_nu=input_nu,
    )
    truncated = truncate_angular_channels(
        coefficients,
        target_lmax=args.target_lmax,
        element=args.element,
    )
    write_periodic_optimizer_coefficients(args.output, truncated)
    output_nu = tuple(channel.shape[1] for channel in truncated[args.element])
    report = {
        "element": args.element,
        "input": str(args.input.resolve()),
        "input_nu": input_nu,
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_nu": output_nu,
        "output_sha256": sha256(args.output),
        "radial_rows": args.radial_rows,
        "target_lmax": args.target_lmax,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.write_text(encoded, encoding="ascii")
    print(encoded, end="")


if __name__ == "__main__":
    main()
