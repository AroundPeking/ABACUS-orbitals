#!/usr/bin/env python3
"""Append deterministic Bessel-complement columns to periodic SIAB coefficients."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

import torch  # noqa: E402

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


def expand_coefficients(coefficients, target_nu, *, element):
    if set(coefficients) != {element}:
        raise ValueError("coefficient elements do not match the requested element")
    channels = coefficients[element]
    target_nu = tuple(target_nu)
    if len(channels) != len(target_nu):
        raise ValueError("target_nu must define every angular channel")

    expanded = []
    for channel, target_count in zip(channels, target_nu):
        if target_count < channel.shape[1]:
            raise ValueError("target_nu cannot remove existing orbitals")
        value = channel.detach().clone()
        while value.shape[1] < target_count:
            if value.shape[1] >= value.shape[0]:
                raise RuntimeError("radial coefficient channel has no complement")
            if value.shape[1]:
                frame, _ = torch.linalg.qr(value, mode="reduced")
                projector = (
                    torch.eye(value.shape[0], dtype=torch.float64)
                    - frame.matmul(frame.transpose(0, 1))
                )
            else:
                projector = torch.eye(value.shape[0], dtype=torch.float64)
            residual_norm = torch.diagonal(projector)
            index = int(torch.argmax(residual_norm))
            column = projector[:, index]
            norm = torch.linalg.norm(column)
            if float(norm) <= 1.0e-12:
                raise RuntimeError("radial coefficient complement is numerically empty")
            value = torch.cat((value, (column / norm)[:, None]), dim=1)
        expanded.append(value)
    return {element: expanded}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--input-nu", default="3,3,2,0,0")
    parser.add_argument("--target-nu", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_nu = parse_counts(args.input_nu)
    target_nu = parse_counts(args.target_nu)
    if len(input_nu) != len(target_nu):
        raise ValueError("input_nu and target_nu must have the same length")
    coefficients = read_periodic_optimizer_coefficients(
        args.input,
        element=args.element,
        radial_rows=args.radial_rows,
        expected_nu=input_nu,
    )
    expanded = expand_coefficients(
        coefficients,
        target_nu,
        element=args.element,
    )
    write_periodic_optimizer_coefficients(args.output, expanded)


if __name__ == "__main__":
    main()
