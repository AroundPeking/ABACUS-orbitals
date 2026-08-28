#!/usr/bin/env python3
"""Derive a nested periodic SIAB candidate by selecting radial columns."""

from __future__ import annotations

import argparse
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


def parse_zeta_indices(value):
    try:
        indices = tuple(int(field.strip()) for field in value.split(","))
    except ValueError as error:
        raise ValueError("zeta indices must be comma-separated integers") from error
    if not indices or any(index <= 0 for index in indices):
        raise ValueError("zeta indices must be positive and nonempty")
    return indices


def select_radial_columns(
    coefficients,
    *,
    angular_channel,
    zeta_indices,
    element,
):
    if set(coefficients) != {element}:
        raise ValueError("coefficient elements do not match the requested element")
    channels = coefficients[element]
    if (
        type(angular_channel) is not int
        or angular_channel < 0
        or angular_channel >= len(channels)
    ):
        raise ValueError("angular channel is outside the coefficient range")
    zeta_indices = tuple(zeta_indices)
    if not zeta_indices or any(type(index) is not int or index <= 0 for index in zeta_indices):
        raise ValueError("zeta indices must be positive and nonempty")
    if len(set(zeta_indices)) != len(zeta_indices):
        raise ValueError("zeta indices must be unique")
    selected_channel = channels[angular_channel]
    if any(index > selected_channel.shape[1] for index in zeta_indices):
        raise ValueError("zeta index is outside the selected angular channel")

    result = [channel.detach().clone() for channel in channels]
    column_indices = [index - 1 for index in zeta_indices]
    result[angular_channel] = selected_channel[:, column_indices].detach().clone()
    return {element: result}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--input-nu", required=True)
    parser.add_argument("--angular-channel", type=int, required=True)
    parser.add_argument("--zeta-indices", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_nu = parse_counts(args.input_nu)
    zeta_indices = parse_zeta_indices(args.zeta_indices)
    coefficients = read_periodic_optimizer_coefficients(
        args.input,
        element=args.element,
        radial_rows=args.radial_rows,
        expected_nu=input_nu,
    )
    selected = select_radial_columns(
        coefficients,
        angular_channel=args.angular_channel,
        zeta_indices=zeta_indices,
        element=args.element,
    )
    write_periodic_optimizer_coefficients(args.output, selected)


if __name__ == "__main__":
    main()
