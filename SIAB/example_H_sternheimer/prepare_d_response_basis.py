#!/usr/bin/env python3
"""Prepare d-response SIAB coefficients from an atomic Sternheimer target."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


EXAMPLE_DIR = Path(__file__).resolve().parent
OPTIMIZER_DIR = EXAMPLE_DIR.parent / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPTIMIZER_DIR))

import IO.func_C
import IO.read_sternheimer
from attribute_dict import AttributeDict
from main import _expand_fixed_orbitals
from response_basis import canonicalize_columns, replace_channel_coefficients
from sternheimer_spillage import (
    radial_residual_spectrum,
    shell_count_for_capture,
)


FIXED_DZP = (
    {"element": "H", "l": 0, "zeta": 1},
    {"element": "H", "l": 0, "zeta": 2},
    {"element": "H", "l": 1, "zeta": 1},
)
REPORT_THRESHOLDS = (0.90, 0.95, 0.99, 0.999)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_info_element(nprimitive, nu):
    result = AttributeDict()
    result["H"].index = 0
    result["H"].Nl = len(nu)
    result["H"].Ne = nprimitive
    result["H"].Nu = list(nu)
    return result


def validate_h_target(data):
    counts = []
    for l in range(3):
        blocks = sorted(
            (
                block
                for block in data.blocks
                if block.element == "H" and block.atom_index == 0 and block.l == l
            ),
            key=lambda block: block.m,
        )
        expected_m = list(range(-l, l + 1))
        if [block.m for block in blocks] != expected_m:
            raise ValueError(
                f"H/atom0/l{l} blocks must contain m={expected_m}"
            )
        block_counts = {block.n_primitive for block in blocks}
        if len(block_counts) != 1:
            raise ValueError(f"H/atom0/l{l} has inconsistent radial counts")
        counts.append(block_counts.pop())
    if len(set(counts)) != 1:
        raise ValueError(f"H s/p/d radial counts differ: {counts}")
    return counts[0]


def spectrum_record(spectrum):
    return {
        "l": spectrum.l,
        "magnetic_channels": list(spectrum.magnetic_channels),
        "numerical_rank": spectrum.numerical_rank,
        "overlap_relative_deviation": spectrum.overlap_relative_deviation,
        "eigenvalues": spectrum.eigenvalues.tolist(),
        "cumulative_capture": spectrum.cumulative_capture.tolist(),
        "shell_counts": {
            f"{threshold:.3f}": shell_count_for_capture(spectrum, threshold)
            for threshold in REPORT_THRESHOLDS
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--capture-threshold", type=float, default=0.99)
    parser.add_argument("--s-shells", type=int, default=4)
    parser.add_argument("--p-shells", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.capture_threshold <= 1.0:
        raise ValueError("capture-threshold must satisfy 0 < value <= 1")
    if args.s_shells < 2 or args.p_shells < 1:
        raise ValueError("the candidate must contain the fixed H DZP core")
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("seed must satisfy 0 <= seed < 2**32")

    target = args.target.resolve(strict=True)
    baseline = args.baseline.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    data = IO.read_sternheimer.read_sternheimer(target)
    nprimitive = validate_h_target(data)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    analysis_info = make_info_element(
        nprimitive, [args.s_shells, args.p_shells, 1]
    )
    analysis_c, _ = IO.func_C.read_C_init(
        baseline, analysis_info, return_metadata=True
    )
    fixed_orbitals = _expand_fixed_orbitals(data, analysis_c, FIXED_DZP)
    spectra = [
        radial_residual_spectrum(
            data,
            analysis_c,
            fixed_orbitals,
            element="H",
            atom_index=0,
            l=l,
        )
        for l in range(3)
    ]
    d_shells = shell_count_for_capture(spectra[2], args.capture_threshold)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    candidate_info = make_info_element(
        nprimitive, [args.s_shells, args.p_shells, d_shells]
    )
    candidate, metadata = IO.func_C.read_C_init(
        baseline, candidate_info, return_metadata=True
    )
    d_coefficients = canonicalize_columns(
        spectra[2].coefficients[:, :d_shells]
    )
    replace_channel_coefficients(candidate, "H", 2, d_coefficients)

    for l in (0, 1):
        loaded = analysis_c["H"][l]
        if not torch.equal(candidate["H"][l], loaded):
            raise RuntimeError(f"candidate changed baseline H/l{l} coefficients")

    coefficient_path = output_dir / "INITIAL_RESPONSE_COEFFICIENTS.txt"
    IO.func_C.write_C(coefficient_path, candidate, 0.0)
    summary = {
        "format_version": 1,
        "selection_rule": "atomic fixed-DZP residual spectrum",
        "capture_threshold": args.capture_threshold,
        "selected_nu": {"H": [args.s_shells, args.p_shells, d_shells]},
        "seed": args.seed,
        "target": str(target),
        "target_sha256": sha256(target),
        "baseline": str(baseline),
        "baseline_sha256": sha256(baseline),
        "candidate": str(coefficient_path),
        "candidate_sha256": sha256(coefficient_path),
        "nprimitive": nprimitive,
        "loaded_indices": [list(value) for value in sorted(metadata.loaded_indices)],
        "appended_indices": [
            list(value) for value in sorted(metadata.appended_indices)
        ],
        "spectra": [spectrum_record(spectrum) for spectrum in spectra],
    }
    summary_path = output_dir / "response_spectrum.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
