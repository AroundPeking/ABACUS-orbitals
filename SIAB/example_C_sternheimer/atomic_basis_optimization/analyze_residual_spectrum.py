#!/usr/bin/env python3
"""Rank C atomic residual response channels and seed one radial shell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
C_EXAMPLE_DIR = SCRIPT_DIR.parent
SIAB_DIR = C_EXAMPLE_DIR.parent
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
H_SELECTION_DIR = SIAB_DIR / "example_H_sternheimer/greedy_response_selection"
sys.path.insert(0, str(OPT_DIR))
sys.path.insert(0, str(H_SELECTION_DIR))

from IO.read_sternheimer import read_sternheimer  # noqa: E402
from response_selection_campaign import (  # noqa: E402
    read_optimizer_coefficients,
    write_optimizer_coefficients,
)
from sternheimer_spillage import radial_residual_spectrum_many  # noqa: E402


EXPECTED_TARGET_SHA256 = (
    "e976c164595758029cb91ebe3913af6865780ef95fdb48954c07075bd0c7e3ff"
)
EXPECTED_NU = (3, 3, 2, 0, 0)
RADIAL_ROWS = 31


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spectrum_record(spectrum, *, l):
    eigenvalues = [float(value) for value in spectrum.eigenvalues]
    cumulative = [float(value) for value in spectrum.cumulative_capture]
    if not eigenvalues or any(not math.isfinite(value) for value in eigenvalues):
        raise RuntimeError(f"l={l} residual spectrum is empty or non-finite")
    leading = eigenvalues[0]
    added_ao = 2 * l + 1
    return {
        "l": l,
        "total_weight": sum(eigenvalues),
        "leading_eigenvalue": leading,
        "cumulative_capture_first_three": cumulative[:3],
        "numerical_rank": int(spectrum.numerical_rank),
        "overlap_relative_deviation": float(spectrum.overlap_relative_deviation),
        "added_ao": added_ao,
        "score": leading / added_ao,
        "eigenvalues": eigenvalues,
        "cumulative_capture": cumulative,
    }


def select_channel(records, *, tie_fraction=0.01):
    ranked = sorted(
        (record for record in records if record["score"] > 0.0),
        key=lambda record: (-record["score"], record["l"]),
    )
    if not ranked:
        raise RuntimeError("no residual channel has a positive score")
    winner = ranked[0]
    if len(ranked) > 1:
        relative_gap = (winner["score"] - ranked[1]["score"]) / winner["score"]
        if relative_gap < tie_fraction:
            return {
                "status": "REVIEW_REQUIRED",
                "selected_l": None,
                "score": None,
                "top_l": winner["l"],
                "runner_up_l": ranked[1]["l"],
                "relative_score_gap": relative_gap,
            }
    return {
        "status": "UNIQUE_SHELL_SELECTED",
        "selected_l": winner["l"],
        "score": winner["score"],
        "leading_eigenvalue": winner["leading_eigenvalue"],
        "added_ao": winner["added_ao"],
    }


def append_leading_mode(coefficients, element, l, mode):
    if element not in coefficients or not 0 <= l < len(coefficients[element]):
        raise ValueError("requested response channel is absent")
    mode = torch.as_tensor(mode, dtype=torch.float64).reshape(-1, 1)
    source = coefficients[element][l]
    if mode.shape[0] != source.shape[0] or not bool(torch.all(torch.isfinite(mode))):
        raise ValueError("leading residual mode has an invalid radial dimension")
    candidate = {
        name: [channel.detach().clone() for channel in channels]
        for name, channels in coefficients.items()
    }
    candidate[element][l] = torch.cat((source.detach().clone(), mode), dim=1)
    for index, channel in enumerate(coefficients[element]):
        expected = channel.shape[1] + (1 if index == l else 0)
        if candidate[element][index].shape[1] != expected:
            raise RuntimeError("residual seed changed an unexpected shell count")
        if index != l and not torch.equal(candidate[element][index], channel):
            raise RuntimeError("residual seed changed an existing channel")
        if index == l and not torch.equal(
            candidate[element][index][:, : channel.shape[1]], channel
        ):
            raise RuntimeError("residual seed changed existing coefficient columns")
    return candidate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--atom-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--magnetic-overlap-tolerance", type=float, default=3.0e-4)
    parser.add_argument("--condition-limit", type=float, default=1.0e12)
    parser.add_argument("--tie-fraction", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    coefficient_path = args.coefficients.resolve()
    target_path = args.atom_target.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    for path in (coefficient_path, target_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"input must be a nonempty regular file: {path}")
    if sha256(target_path) != EXPECTED_TARGET_SHA256:
        raise ValueError("atomic target SHA256 does not match the accepted FD8 target")
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    coefficients = read_optimizer_coefficients(
        coefficient_path,
        element="C",
        radial_rows=RADIAL_ROWS,
        max_l=4,
        expected_nu=EXPECTED_NU,
    )
    fixed_specs = tuple(
        {"element": "C", "l": l, "zeta": zeta + 1}
        for l, channel in enumerate(coefficients["C"])
        for zeta in range(channel.shape[1])
    )
    data = read_sternheimer(target_path)
    spectra = {}
    records = []
    for l in range(5):
        spectrum = radial_residual_spectrum_many(
            (data,),
            coefficients,
            fixed_specs,
            "C",
            l,
            relative_rank_tolerance=args.relative_rank_tolerance,
            magnetic_overlap_tolerance=args.magnetic_overlap_tolerance,
            condition_limit=args.condition_limit,
        )
        spectra[l] = spectrum
        records.append(spectrum_record(spectrum, l=l))

    selection = select_channel(records, tie_fraction=args.tie_fraction)
    payload = {
        "format_version": 1,
        "scope": "atomic residual-shell seed, not a validated basis",
        "source_coefficients": str(coefficient_path),
        "source_coefficients_sha256": sha256(coefficient_path),
        "source_target": str(target_path),
        "source_target_sha256": sha256(target_path),
        "source_nu": list(EXPECTED_NU),
        "relative_rank_tolerance": args.relative_rank_tolerance,
        "magnetic_overlap_tolerance": args.magnetic_overlap_tolerance,
        "condition_limit": args.condition_limit,
        "tie_fraction": args.tie_fraction,
        "channels": records,
        "selection": selection,
    }
    if selection["status"] == "UNIQUE_SHELL_SELECTED":
        selected_l = selection["selected_l"]
        leading_mode = spectra[selected_l].coefficients[:, 0]
        candidate = append_leading_mode(coefficients, "C", selected_l, leading_mode)
        write_optimizer_coefficients(output_path, candidate)
        next_nu = list(EXPECTED_NU)
        next_nu[selected_l] += 1
        reloaded = read_optimizer_coefficients(
            output_path,
            element="C",
            radial_rows=RADIAL_ROWS,
            max_l=4,
            expected_nu=tuple(next_nu),
        )
        for l, source in enumerate(coefficients["C"]):
            if not torch.equal(reloaded["C"][l][:, : source.shape[1]], source):
                raise RuntimeError("written seed did not preserve existing columns")
        payload["seed"] = {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "nu": next_nu,
        }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
