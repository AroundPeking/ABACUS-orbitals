#!/usr/bin/env python3
"""Optimize a compact C basis against one or more exact periodic Pi datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_DIR))

from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset  # noqa: E402
from periodic_galerkin_fit import optimize_periodic_galerkin_basis  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_commit(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError("siab commit must be a full 40-character hexadecimal hash")
    return value.lower()


def parse_nu(value, *, max_l):
    try:
        counts = tuple(int(field.strip()) for field in value.split(","))
    except ValueError as error:
        raise ValueError("nu must be a comma-separated integer list") from error
    if len(counts) != max_l + 1 or any(count < 0 for count in counts):
        raise ValueError("nu must define one nonnegative count per angular channel")
    if not any(counts):
        raise ValueError("nu must define a nonempty candidate basis")
    return counts


def parse_channel_counts(value, candidate_nu):
    try:
        counts = tuple(int(field.strip()) for field in value.split(","))
    except ValueError as error:
        raise ValueError("fixed counts must be comma-separated integers") from error
    if len(counts) != len(candidate_nu) or any(count < 0 for count in counts):
        raise ValueError("fixed counts must match the candidate angular channels")
    if any(fixed > total for fixed, total in zip(counts, candidate_nu)):
        raise ValueError("fixed radial prefix exceeds candidate counts")
    if counts == tuple(candidate_nu):
        raise ValueError("fixed radial prefix leaves no optimization variable")
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--siab-commit", required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--nu", default="3,3,2,0,0")
    parser.add_argument("--fixed-nu", default="2,2,1,0,0")
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--max-l", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--minimum-steps", type=int, default=200)
    parser.add_argument("--plateau-patience", type=int, default=300)
    parser.add_argument(
        "--plateau-relative-improvement", type=float, default=1.0e-6
    )
    parser.add_argument("--maximum-backtracks", type=int, default=20)
    parser.add_argument(
        "--occupied-capture-degradation-tolerance",
        type=float,
        default=1.0e-8,
    )
    return parser.parse_args(argv)


def validate_dataset_contract(datasets):
    reference = datasets[0]
    fields = (
        "abacus_commit",
        "executable_sha256",
        "orbital_sha256",
        "pseudopotential_sha256",
        "auxiliary_basis_sha256",
        "primitive_blocks_sha256",
        "primitive_count",
        "raw_auxiliary_dimension",
        "primitive_blocks",
    )
    for dataset in datasets[1:]:
        if any(getattr(dataset, field) != getattr(reference, field) for field in fields):
            raise ValueError("periodic datasets do not share one basis/provenance contract")


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def main(argv=None):
    args = parse_args(argv)
    siab_commit = validate_commit(args.siab_commit)
    nu = parse_nu(args.nu, max_l=args.max_l)
    fixed_nu = parse_channel_counts(args.fixed_nu, nu)
    if args.radial_rows <= 0:
        raise ValueError("radial_rows must be positive")
    initial_path = args.initial.resolve()
    output = args.output_directory.resolve()
    dataset_paths = tuple(path.resolve() for path in args.dataset)
    if output.exists():
        raise FileExistsError(output)
    if (
        not initial_path.is_file()
        or initial_path.is_symlink()
        or initial_path.stat().st_size == 0
    ):
        raise ValueError("initial coefficients must be a nonempty regular file")
    if any(not path.is_dir() or path.is_symlink() for path in dataset_paths):
        raise ValueError("each dataset must be a real directory")

    datasets = tuple(
        read_periodic_galerkin_dataset(path, include_reference_projection=False)
        for path in dataset_paths
    )
    validate_dataset_contract(datasets)
    initial = read_periodic_optimizer_coefficients(
        initial_path,
        element=args.element,
        radial_rows=args.radial_rows,
        expected_nu=nu,
    )
    output.mkdir(parents=True)
    status_path = output / "STATUS.json"
    history_path = output / "OPTIMIZATION_HISTORY.jsonl"
    result_path = output / "OPTIMIZATION_RESULT.json"
    coefficient_path = output / "ORBITAL_RESULTS.txt"
    _write_json(
        status_path,
        {
            "status": "running",
            "siab_commit": siab_commit,
            "dataset_physics_hashes": [dataset.physics_hash for dataset in datasets],
        },
    )

    try:
        with history_path.open("x", encoding="ascii", buffering=1) as history_stream:
            def record_progress(record):
                history_stream.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )
                print(
                    "step={step} loss={loss:.12e} pi={relative_pi_error:.6e} "
                    "capture={minimum_occupied_capture:.12e} "
                    "condition={maximum_overlap_condition:.6e} "
                    "lr={learning_rate:.3e} "
                    "backtracks={backtracks_from_previous_step}".format(**record),
                    flush=True,
                )

            fit = optimize_periodic_galerkin_basis(
                datasets,
                initial,
                fixed_nu={args.element: fixed_nu},
                learning_rate=args.learning_rate,
                max_steps=args.max_steps,
                minimum_steps=args.minimum_steps,
                plateau_patience=args.plateau_patience,
                plateau_relative_improvement=args.plateau_relative_improvement,
                maximum_backtracks=args.maximum_backtracks,
                occupied_capture_degradation_tolerance=(
                    args.occupied_capture_degradation_tolerance
                ),
                progress_callback=record_progress,
            )
        for l, count in enumerate(fixed_nu):
            if not bool(
                fit.coefficients[args.element][l][:, :count].equal(
                    initial[args.element][l][:, :count]
                )
            ):
                raise RuntimeError("fixed radial prefix changed during optimization")
        write_periodic_optimizer_coefficients(coefficient_path, fit.coefficients)
        restored = read_periodic_optimizer_coefficients(
            coefficient_path,
            element=args.element,
            radial_rows=args.radial_rows,
            expected_nu=nu,
        )
        for l, count in enumerate(fixed_nu):
            if not bool(
                restored[args.element][l][:, :count].equal(
                    initial[args.element][l][:, :count]
                )
            ):
                raise RuntimeError("written output changed the fixed radial prefix")

        payload = {
            "format_version": 1,
            "scope": "Galerkin exact-Pi optimization; independent SOS validation required",
            "siab_commit": siab_commit,
            "abacus_commit": datasets[0].abacus_commit,
            "dataset_paths": [str(path) for path in dataset_paths],
            "dataset_physics_hashes": [dataset.physics_hash for dataset in datasets],
            "initial_coefficients": str(initial_path),
            "initial_coefficients_sha256": sha256(initial_path),
            "output_coefficients": str(coefficient_path),
            "output_coefficients_sha256": sha256(coefficient_path),
            "history_sha256": sha256(history_path),
            "nu": list(nu),
            "fixed_nu": list(fixed_nu),
            "initial_loss": fit.initial_loss,
            "initial_relative_pi_error": math.sqrt(fit.initial_loss),
            "best_loss": fit.best_loss,
            "best_relative_pi_error": math.sqrt(fit.best_loss),
            "best_step": fit.best_step,
            "steps_completed": fit.steps_completed,
            "stop_reason": fit.stop_reason,
            "learning_rate": args.learning_rate,
            "maximum_steps": args.max_steps,
            "minimum_steps": args.minimum_steps,
            "plateau_patience": args.plateau_patience,
            "plateau_relative_improvement": args.plateau_relative_improvement,
            "maximum_backtracks": args.maximum_backtracks,
            "occupied_capture_degradation_tolerance": (
                args.occupied_capture_degradation_tolerance
            ),
            "initial_minimum_occupied_capture": (
                fit.initial_minimum_occupied_capture
            ),
            "occupied_capture_floor": fit.occupied_capture_floor,
            "total_backtracks": fit.total_backtracks,
            "final_learning_rate": fit.final_learning_rate,
        }
        _write_json(result_path, payload)
        _write_json(
            status_path,
            {
                "status": "success",
                "result_sha256": sha256(result_path),
                "output_coefficients_sha256": sha256(coefficient_path),
            },
        )
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    except Exception as error:
        _write_json(
            status_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
