#!/usr/bin/env python3
"""Optimize a compact C basis against frozen periodic Pi or body-RPA data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import sys
import time


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
from IO.read_sternheimer import read_sternheimer  # noqa: E402
from IO.read_sternheimer_source import read_sternheimer_source  # noqa: E402
from projected_pi import ProjectedPiEvaluator  # noqa: E402
from sternheimer_source_pair import pair_response_and_source  # noqa: E402


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
    parser.add_argument("--dataset-family", action="append")
    parser.add_argument("--atomic-response", type=Path)
    parser.add_argument("--atomic-source", type=Path)
    parser.add_argument("--atomic-family", default="C_atom")
    parser.add_argument("--objective", choices=("pi", "rpa"), default="pi")
    parser.add_argument("--allow-partial-q", action="store_true")
    parser.add_argument("--rpa-pi-weight", type=float)
    parser.add_argument("--rpa-trace-log-weight", type=float)
    parser.add_argument("--rpa-energy-weight", type=float)
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
    parser.add_argument("--block-cache-workers", type=int, default=1)
    parser.add_argument(
        "--occupied-capture-reference",
        choices=("initial_candidate", "fixed_prefix"),
        default="initial_candidate",
    )
    parser.add_argument(
        "--omitted-reference-projection-validation",
        choices=("sha256", "layout"),
        default="sha256",
    )
    parser.add_argument(
        "--occupied-capture-degradation-tolerance",
        type=float,
        default=1.0e-8,
    )
    return parser.parse_args(argv)


def validate_objective_options(args):
    supplied = {
        "pi_weight": args.rpa_pi_weight,
        "trace_log_weight": args.rpa_trace_log_weight,
        "energy_weight": args.rpa_energy_weight,
    }
    if args.objective == "pi":
        if args.allow_partial_q or any(value is not None for value in supplied.values()):
            raise ValueError("RPA weights and partial-q flag require objective=rpa")
        return None
    if (args.dataset_family is not None or args.atomic_response is not None
            or args.atomic_source is not None):
        raise ValueError("RPA objective requires one periodic family, without atomic inputs")
    weights = {key: 1.0 if value is None else value for key, value in supplied.items()}
    for key, value in weights.items():
        if not math.isfinite(value) or value < 0 or (key != "energy_weight" and value == 0):
            raise ValueError("invalid RPA objective weight: " + key)
    return weights


def normalize_dataset_families(values, count):
    if type(count) is not int or count <= 0:
        raise ValueError("dataset count must be a positive integer")
    if values is None:
        return ("periodic",) * count
    values = tuple(values)
    if len(values) != count:
        raise ValueError("one --dataset-family is required per --dataset")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("dataset-family names must be nonempty")
    return values


def validate_atomic_pair_options(response, source):
    if (response is None) != (source is None):
        raise ValueError("atomic response and source must be provided together")
    return response is not None


def validate_atomic_periodic_contract(response, datasets, *, element, radial_rows):
    reference = datasets[0]
    provenance = response.provenance
    for field, expected in (
        ("pseudopotential_sha256", reference.pseudopotential_sha256),
        ("orbital_sha256", reference.orbital_sha256),
    ):
        if provenance.get(field) != expected:
            raise ValueError("atomic and periodic response provenance differs: " + field)
    if provenance.get("kernel") != "full_coulomb":
        raise ValueError("atomic response must use the full periodic Poisson kernel")
    if any(
        block.element != element or block.n_primitive != radial_rows
        for block in response.blocks
    ):
        raise ValueError("atomic response primitive blocks do not match the candidate basis")


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


def write_best_checkpoint(output, step, loss, coefficients, *, objective="pi"):
    output = Path(output)
    orbital_path = output / "BEST_ORBITAL_CHECKPOINT.txt"
    metadata_path = output / "BEST_CHECKPOINT.json"
    orbital_temporary = output / ".BEST_ORBITAL_CHECKPOINT.txt.tmp"
    metadata_temporary = output / ".BEST_CHECKPOINT.json.tmp"
    for temporary in (orbital_temporary, metadata_temporary):
        if temporary.exists():
            temporary.unlink()
    write_periodic_optimizer_coefficients(orbital_temporary, coefficients)
    orbital_hash = sha256(orbital_temporary)
    metadata = {
        "format_version": 1,
        "step": int(step),
        "loss": float(loss),
        "orbital_file": orbital_path.name,
        "orbital_sha256": orbital_hash,
    }
    if objective == "pi":
        metadata["relative_pi_error"] = math.sqrt(float(loss))
    else:
        metadata["objective"] = objective
    _write_json(metadata_temporary, metadata)
    orbital_temporary.replace(orbital_path)
    metadata_temporary.replace(metadata_path)


def main(argv=None):
    start = time.perf_counter()
    args = parse_args(argv)
    rpa_weights = validate_objective_options(args)
    siab_commit = validate_commit(args.siab_commit)
    nu = parse_nu(args.nu, max_l=args.max_l)
    fixed_nu = parse_channel_counts(args.fixed_nu, nu)
    if args.radial_rows <= 0:
        raise ValueError("radial_rows must be positive")
    initial_path = args.initial.resolve()
    output = args.output_directory.resolve()
    dataset_paths = tuple(path.resolve() for path in args.dataset)
    dataset_families = normalize_dataset_families(
        args.dataset_family,
        len(dataset_paths),
    )
    has_atomic_pair = validate_atomic_pair_options(
        args.atomic_response,
        args.atomic_source,
    )
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
        read_periodic_galerkin_dataset(
            path,
            include_reference_projection=False,
            verify_omitted_chunks=(
                args.omitted_reference_projection_validation == "sha256"
            ),
        )
        for path in dataset_paths
    )
    validate_dataset_contract(datasets)
    if args.objective == "rpa":
        coverage = math.fsum(dataset.q_weight for dataset in datasets)
        complete_q_weight = math.isclose(coverage, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        if not complete_q_weight and not args.allow_partial_q:
            raise ValueError("partial-q RPA requires explicit --allow-partial-q")
    additional_family_evaluators = {}
    atomic_pair = None
    atomic_paths = None
    if has_atomic_pair:
        atomic_response_path = args.atomic_response.resolve()
        atomic_source_path = args.atomic_source.resolve()
        for path in (atomic_response_path, atomic_source_path):
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                raise ValueError("atomic response/source must be nonempty regular files")
        if (
            not isinstance(args.atomic_family, str)
            or not args.atomic_family.strip()
            or args.atomic_family in set(dataset_families)
        ):
            raise ValueError("atomic family name must be nonempty and distinct")
        atomic_response = read_sternheimer(atomic_response_path)
        atomic_source = read_sternheimer_source(atomic_source_path)
        atomic_pair = pair_response_and_source(atomic_response, atomic_source)
        validate_atomic_periodic_contract(
            atomic_response,
            datasets,
            element=args.element,
            radial_rows=args.radial_rows,
        )
        additional_family_evaluators[args.atomic_family] = ProjectedPiEvaluator(
            atomic_pair,
        )
        atomic_paths = (atomic_response_path, atomic_source_path)
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
    checkpoint_path = output / "BEST_ORBITAL_CHECKPOINT.txt"
    checkpoint_metadata_path = output / "BEST_CHECKPOINT.json"
    _write_json(
        status_path,
        {
            "status": "running",
            "objective": args.objective,
            "siab_commit": siab_commit,
            "dataset_physics_hashes": [dataset.physics_hash for dataset in datasets],
            "dataset_families": list(dataset_families),
            "atomic_family": args.atomic_family if atomic_pair is not None else None,
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

            def record_best(step, loss, coefficients):
                write_best_checkpoint(output, step, loss, coefficients, objective=args.objective)

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
                occupied_capture_reference=args.occupied_capture_reference,
                block_cache_workers=args.block_cache_workers,
                dataset_families=(None if args.objective == "rpa" else dataset_families),
                additional_family_evaluators=additional_family_evaluators,
                objective=args.objective,
                rpa_weights=rpa_weights,
                progress_callback=record_progress,
                best_callback=record_best,
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
        if not checkpoint_metadata_path.is_file():
            raise RuntimeError("best checkpoint metadata is missing")
        checkpoint_metadata = json.loads(
            checkpoint_metadata_path.read_text(encoding="ascii")
        )
        if (
            checkpoint_metadata["step"] != fit.best_step
            or checkpoint_metadata["loss"] != fit.best_loss
            or checkpoint_metadata["orbital_sha256"] != sha256(checkpoint_path)
            or sha256(checkpoint_path) != sha256(coefficient_path)
        ):
            raise RuntimeError("best checkpoint does not match the final result")

        payload = {
            "format_version": 1,
            "scope": "Galerkin exact-Pi optimization; independent SOS validation required",
            "objective": args.objective,
            "siab_commit": siab_commit,
            "abacus_commit": datasets[0].abacus_commit,
            "dataset_paths": [str(path) for path in dataset_paths],
            "dataset_families": list(dataset_families),
            "dataset_physics_hashes": [dataset.physics_hash for dataset in datasets],
            "atomic_family": args.atomic_family if atomic_pair is not None else None,
            "atomic_response": (
                str(atomic_paths[0]) if atomic_paths is not None else None
            ),
            "atomic_response_sha256": (
                sha256(atomic_paths[0]) if atomic_paths is not None else None
            ),
            "atomic_source": (
                str(atomic_paths[1]) if atomic_paths is not None else None
            ),
            "atomic_source_sha256": (
                sha256(atomic_paths[1]) if atomic_paths is not None else None
            ),
            "atomic_provenance_warnings": (
                list(atomic_pair.provenance_warnings)
                if atomic_pair is not None
                else []
            ),
            "initial_coefficients": str(initial_path),
            "initial_coefficients_sha256": sha256(initial_path),
            "output_coefficients": str(coefficient_path),
            "output_coefficients_sha256": sha256(coefficient_path),
            "best_checkpoint": str(checkpoint_path),
            "best_checkpoint_sha256": sha256(checkpoint_path),
            "history_sha256": sha256(history_path),
            "nu": list(nu),
            "fixed_nu": list(fixed_nu),
            "initial_loss": fit.initial_loss,
            "initial_family_losses": fit.initial_family_losses,
            "initial_relative_pi_error": math.sqrt(fit.initial_loss),
            "best_loss": fit.best_loss,
            "best_family_losses": fit.best_family_losses,
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
            "block_cache_workers": args.block_cache_workers,
            "omitted_reference_projection_validation": (
                args.omitted_reference_projection_validation
            ),
            "occupied_capture_reference": fit.occupied_capture_reference,
            "reference_minimum_occupied_capture": (
                fit.reference_minimum_occupied_capture
            ),
            "occupied_capture_degradation_tolerance": (
                args.occupied_capture_degradation_tolerance
            ),
            "initial_minimum_occupied_capture": (
                fit.initial_minimum_occupied_capture
            ),
            "occupied_capture_floor": fit.occupied_capture_floor,
            "total_backtracks": fit.total_backtracks,
            "final_learning_rate": fit.final_learning_rate,
            "elapsed_seconds": time.perf_counter() - start,
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1 if sys.platform == "darwin" else 1024),
        }
        if args.objective == "rpa":
            initial_record = fit.history[0]
            best_record = next(record for record in fit.history if record["step"] == fit.best_step)
            details = best_record["rpa"]
            payload.update(
                scope=("full-q frozen-space body RPA optimization" if details["complete_q_weight"]
                       else "partial-q frozen-space body RPA engineering trial"),
                physical_release_gate="hold",
                objective_weights=fit.objective_weights,
                q_weight_coverage=details["q_weight_coverage"],
                complete_q_weight=details["complete_q_weight"],
                q_weight_normalization="physical_unrenormalized",
                initial_relative_pi_error=initial_record["relative_pi_error"],
                best_relative_pi_error=best_record["relative_pi_error"],
                initial_rpa=initial_record["rpa"],
                best_rpa=details,
            )
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
