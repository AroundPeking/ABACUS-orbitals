#!/usr/bin/env python3
"""Screen C bases with an atom-solid differential trace-log proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OPT_DIR))

from compare_periodic_candidates import (  # noqa: E402
    parse_candidate,
    response_metrics,
    sha256,
    validate_occupied_capture_floor,
)
from IO.read_sternheimer import read_sternheimer  # noqa: E402
from IO.read_sternheimer_source import read_sternheimer_source  # noqa: E402
from optimize_periodic_basis import (  # noqa: E402
    validate_atomic_periodic_contract,
    validate_dataset_contract,
)
from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset  # noqa: E402
from periodic_galerkin_optimization import (  # noqa: E402
    evaluate_periodic_galerkin_coefficient_response,
)
from periodic_galerkin_sternheimer import (  # noqa: E402
    prepare_periodic_occupied_reference,
)
from projected_pi import ProjectedPiEvaluator  # noqa: E402
from sternheimer_source_pair import pair_response_and_source  # noqa: E402


PROXY_NAMES = (
    "raw_q_weight",
    "star_partial",
    "star_normalized_extrapolation",
)


def build_differential_proxies(
    *,
    atom_difference,
    solid_differences,
    q_weights,
    star_multiplicities,
    full_q_count,
):
    """Return clearly labeled partial atom-minus-half-solid trace-log proxies."""
    q_indices = set(solid_differences)
    if q_indices != set(q_weights) or q_indices != set(star_multiplicities):
        raise ValueError("solid differences, weights, and stars must use the same q indices")
    if not q_indices:
        raise ValueError("at least one q index is required")
    if not math.isfinite(atom_difference):
        raise ValueError("atom difference must be finite")
    if any(not math.isfinite(value) for value in solid_differences.values()):
        raise ValueError("solid differences must be finite")
    if any(not math.isfinite(value) or value <= 0.0 for value in q_weights.values()):
        raise ValueError("q weights must be finite and positive")
    if any(type(value) is not int or value <= 0 for value in star_multiplicities.values()):
        raise ValueError("star multiplicities must be positive integers")
    included_multiplicity = sum(star_multiplicities.values())
    if type(full_q_count) is not int or full_q_count < included_multiplicity:
        raise ValueError("full q count must cover the included star multiplicities")

    raw_solid = sum(
        q_weights[index] * solid_differences[index] for index in q_indices
    )
    star_sum = sum(
        star_multiplicities[index] * solid_differences[index]
        for index in q_indices
    )
    star_partial_solid = star_sum / full_q_count
    star_normalized_solid = star_sum / included_multiplicity

    def entry(solid_difference, scope):
        return {
            "scope": scope,
            "atom_difference": float(atom_difference),
            "solid_difference": float(solid_difference),
            "atom_minus_half_solid_difference": float(
                atom_difference - 0.5 * solid_difference
            ),
        }

    return {
        "raw_q_weight": entry(
            raw_solid,
            "available q points with their stored individual q weights",
        ),
        "star_partial": entry(
            star_partial_solid,
            "available symmetry stars divided by the full q-grid size",
        ),
        "star_normalized_extrapolation": entry(
            star_normalized_solid,
            "available symmetry stars renormalized to unit solid weight; diagnostic only",
        ),
        "included_star_multiplicity": included_multiplicity,
        "full_q_count": full_q_count,
        "included_star_fraction": included_multiplicity / full_q_count,
    }


def score_proxy_order(candidates, proxy_name):
    """Compare absolute proxy magnitudes with known absolute SOS binding errors."""
    known = [
        candidate
        for candidate in candidates
        if candidate.get("known_sos_binding_error_ev") is not None
    ]
    values = []
    for candidate in known:
        error = float(candidate["known_sos_binding_error_ev"])
        proxy = float(
            candidate["proxies"][proxy_name][
                "atom_minus_half_solid_difference"
            ]
        )
        if not math.isfinite(error) or not math.isfinite(proxy):
            raise ValueError("known SOS errors and proxy values must be finite")
        values.append((candidate["label"], abs(error), abs(proxy)))

    concordant = 0
    discordant = 0
    unresolved = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            known_delta = values[left][1] - values[right][1]
            proxy_delta = values[left][2] - values[right][2]
            if known_delta == 0.0 or proxy_delta == 0.0:
                unresolved += 1
            elif known_delta * proxy_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    resolved = concordant + discordant
    return {
        "candidate_count": len(values),
        "pair_count": concordant + discordant + unresolved,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "unresolved_pairs": unresolved,
        "agreement_fraction": concordant / resolved if resolved else None,
        "known_sos_error_order_small_to_large": [
            value[0] for value in sorted(values, key=lambda item: item[1])
        ],
        "proxy_order_small_to_large": [
            value[0] for value in sorted(values, key=lambda item: item[2])
        ],
    }


def parse_index_value(value, *, name, converter):
    fields = value.split(":", 1)
    if len(fields) != 2:
        raise ValueError(name + " must use q-index:value")
    try:
        index = int(fields[0])
        parsed = converter(fields[1])
    except ValueError as error:
        raise ValueError(name + " must use q-index:value") from error
    if index <= 0:
        raise ValueError(name + " q index must be positive")
    return index, parsed


def parse_known_error(value):
    fields = value.split(":", 1)
    if len(fields) != 2 or not fields[0].strip():
        raise ValueError("known SOS error must use label:value")
    try:
        error = float(fields[1])
    except ValueError as exception:
        raise ValueError("known SOS error must use label:value") from exception
    if not math.isfinite(error):
        raise ValueError("known SOS error must be finite")
    return fields[0].strip(), error


def hash_numeric_sequences(*sequences):
    payload = json.dumps(
        [[float(value) for value in sequence] for sequence in sequences],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument(
        "--q-star-multiplicity",
        action="append",
        required=True,
        help="symmetry-star multiplicity as q-index:value",
    )
    parser.add_argument("--full-q-count", type=int, default=64)
    parser.add_argument("--atomic-response", type=Path, required=True)
    parser.add_argument("--atomic-source", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--known-sos-binding-error-ev", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--occupied-capture-floor", type=float, default=0.9998)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    occupied_capture_floor = validate_occupied_capture_floor(
        args.occupied_capture_floor
    )
    if args.radial_rows <= 0:
        raise ValueError("radial rows must be positive")

    dataset_paths = tuple(path.resolve() for path in args.dataset)
    candidate_specs = tuple(parse_candidate(value) for value in args.candidate)
    if len({label for label, _, _ in candidate_specs}) != len(candidate_specs):
        raise ValueError("candidate labels must be unique")
    known_errors = dict(parse_known_error(value) for value in args.known_sos_binding_error_ev)
    unknown_error_labels = set(known_errors) - {label for label, _, _ in candidate_specs}
    if unknown_error_labels:
        raise ValueError("known SOS error label has no candidate")
    star_multiplicities = dict(
        parse_index_value(value, name="q-star multiplicity", converter=int)
        for value in args.q_star_multiplicity
    )

    for path in dataset_paths:
        if not path.is_dir() or path.is_symlink():
            raise ValueError("each periodic dataset must be a real directory")
    for _, path, _ in candidate_specs:
        if not path.is_file() or path.is_symlink():
            raise ValueError("each candidate must be a real file")
    atomic_response_path = args.atomic_response.resolve()
    atomic_source_path = args.atomic_source.resolve()
    for path in (atomic_response_path, atomic_source_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError("atomic response/source must be nonempty regular files")

    datasets = tuple(
        prepare_periodic_occupied_reference(
            read_periodic_galerkin_dataset(
                path,
                include_reference_projection=False,
            )
        )
        for path in dataset_paths
    )
    validate_dataset_contract(datasets)
    q_indices = {dataset.selected_iq for dataset in datasets}
    if len(q_indices) != len(datasets):
        raise ValueError("periodic datasets must use distinct q indices")
    if q_indices != set(star_multiplicities):
        raise ValueError("q-star multiplicities must cover exactly the periodic datasets")

    atomic_response = read_sternheimer(atomic_response_path)
    atomic_source = read_sternheimer_source(atomic_source_path)
    atomic_pair = pair_response_and_source(atomic_response, atomic_source)
    validate_atomic_periodic_contract(
        atomic_response,
        datasets,
        element=args.element,
        radial_rows=args.radial_rows,
    )
    atomic_evaluator = ProjectedPiEvaluator(atomic_pair)

    candidates = []
    for label, path, nu in candidate_specs:
        coefficients = read_periodic_optimizer_coefficients(
            path,
            element=args.element,
            radial_rows=args.radial_rows,
            expected_nu=nu,
        )
        atomic_result = atomic_evaluator.evaluate(coefficients)
        atomic_metrics = response_metrics(
            atomic_result.candidate_pi,
            atomic_result.reference_pi,
            atomic_result.frequency_weight,
        )
        solid_differences = {}
        q_weights = {}
        solid_reports = []
        minimum_capture = math.inf
        maximum_condition = 1.0
        for dataset in datasets:
            result = evaluate_periodic_galerkin_coefficient_response(
                dataset,
                coefficients,
                contraction_backend="dense",
                occupied_capture_tolerance=1.0 - occupied_capture_floor,
            )
            metrics = response_metrics(
                result.response,
                dataset.reference_response,
                dataset.frequency_weights_ha,
            )
            index = dataset.selected_iq
            solid_differences[index] = metrics["integrated_trace_log_candidate"] - metrics[
                "integrated_trace_log_reference"
            ]
            q_weights[index] = dataset.q_weight
            minimum_capture = min(minimum_capture, result.minimum_occupied_capture)
            maximum_condition = max(maximum_condition, result.maximum_overlap_condition)
            solid_reports.append(
                {
                    "selected_iq": index,
                    "qpoint": [float(value) for value in dataset.qpoint],
                    "q_weight": dataset.q_weight,
                    "star_multiplicity": star_multiplicities[index],
                    "physics_hash": dataset.physics_hash,
                    "frequency_grid_sha256": hash_numeric_sequences(
                        dataset.frequency_ha,
                        dataset.frequency_weights_ha,
                    ),
                    **metrics,
                }
            )

        atom_difference = (
            atomic_metrics["integrated_trace_log_candidate"]
            - atomic_metrics["integrated_trace_log_reference"]
        )
        proxies = build_differential_proxies(
            atom_difference=atom_difference,
            solid_differences=solid_differences,
            q_weights=q_weights,
            star_multiplicities=star_multiplicities,
            full_q_count=args.full_q_count,
        )
        candidates.append(
            {
                "label": label,
                "coefficients": str(path),
                "coefficients_sha256": sha256(path),
                "nu": list(nu),
                "known_sos_binding_error_ev": known_errors.get(label),
                "minimum_periodic_occupied_capture": minimum_capture,
                "maximum_periodic_overlap_condition": maximum_condition,
                "atomic_maximum_overlap_condition": float(
                    atomic_result.max_candidate_condition
                ),
                "atomic": {
                    "frequency_grid_sha256": hash_numeric_sequences(
                        atomic_result.frequency_ha,
                        atomic_result.frequency_weight,
                    ),
                    **atomic_metrics,
                },
                "solid": sorted(
                    solid_reports,
                    key=lambda report: report["selected_iq"],
                ),
                "proxies": proxies,
            }
        )

    ranking = {
        name: score_proxy_order(candidates, name) for name in PROXY_NAMES
    }
    report = {
        "format_version": 1,
        "scope": (
            "offline Galerkin atom-solid differential trace-log diagnosis; "
            "not an RPA binding energy"
        ),
        "limitations": (
            "only the supplied solid q stars are represented; star-normalized "
            "values are extrapolations and require ranking validation"
        ),
        "atomic_response": str(atomic_response_path),
        "atomic_response_sha256": sha256(atomic_response_path),
        "atomic_source": str(atomic_source_path),
        "atomic_source_sha256": sha256(atomic_source_path),
        "periodic_datasets": [
            {
                "path": str(path),
                "selected_iq": dataset.selected_iq,
                "qpoint": [float(value) for value in dataset.qpoint],
                "q_weight": dataset.q_weight,
                "star_multiplicity": star_multiplicities[dataset.selected_iq],
                "physics_hash": dataset.physics_hash,
            }
            for path, dataset in zip(dataset_paths, datasets)
        ],
        "occupied_capture_floor": occupied_capture_floor,
        "full_q_count": args.full_q_count,
        "candidates": candidates,
        "ranking_validation": ranking,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
