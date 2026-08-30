#!/usr/bin/env python3
"""Compare several compact bases against one or more exact periodic Pi datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
SIAB_DIR = HERE.parents[1]
OPT_DIR = SIAB_DIR / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OPT_DIR))

from periodic_galerkin_basis import (  # noqa: E402
    build_primitive_to_candidate,
    read_periodic_optimizer_coefficients,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset  # noqa: E402
from periodic_galerkin_optimization import (  # noqa: E402
    evaluate_periodic_galerkin_coefficient_response,
)
from periodic_galerkin_sternheimer import (  # noqa: E402
    prepare_periodic_occupied_reference,
)
from optimize_periodic_basis import parse_nu, validate_dataset_contract  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_candidate(value):
    fields = value.split(":", 2)
    if len(fields) != 3 or not fields[0].strip() or not fields[1].strip():
        raise ValueError("candidate must use label:path:nu")
    label = fields[0].strip()
    path = Path(fields[1].strip())
    nu = parse_nu(fields[2], max_l=4)
    return label, path, nu


def validate_occupied_capture_floor(value):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
        or value >= 1.0
    ):
        raise ValueError("occupied capture floor must be finite in (0, 1)")
    return float(value)


def trace_log_value(response, *, name="response"):
    if response.ndim != 2 or response.shape[0] != response.shape[1]:
        raise ValueError("trace-log response must be square")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("trace-log response name must be nonempty")
    hermitian = 0.5 * (response + response.transpose(-2, -1).conj())
    eigenvalue = torch.linalg.eigvalsh(hermitian)
    argument = 1.0 - eigenvalue
    if not bool(torch.isfinite(argument).all()) or bool(torch.any(argument <= 0.0)):
        minimum_argument = float(torch.min(argument).detach())
        minimum_eigenvalue = float(torch.min(eigenvalue).detach())
        maximum_eigenvalue = float(torch.max(eigenvalue).detach())
        raise RuntimeError(
            f"{name} trace-log argument is not positive: "
            f"minimum_argument={minimum_argument:.17g}, "
            f"minimum_eigenvalue={minimum_eigenvalue:.17g}, "
            f"maximum_eigenvalue={maximum_eigenvalue:.17g}"
        )
    return float(torch.sum(torch.log(argument) + eigenvalue).detach())


def response_metrics(candidate, reference, weights):
    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError("candidate and reference responses must have matching 3-D shapes")
    if weights.ndim != 1 or weights.shape[0] != candidate.shape[0]:
        raise ValueError("frequency weights do not match response count")
    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights <= 0.0)):
        raise ValueError("frequency weights must be finite and positive")

    error_by_frequency = []
    numerator_by_frequency = []
    denominator_by_frequency = []
    trace_candidate = []
    trace_reference = []
    for ifrequency in range(candidate.shape[0]):
        difference = candidate[ifrequency] - reference[ifrequency]
        numerator = float(torch.sum(torch.abs(difference) ** 2).detach())
        denominator = float(torch.sum(torch.abs(reference[ifrequency]) ** 2).detach())
        error_by_frequency.append(
            math.sqrt(numerator / denominator) if denominator > 0.0 else math.sqrt(numerator)
        )
        numerator_by_frequency.append(numerator)
        denominator_by_frequency.append(denominator)
        trace_reference.append(
            trace_log_value(
                reference[ifrequency],
                name=f"reference frequency {ifrequency}",
            )
        )
        trace_candidate.append(
            trace_log_value(
                candidate[ifrequency],
                name=f"candidate frequency {ifrequency}",
            )
        )

    weight = [float(value) for value in weights]
    weighted_numerator = sum(
        value * numerator for value, numerator in zip(weight, numerator_by_frequency)
    )
    weighted_denominator = sum(
        value * denominator for value, denominator in zip(weight, denominator_by_frequency)
    )
    trace_numerator = sum(
        value * (candidate_value - reference_value) ** 2
        for value, candidate_value, reference_value in zip(
            weight, trace_candidate, trace_reference
        )
    )
    trace_denominator = sum(
        value * reference_value ** 2
        for value, reference_value in zip(weight, trace_reference)
    )
    return {
        "relative_pi_error_by_frequency": error_by_frequency,
        "weighted_relative_pi_error": math.sqrt(
            weighted_numerator / weighted_denominator
        ),
        "pi_error_numerator": weighted_numerator,
        "pi_error_denominator": weighted_denominator,
        "trace_log_candidate": trace_candidate,
        "trace_log_reference": trace_reference,
        "weighted_relative_trace_log_error": math.sqrt(
            trace_numerator / trace_denominator
        ),
        "trace_log_error_numerator": trace_numerator,
        "trace_log_error_denominator": trace_denominator,
        "integrated_trace_log_candidate": sum(
            value * item for value, item in zip(weight, trace_candidate)
        ),
        "integrated_trace_log_reference": sum(
            value * item for value, item in zip(weight, trace_reference)
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="candidate definition label:path:nu, where nu has five comma-separated counts",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--element", default="C")
    parser.add_argument("--radial-rows", type=int, default=31)
    parser.add_argument("--occupied-capture-floor", type=float, default=0.999999)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_paths = tuple(path.resolve() for path in args.dataset)
    candidate_specs = tuple(parse_candidate(value) for value in args.candidate)
    occupied_capture_floor = validate_occupied_capture_floor(
        args.occupied_capture_floor
    )
    output = args.output.resolve()
    if args.radial_rows <= 0:
        raise ValueError("radial_rows must be positive")
    if output.exists():
        raise FileExistsError(output)
    if len({label for label, _, _ in candidate_specs}) != len(candidate_specs):
        raise ValueError("candidate labels must be unique")
    if any(not path.is_dir() or path.is_symlink() for path in dataset_paths):
        raise ValueError("each dataset must be a real directory")
    if any(not path.is_file() or path.is_symlink() for _, path, _ in candidate_specs):
        raise ValueError("each candidate coefficient path must be a real file")

    datasets = tuple(
        prepare_periodic_occupied_reference(
            read_periodic_galerkin_dataset(
                path, include_reference_projection=False
            )
        )
        for path in dataset_paths
    )
    validate_dataset_contract(datasets)

    candidates = []
    for label, path, nu in candidate_specs:
        coefficients = read_periodic_optimizer_coefficients(
            path,
            element=args.element,
            radial_rows=args.radial_rows,
            expected_nu=nu,
        )
        basis = build_primitive_to_candidate(
            datasets[0].primitive_blocks,
            datasets[0].primitive_count,
            coefficients,
        )
        dataset_reports = []
        global_pi_numerator = 0.0
        global_pi_denominator = 0.0
        global_trace_numerator = 0.0
        global_trace_denominator = 0.0
        integrated_candidate = 0.0
        integrated_reference = 0.0
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
            q_weight = dataset.q_weight
            global_pi_numerator += q_weight * metrics["pi_error_numerator"]
            global_pi_denominator += q_weight * metrics["pi_error_denominator"]
            global_trace_numerator += q_weight * metrics["trace_log_error_numerator"]
            global_trace_denominator += q_weight * metrics["trace_log_error_denominator"]
            integrated_candidate += q_weight * metrics["integrated_trace_log_candidate"]
            integrated_reference += q_weight * metrics["integrated_trace_log_reference"]
            minimum_capture = min(minimum_capture, result.minimum_occupied_capture)
            maximum_condition = max(maximum_condition, result.maximum_overlap_condition)
            dataset_reports.append(
                {
                    "physics_hash": dataset.physics_hash,
                    "selected_iq": dataset.selected_iq,
                    "qpoint": list(dataset.qpoint),
                    "q_weight": dataset.q_weight,
                    "frequency_ha": [float(value) for value in dataset.frequency_ha],
                    "frequency_weights_ha": [
                        float(value) for value in dataset.frequency_weights_ha
                    ],
                    **metrics,
                }
            )
        candidates.append(
            {
                "label": label,
                "coefficients": str(path.resolve()),
                "coefficients_sha256": sha256(path),
                "nu": list(nu),
                "ao_count_cell": int(basis.transform.shape[1]),
                "minimum_occupied_capture": minimum_capture,
                "maximum_overlap_condition": maximum_condition,
                "global_weighted_relative_pi_error": math.sqrt(
                    global_pi_numerator / global_pi_denominator
                ),
                "global_weighted_relative_trace_log_error": math.sqrt(
                    global_trace_numerator / global_trace_denominator
                ),
                "integrated_trace_log_candidate": integrated_candidate,
                "integrated_trace_log_reference": integrated_reference,
                "integrated_trace_log_difference": (
                    integrated_candidate - integrated_reference
                ),
                "datasets": dataset_reports,
            }
        )

    report = {
        "format_version": 1,
        "scope": "Galerkin Pi and trace-log screening; independent SOS validation required",
        "occupied_capture_floor": occupied_capture_floor,
        "datasets": [
            {
                "path": str(path),
                "physics_hash": dataset.physics_hash,
                "selected_iq": dataset.selected_iq,
                "whitened_auxiliary_rank": dataset.whitened_auxiliary_rank,
            }
            for path, dataset in zip(dataset_paths, datasets)
        ],
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
