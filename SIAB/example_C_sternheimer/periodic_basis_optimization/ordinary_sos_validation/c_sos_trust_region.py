#!/usr/bin/env python3
"""Audit a constrained local surrogate for C atom-solid SOS basis searches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from diamond_qstar_sos_gate import QSTAR_REPRESENTATIVES, parse_librpa_q_contributions


DEFAULT_DELTA_REFERENCE_EV_PER_C = 6.902326
DEFAULT_PBE_TOLERANCE_EV = 0.010


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite(value, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


def load_candidate(
    spec: dict,
    *,
    pbe_tolerance_ev: float = DEFAULT_PBE_TOLERANCE_EV,
    delta_reference_ev_per_c: float = DEFAULT_DELTA_REFERENCE_EV_PER_C,
) -> dict:
    name = str(spec.get("name", "")).strip()
    if not name:
        raise ValueError("candidate name is required")
    stability = str(spec.get("stability", "stable"))
    if stability not in {"stable", "unstable"}:
        raise ValueError(f"invalid stability label for {name}: {stability}")
    coordinates = spec.get("coordinates")
    if not isinstance(coordinates, dict):
        raise ValueError(f"coordinates must be an object for {name}")
    parsed_coordinates = {
        str(key): _finite(value, label=f"coordinate {name}.{key}")
        for key, value in coordinates.items()
    }

    pbe_path = Path(spec["pbe_gate"]).resolve(strict=True)
    binding_path = Path(spec["binding_result"]).resolve(strict=True)
    pbe = _read_json(pbe_path)
    binding = _read_json(binding_path)
    if pbe.get("status") != "success" or pbe.get("pbe_gate") != "pass":
        raise ValueError(f"PBE gate is not successful for {name}")
    if binding.get("status") != "success":
        raise ValueError(f"binding result is not successful for {name}")

    pbe_differences = {
        "atom_ev": _finite(pbe["atom_energy_difference_ev"], label=f"{name} atom PBE difference"),
        "solid_per_c_ev": _finite(
            pbe["solid_energy_difference_ev_per_c"],
            label=f"{name} solid PBE difference",
        ),
        "binding_per_c_ev": _finite(
            pbe["binding_energy_difference_ev_per_c"],
            label=f"{name} binding PBE difference",
        ),
    }
    maximum_pbe_difference = max(abs(value) for value in pbe_differences.values())
    pbe_pass = maximum_pbe_difference <= float(pbe_tolerance_ev)
    if stability == "stable" and not pbe_pass:
        raise ValueError(
            f"claimed stable candidate {name} violates the PBE {pbe_tolerance_ev:g} eV gate"
        )

    zero_order = _finite(
        binding["zero_order_binding_ev_per_c"],
        label=f"{name} zero-order binding",
    )
    correlation = _finite(
        binding["correlation_binding_ev_per_c"],
        label=f"{name} correlation binding",
    )
    total = _finite(binding["sos_total_binding_ev_per_c"], label=f"{name} SOS binding")
    if not math.isclose(zero_order + correlation, total, rel_tol=0.0, abs_tol=2.0e-9):
        raise ValueError(f"binding components do not reconstruct the total for {name}")
    signed_error = total - float(delta_reference_ev_per_c)

    result = {
        "name": name,
        "coordinates": parsed_coordinates,
        "stability": stability,
        "pbe_pass": pbe_pass,
        "pbe_differences_ev": pbe_differences,
        "maximum_abs_pbe_difference_ev": maximum_pbe_difference,
        "zero_order_binding_ev_per_c": zero_order,
        "correlation_binding_ev_per_c": correlation,
        "sos_total_binding_ev_per_c": total,
        "sos_error_ev_per_c": signed_error,
        "absolute_sos_error_ev_per_c": abs(signed_error),
        "selected_orbital_sha256": str(binding.get("selected_orbital_sha256", "")),
        "input_paths": {
            "pbe_gate": str(pbe_path),
            "binding_result": str(binding_path),
        },
        "input_sha256": {
            "pbe_gate": _sha256(pbe_path),
            "binding_result": _sha256(binding_path),
        },
    }

    if spec.get("solid_librpa_output"):
        solid_path = Path(spec["solid_librpa_output"]).resolve(strict=True)
        parsed = parse_librpa_q_contributions(solid_path)
        contributions = parsed.pop("q_contributions_ha")
        result["qstar_weighted_contributions_ha"] = {
            f"q{q_index}": contributions[q_index - 1].real * multiplicity
            for q_index, multiplicity in QSTAR_REPRESENTATIVES
        }
        result["solid_librpa"] = parsed
        result["input_paths"]["solid_librpa_output"] = str(solid_path)
        result["input_sha256"]["solid_librpa_output"] = parsed["sha256"]

    if spec.get("frequency_decomposition"):
        decomposition_path = Path(spec["frequency_decomposition"]).resolve(strict=True)
        result["high_frequency_tail"] = parse_frequency_decomposition(decomposition_path)
        result["input_paths"]["frequency_decomposition"] = str(decomposition_path)
        result["input_sha256"]["frequency_decomposition"] = _sha256(decomposition_path)
    return result


def _term_value(coordinates: dict[str, float], term: dict) -> float:
    powers = term.get("powers")
    if not isinstance(powers, dict):
        raise ValueError(f"term powers must be an object: {term}")
    value = 1.0
    for coordinate, power in powers.items():
        integer_power = int(power)
        if integer_power != power or integer_power < 0:
            raise ValueError(f"term powers must be nonnegative integers: {term}")
        value *= coordinates.get(str(coordinate), 0.0) ** integer_power
    return value


def _matrix_rank(matrix: list[list[float]], *, tolerance: float = 1.0e-12) -> int:
    if not matrix:
        return 0
    work = [list(map(float, row)) for row in matrix]
    rows = len(work)
    columns = len(work[0])
    rank = 0
    for column in range(columns):
        pivot = max(range(rank, rows), key=lambda row: abs(work[row][column]), default=rank)
        if pivot >= rows or abs(work[pivot][column]) <= tolerance:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for item in range(column, columns):
            work[rank][item] /= pivot_value
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column]
            if abs(factor) <= tolerance:
                continue
            for item in range(column, columns):
                work[row][item] -= factor * work[rank][item]
        rank += 1
        if rank == rows:
            break
    return rank


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    work = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= 1.0e-14:
            raise ValueError("singular normal equation")
        work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        for item in range(column, size + 1):
            work[column][item] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            for item in range(column, size + 1):
                work[row][item] -= factor * work[column][item]
    return [work[row][-1] for row in range(size)]


def _least_squares(matrix: list[list[float]], vector: list[float]) -> list[float]:
    columns = len(matrix[0])
    normal = [
        [sum(row[left] * row[right] for row in matrix) for right in range(columns)]
        for left in range(columns)
    ]
    rhs = [sum(row[column] * value for row, value in zip(matrix, vector)) for column in range(columns)]
    return _solve(normal, rhs)


def _predict(row: list[float], coefficients: list[float]) -> float:
    return sum(value * coefficient for value, coefficient in zip(row, coefficients))


def _rank_concordance(actual: list[float], predicted: list[float]) -> float:
    concordant = 0
    comparable = 0
    for left in range(len(actual)):
        for right in range(left + 1, len(actual)):
            actual_difference = actual[left] - actual[right]
            predicted_difference = predicted[left] - predicted[right]
            if abs(actual_difference) <= 1.0e-14 or abs(predicted_difference) <= 1.0e-14:
                continue
            comparable += 1
            if actual_difference * predicted_difference > 0.0:
                concordant += 1
    return 1.0 if comparable == 0 else concordant / comparable


def assess_surrogate(
    rows: list[dict],
    *,
    terms: list[dict],
    minimum_rank_concordance: float,
) -> dict:
    if not terms or any(not str(term.get("name", "")).strip() for term in terms):
        raise ValueError("surrogate terms must have names")
    stable = [row for row in rows if row.get("stability") == "stable"]
    unstable = [str(row["name"]) for row in rows if row.get("stability") == "unstable"]
    design = [[_term_value(row["coordinates"], term) for term in terms] for row in stable]
    targets = [_finite(row["sos_error_ev_per_c"], label=f'{row["name"]} SOS error') for row in stable]
    term_count = len(terms)
    full_rank = _matrix_rank(design)
    failure_reasons = []
    if len(stable) <= term_count:
        failure_reasons.append("insufficient_redundant_points")
    if full_rank < term_count:
        failure_reasons.append("full_design_rank_deficient")

    loo_predictions = []
    loo_ranks = []
    if full_rank == term_count:
        for holdout in range(len(stable)):
            training_design = [row for index, row in enumerate(design) if index != holdout]
            training_targets = [value for index, value in enumerate(targets) if index != holdout]
            rank = _matrix_rank(training_design)
            loo_ranks.append(rank)
            if rank < term_count:
                loo_predictions.append(None)
                continue
            coefficients = _least_squares(training_design, training_targets)
            loo_predictions.append(_predict(design[holdout], coefficients))
    if loo_ranks and min(loo_ranks) < term_count:
        failure_reasons.append("leave_one_out_rank_deficient")

    result = {
        "stable_point_count": len(stable),
        "unstable_excluded": unstable,
        "term_names": [str(term["name"]) for term in terms],
        "term_count": term_count,
        "full_design_rank": full_rank,
        "leave_one_out_ranks": loo_ranks,
        "failure_reasons": failure_reasons,
    }
    if loo_predictions and all(value is not None for value in loo_predictions):
        predictions = [float(value) for value in loo_predictions]
        mae = sum(abs(predicted - actual) for predicted, actual in zip(predictions, targets)) / len(targets)
        concordance = _rank_concordance(targets, predictions)
        result["loo_predictions_ev_per_c"] = {
            row["name"]: value for row, value in zip(stable, predictions)
        }
        result["loo_mae_ev_per_c"] = mae
        result["loo_rank_concordance"] = concordance
        if concordance < float(minimum_rank_concordance):
            failure_reasons.append("leave_one_out_ranking_failed")
    result["model_gate"] = "pass" if not failure_reasons else "fail"
    if result["model_gate"] == "pass":
        coefficients = _least_squares(design, targets)
        result["coefficients"] = {
            str(term["name"]): coefficient for term, coefficient in zip(terms, coefficients)
        }
    return result


def parse_frequency_decomposition(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    records = {2: [], 6: []}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        fields = line.split("\t")
        if not fields or fields[0] != "RECORD":
            continue
        if len(fields) != 13:
            raise ValueError(f"expected 13 RECORD fields at {path}:{line_number}")
        q_index = int(fields[2])
        if q_index not in records:
            continue
        frequency_index = int(fields[4])
        trace = _finite(fields[7], label=f"trace at {path}:{line_number}")
        logdet = _finite(fields[8], label=f"logdet at {path}:{line_number}")
        raw = _finite(fields[9], label=f"raw at {path}:{line_number}")
        weighted_raw = _finite(fields[12], label=f"weighted raw at {path}:{line_number}")
        if not math.isclose(trace + logdet, raw, rel_tol=0.0, abs_tol=2.0e-10):
            raise ValueError(f"trace/logdet split does not reconstruct raw at {path}:{line_number}")
        records[q_index].append(
            {
                "frequency_index": frequency_index,
                "trace": trace,
                "raw": raw,
                "weighted_raw": weighted_raw,
            }
        )

    metrics = {}
    counts = set()
    for q_index, values in records.items():
        values.sort(key=lambda value: value["frequency_index"])
        expected = list(range(len(values)))
        actual = [value["frequency_index"] for value in values]
        if not values or actual != expected:
            raise ValueError(f"incomplete or duplicate frequency indices for q{q_index}: {path}")
        counts.add(len(values))
        highest = values[-1]
        trace_scale = abs(highest["trace"])
        cancellation_ratio = abs(highest["raw"]) / trace_scale if trace_scale else math.inf
        absolute_weighted_sum = sum(abs(value["weighted_raw"]) for value in values)
        tail_fraction = (
            abs(highest["weighted_raw"]) / absolute_weighted_sum
            if absolute_weighted_sum
            else math.inf
        )
        metrics[f"q{q_index}"] = {
            "highest_frequency_cancellation_ratio": cancellation_ratio,
            "high_frequency_tail_fraction": tail_fraction,
            "highest_frequency_raw": highest["raw"],
            "highest_frequency_trace": highest["trace"],
        }
    if len(counts) != 1:
        raise ValueError(f"q2/q6 frequency counts differ: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "frequency_count_per_q": counts.pop(),
        **metrics,
    }


def calibrate_high_frequency_tail(
    stable_metrics: list[dict],
    *,
    minimum_stable_points: int = 3,
    safety_factor: float = 3.0,
) -> dict:
    if len(stable_metrics) < int(minimum_stable_points):
        return {
            "calibration_gate": "insufficient",
            "stable_point_count": len(stable_metrics),
            "minimum_stable_points": int(minimum_stable_points),
            "reason": "thresholds require multiple independently stable candidates",
        }
    thresholds = {}
    for q_label in ("q2", "q6"):
        thresholds[q_label] = {}
        for metric_name in (
            "highest_frequency_cancellation_ratio",
            "high_frequency_tail_fraction",
        ):
            maximum = max(_finite(item[q_label][metric_name], label=metric_name) for item in stable_metrics)
            thresholds[q_label][metric_name] = maximum * float(safety_factor)
    return {
        "calibration_gate": "pass",
        "stable_point_count": len(stable_metrics),
        "minimum_stable_points": int(minimum_stable_points),
        "safety_factor": float(safety_factor),
        "thresholds": thresholds,
    }


def analyze_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _read_json(manifest_path)
    delta_reference = _finite(
        manifest.get("delta_st_reference_ev_per_c", DEFAULT_DELTA_REFERENCE_EV_PER_C),
        label="Delta-ST reference",
    )
    pbe_tolerance = _finite(
        manifest.get("pbe_tolerance_ev", DEFAULT_PBE_TOLERANCE_EV),
        label="PBE tolerance",
    )
    candidates = [
        load_candidate(
            spec,
            pbe_tolerance_ev=pbe_tolerance,
            delta_reference_ev_per_c=delta_reference,
        )
        for spec in manifest["candidates"]
    ]
    surrogate = assess_surrogate(
        candidates,
        terms=manifest["surrogate_terms"],
        minimum_rank_concordance=_finite(
            manifest.get("minimum_loo_rank_concordance", 0.8),
            label="minimum LOO rank concordance",
        ),
    )
    stable_tail_metrics = [
        candidate["high_frequency_tail"]
        for candidate in candidates
        if candidate["stability"] == "stable" and "high_frequency_tail" in candidate
    ]
    tail_calibration = calibrate_high_frequency_tail(
        stable_tail_metrics,
        minimum_stable_points=int(manifest.get("minimum_tail_calibration_points", 3)),
    )
    return {
        "status": "success",
        "quantity": "c_atom_diamond_sos_local_trust_region_audit",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "delta_st_reference_ev_per_c": delta_reference,
        "pbe_tolerance_ev": pbe_tolerance,
        "candidates": candidates,
        "surrogate": surrogate,
        "high_frequency_tail_calibration": tail_calibration,
        "new_candidate_gate": "pass"
        if surrogate["model_gate"] == "pass" and tail_calibration["calibration_gate"] == "pass"
        else "hold",
        "recommended_action": "evaluate_manifest_proposals"
        if surrogate["model_gate"] == "pass" and tail_calibration["calibration_gate"] == "pass"
        else "do_not_submit_new_physics_candidate",
    }


def _atomic_write(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_artifacts(output_root: Path, result: dict) -> None:
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    _atomic_write(
        output_root / "RESULT.json",
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    lines = [
        "name\tstability\tmax_pbe_ev\tebind0_ev_per_c\tebindc_ev_per_c\t"
        "ebind_sos_ev_per_c\terror_ev_per_c"
    ]
    for row in result["candidates"]:
        lines.append(
            "\t".join(
                (
                    row["name"],
                    row["stability"],
                    f'{row["maximum_abs_pbe_difference_ev"]:.17g}',
                    f'{row["zero_order_binding_ev_per_c"]:.17g}',
                    f'{row["correlation_binding_ev_per_c"]:.17g}',
                    f'{row["sos_total_binding_ev_per_c"]:.17g}',
                    f'{row["sos_error_ev_per_c"]:.17g}',
                )
            )
        )
    _atomic_write(output_root / "DATASET.tsv", "\n".join(lines) + "\n")
    provenance = [
        "status success",
        f'manifest_sha256 {result["manifest_sha256"]}',
        f'manifest_path {result["manifest_path"]}',
    ]
    for candidate in result["candidates"]:
        for label, digest in sorted(candidate["input_sha256"].items()):
            provenance.append(f'input_{candidate["name"]}_{label}_sha256 {digest}')
            provenance.append(
                f'input_{candidate["name"]}_{label}_path {candidate["input_paths"][label]}'
            )
    _atomic_write(output_root / "PROVENANCE.txt", "\n".join(provenance) + "\n")
    _atomic_write(output_root / "STATUS", "success\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = analyze_manifest(args.manifest)
    write_artifacts(args.output_root, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
