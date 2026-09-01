#!/usr/bin/env python3
"""Calibrate a direct atom-solid ordinary-SOS proxy from selected q stars."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from diamond_qstar_sos_gate import (
    PRINTED_SUM_TOLERANCE_HA,
    QSTAR_REPRESENTATIVES,
    parse_librpa_q_contributions,
)


HARTREE_TO_EV = 27.211386245988


def _finite(value, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


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


def _read_summary(path: Path) -> dict[str, str]:
    path = Path(path).resolve(strict=True)
    values = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            values[key] = value.strip()
    if values.get("status") != "success":
        raise ValueError(f"summary is not successful: {path}")
    return values


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
    rhs = [
        sum(row[column] * value for row, value in zip(matrix, vector))
        for column in range(columns)
    ]
    return _solve(normal, rhs)


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


def _design_row(row: dict, q_indices: tuple[int, ...]) -> list[float]:
    contributions = row.get("qstar_weighted_contributions_ha")
    if not isinstance(contributions, dict):
        raise ValueError(f'missing q-star contributions for {row.get("name", "candidate")}')
    return [1.0] + [
        _finite(
            contributions.get(index, contributions.get(str(index))),
            label=f'{row.get("name", "candidate")} q{index} contribution',
        )
        for index in q_indices
    ]


def _predict_total(row: dict, predicted_solid_ecrpa_ha: float) -> float:
    zero_order = _finite(
        row["zero_order_binding_ev_per_c"],
        label=f'{row.get("name", "candidate")} zero-order binding',
    )
    atom_ecrpa = _finite(
        row["atom_ecrpa_ha"],
        label=f'{row.get("name", "candidate")} atom EcRPA',
    )
    return zero_order + (atom_ecrpa - 0.5 * predicted_solid_ecrpa_ha) * HARTREE_TO_EV


def assess_differential_proxy(
    rows: list[dict],
    *,
    q_indices: tuple[int, ...],
    maximum_loo_error_ev_per_c: float,
    minimum_rank_concordance: float,
) -> dict:
    q_indices = tuple(int(index) for index in q_indices)
    if not q_indices or len(q_indices) != len(set(q_indices)):
        raise ValueError("q indices must be nonempty and unique")
    known_q_indices = {index for index, _ in QSTAR_REPRESENTATIVES}
    if not set(q_indices).issubset(known_q_indices):
        raise ValueError("proxy q indices must be diamond q-star representatives")
    maximum_error = _finite(maximum_loo_error_ev_per_c, label="maximum LOO error")
    minimum_concordance = _finite(minimum_rank_concordance, label="minimum concordance")
    if maximum_error <= 0.0 or not 0.0 <= minimum_concordance <= 1.0:
        raise ValueError("invalid proxy gate thresholds")
    if not rows:
        raise ValueError("at least one calibration row is required")
    names = [str(row.get("name", "")).strip() for row in rows]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("calibration names must be nonempty and unique")

    design = [_design_row(row, q_indices) for row in rows]
    solid_targets = [
        _finite(row["solid_ecrpa_ha"], label=f'{row["name"]} solid EcRPA')
        for row in rows
    ]
    total_targets = [
        _finite(row["sos_total_binding_ev_per_c"], label=f'{row["name"]} total binding')
        for row in rows
    ]
    coefficient_count = len(q_indices) + 1
    full_rank = _matrix_rank(design)
    failure_reasons = []
    if len(rows) <= coefficient_count:
        failure_reasons.append("insufficient_redundant_points")
    if full_rank < coefficient_count:
        failure_reasons.append("full_design_rank_deficient")

    loo_rows = []
    loo_ranks = []
    if full_rank == coefficient_count and len(rows) > coefficient_count:
        for held_out, row in enumerate(rows):
            train_design = [item for index, item in enumerate(design) if index != held_out]
            train_targets = [item for index, item in enumerate(solid_targets) if index != held_out]
            rank = _matrix_rank(train_design)
            loo_ranks.append(rank)
            if rank < coefficient_count:
                continue
            coefficients = _least_squares(train_design, train_targets)
            predicted_solid = sum(
                coefficient * value
                for coefficient, value in zip(coefficients, design[held_out])
            )
            predicted_total = _predict_total(row, predicted_solid)
            actual_total = total_targets[held_out]
            loo_rows.append(
                {
                    "name": row["name"],
                    "predicted_solid_ecrpa_ha": predicted_solid,
                    "actual_solid_ecrpa_ha": solid_targets[held_out],
                    "predicted_total_binding_ev_per_c": predicted_total,
                    "actual_total_binding_ev_per_c": actual_total,
                    "prediction_error_ev_per_c": predicted_total - actual_total,
                }
            )
    if len(loo_ranks) != len(rows) or any(rank < coefficient_count for rank in loo_ranks):
        failure_reasons.append("leave_one_out_rank_deficient")

    loo_mae = None
    loo_max = None
    loo_concordance = None
    if len(loo_rows) == len(rows):
        errors = [abs(row["prediction_error_ev_per_c"]) for row in loo_rows]
        loo_mae = sum(errors) / len(errors)
        loo_max = max(errors)
        loo_concordance = _rank_concordance(
            [row["actual_total_binding_ev_per_c"] for row in loo_rows],
            [row["predicted_total_binding_ev_per_c"] for row in loo_rows],
        )
        if loo_max > maximum_error:
            failure_reasons.append("loo_error_too_large")
        if loo_concordance < minimum_concordance:
            failure_reasons.append("loo_rank_concordance_too_low")

    coefficients = None
    if full_rank == coefficient_count:
        fitted = _least_squares(design, solid_targets)
        coefficients = {"intercept_ha": fitted[0]}
        coefficients.update(
            {f"q{index}_coefficient": fitted[position + 1] for position, index in enumerate(q_indices)}
        )
    return {
        "status": "success",
        "quantity": "c_atom_diamond_ordinary_sos_differential_qstar_proxy",
        "q_indices_one_based": list(q_indices),
        "calibration_count": len(rows),
        "coefficient_count": coefficient_count,
        "full_design_rank": full_rank,
        "loo_design_ranks": loo_ranks,
        "loo_mae_ev_per_c": loo_mae,
        "loo_max_abs_error_ev_per_c": loo_max,
        "loo_rank_concordance": loo_concordance,
        "maximum_allowed_loo_error_ev_per_c": maximum_error,
        "minimum_required_rank_concordance": minimum_concordance,
        "fit_coefficients": coefficients,
        "failure_reasons": sorted(set(failure_reasons)),
        "proxy_gate": "pass" if not failure_reasons else "fail",
        "leave_one_out": loo_rows,
    }


def load_endpoint(spec: dict) -> dict:
    name = str(spec.get("name", "")).strip()
    if not name:
        raise ValueError("endpoint name is required")
    inputs = {}
    hashes = {}
    if spec.get("binding_result"):
        binding_path = Path(spec["binding_result"]).resolve(strict=True)
        binding = _read_json(binding_path)
        if binding.get("status") != "success":
            raise ValueError(f"binding result is not successful for {name}")
        atom_ecrpa = _finite(binding["atom_ecrpa_ha"], label=f"{name} atom EcRPA")
        solid_ecrpa = _finite(
            binding["solid_ecrpa_qstar_reconstructed_ha"], label=f"{name} solid EcRPA"
        )
        zero_order = _finite(
            binding["zero_order_binding_ev_per_c"], label=f"{name} zero-order binding"
        )
        total = _finite(binding["sos_total_binding_ev_per_c"], label=f"{name} total binding")
        orbital_sha256 = str(binding.get("selected_orbital_sha256", ""))
        inputs["binding_result"] = str(binding_path)
        hashes["binding_result"] = _sha256(binding_path)
    else:
        atom_path = Path(spec["atom_summary"]).resolve(strict=True)
        solid_path = Path(spec["solid_summary"]).resolve(strict=True)
        atom = _read_summary(atom_path)
        solid = _read_summary(solid_path)
        if atom.get("side") != "atom" or solid.get("side") != "solid":
            raise ValueError(f"endpoint summary sides are invalid for {name}")
        orbital_sha256 = atom.get("selected_orbital_sha256", "")
        if not orbital_sha256 or orbital_sha256 != solid.get("selected_orbital_sha256"):
            raise ValueError(f"endpoint orbital hashes do not match for {name}")
        atom_zero = _finite(atom["reference_ha"], label=f"{name} atom zero-order")
        solid_zero = _finite(solid["reference_ha"], label=f"{name} solid zero-order")
        atom_ecrpa = _finite(atom["ecrpa_ha"], label=f"{name} atom EcRPA")
        solid_ecrpa = _finite(solid["ecrpa_ha"], label=f"{name} solid EcRPA")
        zero_order = (atom_zero - 0.5 * solid_zero) * HARTREE_TO_EV
        total = zero_order + (atom_ecrpa - 0.5 * solid_ecrpa) * HARTREE_TO_EV
        inputs.update({"atom_summary": str(atom_path), "solid_summary": str(solid_path)})
        hashes.update({"atom_summary": _sha256(atom_path), "solid_summary": _sha256(solid_path)})

    librpa_path = Path(spec["solid_librpa_output"]).resolve(strict=True)
    parsed = parse_librpa_q_contributions(librpa_path)
    contributions = parsed.pop("q_contributions_ha")
    qstar = {
        index: contributions[index - 1].real * multiplicity
        for index, multiplicity in QSTAR_REPRESENTATIVES
    }
    reconstructed_solid = sum(qstar.values())
    if not math.isclose(
        reconstructed_solid,
        solid_ecrpa,
        rel_tol=0.0,
        abs_tol=PRINTED_SUM_TOLERANCE_HA,
    ):
        raise ValueError(f"q-star solid EcRPA does not match endpoint for {name}")
    inputs["solid_librpa_output"] = str(librpa_path)
    hashes["solid_librpa_output"] = parsed["sha256"]
    return {
        "name": name,
        "selected_orbital_sha256": orbital_sha256,
        "zero_order_binding_ev_per_c": zero_order,
        "atom_ecrpa_ha": atom_ecrpa,
        "solid_ecrpa_ha": solid_ecrpa,
        "sos_total_binding_ev_per_c": total,
        "qstar_weighted_contributions_ha": qstar,
        "input_paths": inputs,
        "input_sha256": hashes,
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = _read_json(manifest_path)
    endpoints = manifest.get("calibration_endpoints")
    if not isinstance(endpoints, list):
        raise ValueError("calibration_endpoints must be a list")
    rows = [load_endpoint(spec) for spec in endpoints]
    result = assess_differential_proxy(
        rows,
        q_indices=tuple(manifest.get("q_indices_one_based", (6, 7, 8))),
        maximum_loo_error_ev_per_c=manifest.get("maximum_loo_error_ev_per_c", 0.01),
        minimum_rank_concordance=manifest.get("minimum_rank_concordance", 0.95),
    )
    result["manifest_path"] = str(manifest_path)
    result["manifest_sha256"] = _sha256(manifest_path)
    result["calibration_endpoints"] = rows
    args.output_root.mkdir(parents=True)
    _atomic_write(
        args.output_root / "RESULT.json",
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(args.output_root / "STATUS", "success\n")
    _atomic_write(
        args.output_root / "provenance.txt",
        "\n".join(
            [
                f"manifest_path {manifest_path}",
                f"manifest_sha256 {_sha256(manifest_path)}",
                f"script_path {Path(__file__).resolve()}",
                f"script_sha256 {_sha256(Path(__file__).resolve())}",
                f"proxy_gate {result['proxy_gate']}",
            ]
        )
        + "\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
