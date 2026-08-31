#!/usr/bin/env python3
"""Audit completed C SOS candidates without mixing incompatible basis directions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from pathlib import Path

from c_sos_trust_region import assess_surrogate
from diamond_qstar_sos_gate import QSTAR_REPRESENTATIVES, parse_librpa_q_contributions


DELTA_REFERENCE_EV_PER_C = 6.902326
ACCEPTANCE_TOLERANCE_EV_PER_C = 0.1


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
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


def candidate_family_key(candidate: dict) -> tuple:
    """Return the immutable fields that define one physical interpolation family."""
    return (
        tuple(candidate.get("nu", [])),
        candidate.get("profile"),
        candidate.get("original_coefficients_sha256"),
        candidate.get("optimized_coefficients_sha256"),
        candidate.get("secondary_optimized_coefficients_sha256"),
        candidate.get("direction"),
    )


def _family_contract(key: tuple) -> dict:
    return {
        "nu": list(key[0]),
        "profile": key[1],
        "original_coefficients_sha256": key[2],
        "optimized_coefficients_sha256": key[3],
        "secondary_optimized_coefficients_sha256": key[4],
        "direction": key[5],
    }


def _family_id(key: tuple) -> str:
    encoded = json.dumps(_family_contract(key), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _vector(value, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected vector length {length}: {value!r}")
    result = [_finite(item, label="coordinate") for item in value]
    return result


def extract_coordinates(candidate: dict) -> dict[str, float]:
    direction = candidate.get("direction")
    coordinates: dict[str, float] = {}
    if direction == "original_plus_channel_alpha_times_optimized_minus_original":
        values = _vector(candidate.get("channel_alphas"), 5)
        return {"alpha_s": values[0], "alpha_p": values[1], "alpha_d": values[2]}
    if direction == "original_plus_alpha_times_optimized_minus_original":
        return {"alpha": _finite(candidate.get("alpha"), label="alpha")}
    if direction == "original_plus_two_channel_resolved_directions":
        values = _vector(candidate.get("secondary_channel_alphas"), 5)
        for name, value in zip(("relaxed_s_all", "relaxed_p_all", "relaxed_d_all"), values[:3]):
            if value != 0.0:
                coordinates[name] = value
        return coordinates
    if direction == "original_plus_channel_and_zeta_resolved_directions":
        values = candidate.get("secondary_zeta_alphas")
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError("invalid secondary zeta layout")
        parsed = [_vector(channel, length) for channel, length in zip(values, (3, 3, 2, 0, 0))]
        for name, value in (
            ("beta_s3", parsed[0][2]),
            ("beta_p3", parsed[1][2]),
            ("beta_d2", parsed[2][1]),
        ):
            if value != 0.0:
                coordinates[name] = value
        return coordinates
    return coordinates


def _feature_terms(coordinate_names: list[str]) -> list[dict]:
    terms = []
    for name in coordinate_names:
        terms.append({"name": name, "powers": {name: 1}})
        terms.append({"name": f"{name}^2", "powers": {name: 2}})
    for left, right in itertools.combinations(coordinate_names, 2):
        terms.append({"name": f"{left}*{right}", "powers": {left: 1, right: 1}})
    return terms


def audit_low_order_models(
    rows: list[dict],
    *,
    maximum_loo_mae_ev_per_c: float = 0.05,
) -> dict:
    stable = [row for row in rows if row.get("stability") == "stable"]
    coordinate_names = sorted({name for row in stable for name in row.get("coordinates", {})})
    feature_terms = _feature_terms(coordinate_names)
    maximum_nonconstant_terms = min(3, max(0, len(stable) - 2))
    model_results = []
    for term_count in range(maximum_nonconstant_terms + 1):
        for selected in itertools.combinations(feature_terms, term_count):
            terms = [{"name": "constant", "powers": {}}] + list(selected)
            result = assess_surrogate(rows, terms=terms, minimum_rank_concordance=1.0)
            if result.get("model_gate") == "pass" and result.get(
                "loo_mae_ev_per_c", math.inf
            ) > float(maximum_loo_mae_ev_per_c):
                result["failure_reasons"].append("leave_one_out_mae_too_large")
                result["model_gate"] = "fail"
            model_results.append(result)
    validated = [result for result in model_results if result.get("model_gate") == "pass"]
    failure_reasons = [] if validated else ["no_leave_one_out_valid_model"]
    return {
        "stable_point_count": len(stable),
        "coordinate_names": coordinate_names,
        "maximum_loo_mae_ev_per_c": float(maximum_loo_mae_ev_per_c),
        "tested_model_count": len(model_results),
        "validated_model_count": len(validated),
        "model_gate": "pass" if validated else "fail",
        "failure_reasons": failure_reasons,
        "validated_models": validated,
        "tested_models": model_results,
    }


def _load_qstar(binding: dict) -> dict:
    summary = Path(binding["solid_summary"]).resolve(strict=True)
    qstar_path = summary.parent / "qstar-gate" / "RESULT.json"
    qstar = _read_json(qstar_path)
    if qstar.get("status") != "success" or qstar.get("sparse_qstar_gate") != "pass":
        raise ValueError(f"invalid q-star gate: {qstar_path}")
    rows = qstar.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"expected one q-star row: {qstar_path}")
    librpa_path = Path(rows[0]["path"]).resolve(strict=True)
    parsed = parse_librpa_q_contributions(librpa_path)
    contributions = parsed.pop("q_contributions_ha")
    weighted = {
        f"q{q_index}": contributions[q_index - 1].real * multiplicity
        for q_index, multiplicity in QSTAR_REPRESENTATIVES
    }
    reconstructed = sum(weighted.values())
    expected = _finite(
        binding["solid_ecrpa_qstar_reconstructed_ha"],
        label="solid q-star reconstructed EcRPA",
    )
    if not math.isclose(reconstructed, expected, rel_tol=0.0, abs_tol=5.0e-8):
        raise ValueError(f"q-star reconstruction mismatch: {librpa_path}")
    return {
        "status": "success",
        "weighted_contributions_ha": weighted,
        "reconstructed_ecrpa_ha": reconstructed,
        "qstar_result_path": str(qstar_path.resolve()),
        "qstar_result_sha256": _sha256(qstar_path),
        "librpa_output_path": str(librpa_path),
        "librpa_output_sha256": parsed["sha256"],
    }


def _record_from_inventory(record: dict, trust_stability: dict[str, str]) -> dict:
    candidate_path = Path(record["candidate_path"]).resolve(strict=True)
    candidate = _read_json(candidate_path)
    digest = str(record.get("orbital_sha256", ""))
    pbe = (record.get("pbe") or {}).get("value", {})
    binding = (record.get("binding") or {}).get("value", {})
    complete = bool(record.get("pbe_pass") and record.get("binding_success"))
    if digest in trust_stability:
        stability = trust_stability[digest]
    elif complete:
        stability = "finite_ordinary_sos_tail_not_calibrated"
    elif record.get("pbe") and not record.get("pbe_pass"):
        stability = "pbe_rejected"
    else:
        stability = "not_evaluated"
    result = {
        "name": record["name"],
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "orbital_sha256": digest,
        "coordinates": extract_coordinates(candidate),
        "pbe_pass": bool(record.get("pbe_pass")),
        "binding_success": bool(record.get("binding_success")),
        "stability": stability,
    }
    if record.get("pbe"):
        result["pbe_path"] = record["pbe"]["path"]
        result["pbe_sha256"] = record["pbe"]["sha256"]
    if complete:
        zero_order = _finite(binding["zero_order_binding_ev_per_c"], label="zero-order binding")
        correlation = _finite(binding["correlation_binding_ev_per_c"], label="correlation binding")
        total = _finite(binding["sos_total_binding_ev_per_c"], label="total binding")
        if not math.isclose(zero_order + correlation, total, rel_tol=0.0, abs_tol=2.0e-9):
            raise ValueError(f"binding reconstruction failed for {record['name']}")
        result.update(
            {
                "pbe_differences_ev": {
                    "atom": _finite(pbe["atom_energy_difference_ev"], label="atom PBE difference"),
                    "solid_per_c": _finite(
                        pbe["solid_energy_difference_ev_per_c"],
                        label="solid PBE difference",
                    ),
                    "binding_per_c": _finite(
                        pbe["binding_energy_difference_ev_per_c"],
                        label="binding PBE difference",
                    ),
                },
                "zero_order_binding_ev_per_c": zero_order,
                "correlation_binding_ev_per_c": correlation,
                "sos_total_binding_ev_per_c": total,
                "sos_error_ev_per_c": total - DELTA_REFERENCE_EV_PER_C,
                "binding_path": record["binding"]["path"],
                "binding_sha256": record["binding"]["sha256"],
                "qstar": _load_qstar(binding),
            }
        )
    return result


def build_family_audit(inventory_path: Path, trust_path: Path) -> dict:
    inventory_path = inventory_path.resolve(strict=True)
    trust_path = trust_path.resolve(strict=True)
    inventory = _read_json(inventory_path)
    trust = _read_json(trust_path)
    if inventory.get("status") != "success" or trust.get("status") != "success":
        raise ValueError("input audit status is not success")
    trust_stability = {
        str(row.get("selected_orbital_sha256", "")): str(row["stability"])
        for row in trust["candidates"]
        if row.get("selected_orbital_sha256")
    }
    families: dict[tuple, list[dict]] = {}
    for inventory_record in inventory["records"]:
        candidate = _read_json(Path(inventory_record["candidate_path"]))
        key = candidate_family_key(candidate)
        families.setdefault(key, []).append(_record_from_inventory(inventory_record, trust_stability))

    family_results = []
    for key, records in families.items():
        complete = [record for record in records if record["pbe_pass"] and record["binding_success"]]
        model_rows = [
            {
                "name": record["name"],
                "coordinates": record["coordinates"],
                "stability": "unstable" if record["stability"] == "unstable" else "stable",
                "sos_error_ev_per_c": record["sos_error_ev_per_c"],
            }
            for record in complete
        ]
        if len(model_rows) >= 2 and any(row["coordinates"] for row in model_rows):
            model_audit = audit_low_order_models(model_rows)
        else:
            model_audit = {
                "model_gate": "fail",
                "validated_model_count": 0,
                "failure_reasons": ["insufficient_complete_coordinate_points"],
            }
        family_results.append(
            {
                "family_id": _family_id(key),
                "contract": _family_contract(key),
                "candidate_count": len(records),
                "complete_pbe_sos_count": len(complete),
                "frequency_tail_calibrated_count": sum(
                    record["stability"] in {"stable", "unstable"} for record in complete
                ),
                "model_audit": model_audit,
                "records": sorted(records, key=lambda record: record["name"]),
            }
        )
    family_results.sort(key=lambda family: (-family["complete_pbe_sos_count"], family["family_id"]))
    validated = [
        family["family_id"]
        for family in family_results
        if family["model_audit"].get("model_gate") == "pass"
    ]
    finite_stable = [
        record
        for family in family_results
        for record in family["records"]
        if record.get("binding_success") and record.get("stability") != "unstable"
    ]
    best = min(finite_stable, key=lambda record: record["sos_total_binding_ev_per_c"])
    improvement_needed = best["sos_total_binding_ev_per_c"] - (
        DELTA_REFERENCE_EV_PER_C + ACCEPTANCE_TOLERANCE_EV_PER_C
    )
    return {
        "status": "success",
        "quantity": "c_sos_strict_candidate_family_and_surrogate_audit",
        "delta_st_reference_ev_per_c": DELTA_REFERENCE_EV_PER_C,
        "acceptance_tolerance_ev_per_c": ACCEPTANCE_TOLERANCE_EV_PER_C,
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "trust_path": str(trust_path),
        "trust_sha256": _sha256(trust_path),
        "family_count": len(family_results),
        "families_with_validated_models": validated,
        "strict_existing_data_blocker": not validated,
        "new_candidate_gate": "hold" if not validated else "review",
        "failure_reasons": [] if validated else ["no_strict_family_has_a_leave_one_out_valid_model"],
        "current_best": {
            "name": best["name"],
            "sos_total_binding_ev_per_c": best["sos_total_binding_ev_per_c"],
            "sos_error_ev_per_c": best["sos_error_ev_per_c"],
            "minimum_improvement_to_acceptance_ev_per_c": improvement_needed,
        },
        "families": family_results,
    }


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_audit(output_root: Path, result: dict) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    result_path = output_root / "RESULT.json"
    _atomic_write(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    provenance = {
        "status": "success",
        "quantity": result["quantity"],
        "generator_path": result["generator_path"],
        "generator_sha256": result["generator_sha256"],
        "result_sha256": _sha256(result_path),
        "inventory_path": result["inventory_path"],
        "inventory_sha256": result["inventory_sha256"],
        "trust_path": result["trust_path"],
        "trust_sha256": result["trust_sha256"],
    }
    _atomic_write(output_root / "PROVENANCE.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    _atomic_write(output_root / "STATUS", "success\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--trust-result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = build_family_audit(args.inventory, args.trust_result)
    generator_path = Path(__file__).resolve(strict=True)
    result["generator_path"] = str(generator_path)
    result["generator_sha256"] = _sha256(generator_path)
    write_audit(args.output_root, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
