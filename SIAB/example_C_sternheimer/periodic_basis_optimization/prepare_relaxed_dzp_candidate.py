#!/usr/bin/env python3
"""Stage a relaxed 3s3p2d C candidate after independent low-cost gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


NU = [3, 3, 2, 0, 0]
FIXED_NU = [1, 1, 0, 0, 0]
AO_COUNT_ATOM = 22


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def prepare_candidate(
    *,
    optimizer_result_path: Path,
    comparison_path: Path,
    original_spectrum_path: Path,
    candidate_spectrum_path: Path,
    orbital_path: Path,
    output_directory: Path,
) -> dict:
    optimizer_result_path = Path(optimizer_result_path).resolve(strict=True)
    comparison_path = Path(comparison_path).resolve(strict=True)
    original_spectrum_path = Path(original_spectrum_path).resolve(strict=True)
    candidate_spectrum_path = Path(candidate_spectrum_path).resolve(strict=True)
    orbital_path = Path(orbital_path).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)

    optimizer = json.loads(optimizer_result_path.read_text(encoding="ascii"))
    if optimizer.get("format_version") != 1:
        raise ValueError("optimizer result must use format version 1")
    if optimizer.get("nu") != NU or optimizer.get("fixed_nu") != FIXED_NU:
        raise ValueError("optimizer layout is not relaxed DZP")
    if optimizer.get("occupied_capture_reference") != "initial_candidate":
        raise ValueError("occupied capture must be referenced to the initial TZDP candidate")
    for family in ("C_atom", "C_solid"):
        initial = _finite(optimizer["initial_family_losses"][family], f"initial {family} loss")
        best = _finite(optimizer["best_family_losses"][family], f"best {family} loss")
        if best >= initial:
            raise ValueError(f"{family} loss did not improve")
    coefficients = Path(optimizer["output_coefficients"]).resolve(strict=True)
    coefficient_sha = sha256(coefficients)
    if coefficient_sha != optimizer.get("output_coefficients_sha256"):
        raise ValueError("optimizer output coefficient SHA256 mismatch")
    if coefficient_sha != optimizer.get("best_checkpoint_sha256"):
        raise ValueError("final coefficients do not equal the best checkpoint")

    comparison = json.loads(comparison_path.read_text(encoding="ascii"))
    if comparison.get("format_version") != 1:
        raise ValueError("q3 comparison must use format version 1")
    if len(comparison.get("datasets", [])) != 1 or comparison["datasets"][0].get("selected_iq") != 43:
        raise ValueError("comparison is not the independent q3 gate")
    records = {record.get("label"): record for record in comparison.get("candidates", [])}
    if set(records) != {"original-tzdp", "relaxed-dzp"}:
        raise ValueError("comparison must contain original and relaxed DZP exactly once")
    original = records["original-tzdp"]
    relaxed = records["relaxed-dzp"]
    if relaxed.get("nu") != NU or relaxed.get("ao_count_cell") != 2 * AO_COUNT_ATOM:
        raise ValueError("relaxed q3 candidate has the wrong AO layout")
    if relaxed.get("coefficients_sha256") != coefficient_sha:
        raise ValueError("q3 comparison used different coefficients")
    for metric in (
        "global_weighted_relative_pi_error",
        "global_weighted_relative_trace_log_error",
    ):
        if _finite(relaxed[metric], f"relaxed {metric}") >= _finite(
            original[metric], f"original {metric}"
        ):
            raise ValueError(f"relaxed DZP does not improve q3 {metric}")
    capture = _finite(relaxed["minimum_occupied_capture"], "q3 occupied capture")
    if capture < 0.9998982409775239:
        raise ValueError("relaxed DZP fails the occupied-capture safety floor")

    original_spectrum = json.loads(original_spectrum_path.read_text(encoding="ascii"))
    candidate_spectrum = json.loads(candidate_spectrum_path.read_text(encoding="ascii"))
    for report, label in ((original_spectrum, "original-tzdp"), (candidate_spectrum, "relaxed-dzp")):
        if report.get("format_version") != 1 or report.get("label") != label:
            raise ValueError(f"invalid {label} spectrum report")
        if report.get("nu") != NU or report.get("ao_count_cell") != 2 * AO_COUNT_ATOM:
            raise ValueError(f"invalid {label} spectrum AO layout")
    condition_ratio = _finite(
        candidate_spectrum["maximum_overlap_condition"], "candidate overlap condition"
    ) / _finite(original_spectrum["maximum_overlap_condition"], "original overlap condition")
    eigenvalue_ratio = _finite(
        candidate_spectrum["maximum_eigenvalue_ev"], "candidate maximum eigenvalue"
    ) / _finite(original_spectrum["maximum_eigenvalue_ev"], "original maximum eigenvalue")
    if condition_ratio >= 3.0 or eigenvalue_ratio >= 1.5:
        raise ValueError("relaxed DZP fails overlap or virtual-spectrum gate")

    orbital_text = orbital_path.read_text(encoding="ascii")
    for marker in (
        "Energy Cutoff(Ry)           100.0",
        "Radius Cutoff(a.u.)         10.0",
        "Lmax                        2",
        "Number of Sorbital-->       3",
        "Number of Porbital-->       3",
        "Number of Dorbital-->       2",
    ):
        if marker not in orbital_text:
            raise ValueError(f"exported orbital is missing marker: {marker}")

    output_directory.mkdir(parents=True)
    orbital_filename = "C_gga_10au_100Ry_3s3p2d_relaxed_response.orb"
    staged_orbital = output_directory / orbital_filename
    shutil.copyfile(orbital_path, staged_orbital)
    payload = {
        "status": "success",
        "format_version": 1,
        "profile": "relaxed_dzp",
        "label": "relaxed-dzp",
        "nu": NU,
        "fixed_nu": FIXED_NU,
        "ao_count_atom": AO_COUNT_ATOM,
        "ao_count_cell": 2 * AO_COUNT_ATOM,
        "coefficients": str(coefficients),
        "coefficients_sha256": coefficient_sha,
        "orbital_filename": orbital_filename,
        "orbital_sha256": sha256(staged_orbital),
        "heldout_q3_pi_error": _finite(relaxed["global_weighted_relative_pi_error"], "q3 Pi error"),
        "heldout_q3_trace_log_error": _finite(
            relaxed["global_weighted_relative_trace_log_error"], "q3 trace-log error"
        ),
        "minimum_occupied_capture": capture,
        "maximum_overlap_condition_ratio": condition_ratio,
        "maximum_eigenvalue_ratio": eigenvalue_ratio,
        "pre_pbe_gate": "pass",
        "pbe_reference_basis": "original_unoptimized_sg15_tzdp",
        "pbe_tolerance_ev": 0.010,
        "ordinary_sos_binding_tolerance_ev_per_c": 0.1,
        "auxiliary_basis_rule": "product_pca_threshold_only_1e-4",
    }
    manifest = output_directory / "CANDIDATE.json"
    _atomic_json(manifest, payload)
    (output_directory / "provenance.txt").write_text(
        "status=success\n"
        "purpose=stage_relaxed_dzp_candidate_for_pbe_then_sos\n"
        "pre_pbe_gate=pass\n"
        f"candidate_manifest_sha256={sha256(manifest)}\n"
        f"orbital_sha256={payload['orbital_sha256']}\n",
        encoding="ascii",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-result", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--original-spectrum", required=True, type=Path)
    parser.add_argument("--candidate-spectrum", required=True, type=Path)
    parser.add_argument("--orbital", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_candidate(
        optimizer_result_path=args.optimizer_result,
        comparison_path=args.comparison,
        original_spectrum_path=args.original_spectrum,
        candidate_spectrum_path=args.candidate_spectrum,
        orbital_path=args.orbital,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
