#!/usr/bin/env python3
"""Stage one named periodic C candidate after independent physics gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


FIXED_NU = [2, 2, 1, 0, 0]
SUPPORTED = {
    "joint-two-g": {
        "nu": [3, 3, 2, 1, 2],
        "ao_count_cell": 94,
        "g_count": 2,
        "overlap_factor": 10.0,
        "eigenvalue_factor": 1.1,
    },
    "one-g-keep-g1": {
        "nu": [3, 3, 2, 1, 1],
        "ao_count_cell": 76,
        "g_count": 1,
        "overlap_factor": 3.0,
        "eigenvalue_factor": 1.1,
    },
    "joint-atom-solid": {
        "nu": [3, 3, 2, 1, 1],
        "ao_count_cell": 76,
        "g_count": 1,
        "overlap_factor": 3.0,
        "eigenvalue_factor": 1.1,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
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
    comparison_path: Path,
    spectrum_path: Path,
    orbital_path: Path,
    output_directory: Path,
    label: str,
    occupied_capture_floor: float,
    reference_overlap_condition: float,
    reference_maximum_eigenvalue_ev: float,
) -> dict:
    comparison_path = Path(comparison_path).resolve(strict=True)
    spectrum_path = Path(spectrum_path).resolve(strict=True)
    orbital_path = Path(orbital_path).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)
    if label not in SUPPORTED:
        raise ValueError(f"unsupported validation candidate: {label}")

    report = json.loads(comparison_path.read_text(encoding="ascii"))
    if report.get("format_version") != 1:
        raise ValueError("comparison report must use format version 1")
    datasets = report.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or datasets[0].get("selected_iq") != 43:
        raise ValueError("candidate preparation requires the independent q3 report")
    matches = [candidate for candidate in report.get("candidates", []) if candidate.get("label") == label]
    if len(matches) != 1:
        raise ValueError("comparison report must contain the named candidate exactly once")
    candidate = matches[0]
    contract = SUPPORTED[label]
    expected_nu = contract["nu"]
    expected_ao_count = contract["ao_count_cell"]
    expected_g = contract["g_count"]
    if candidate.get("nu") != expected_nu or candidate.get("ao_count_cell") != expected_ao_count:
        raise ValueError("named candidate layout does not match the supported layout")

    coefficients = Path(candidate["coefficients"]).resolve(strict=True)
    if sha256(coefficients) != candidate.get("coefficients_sha256"):
        raise ValueError("candidate coefficient SHA256 mismatch")
    capture = _finite(candidate.get("minimum_occupied_capture"), "minimum occupied capture")
    floor = _finite(occupied_capture_floor, "occupied capture floor")
    if capture < floor:
        raise ValueError("candidate fails the fixed-prefix occupied-capture gate")

    spectrum = json.loads(spectrum_path.read_text(encoding="ascii"))
    if spectrum.get("format_version", 1) != 1 or spectrum.get("label", label) != label:
        raise ValueError("spectrum diagnostic does not identify the named candidate")
    if spectrum.get("ao_count_cell", expected_ao_count) != expected_ao_count:
        raise ValueError("spectrum diagnostic AO count mismatch")
    overlap = _finite(spectrum.get("maximum_overlap_condition"), "maximum overlap condition")
    maximum_eigenvalue = _finite(spectrum.get("maximum_eigenvalue_ev"), "maximum eigenvalue")
    reported_overlap = _finite(candidate.get("maximum_overlap_condition"), "reported overlap condition")
    if not math.isclose(overlap, reported_overlap, rel_tol=1.0e-7, abs_tol=1.0e-6):
        raise ValueError("held-out and spectrum overlap conditions do not match")
    overlap_reference = _finite(reference_overlap_condition, "reference overlap condition")
    eigenvalue_reference = _finite(
        reference_maximum_eigenvalue_ev, "reference maximum eigenvalue"
    )
    overlap_ratio = overlap / overlap_reference
    eigenvalue_ratio = maximum_eigenvalue / eigenvalue_reference
    if (
        overlap_ratio >= contract["overlap_factor"]
        or eigenvalue_ratio >= contract["eigenvalue_factor"]
    ):
        raise ValueError("candidate fails the pre-SOS physics gate")

    orbital_text = orbital_path.read_text(encoding="ascii")
    for marker in (
        "Energy Cutoff(Ry)           100.0",
        "Radius Cutoff(a.u.)         10.0",
        "Lmax                        4",
        f"Number of Gorbital-->       {expected_g}",
    ):
        if marker not in orbital_text:
            raise ValueError(f"candidate orbital is missing marker: {marker}")

    output_directory.mkdir(parents=True)
    orbital_filename = "C_gga_10au_100Ry_{}.orb".format(label.replace("-", "_"))
    staged_orbital = output_directory / orbital_filename
    shutil.copyfile(orbital_path, staged_orbital)
    payload = {
        "status": "success",
        "format_version": 1,
        "label": label,
        "nu": expected_nu,
        "ao_count_cell": expected_ao_count,
        "ao_count_atom": expected_ao_count // 2,
        "fixed_nu": FIXED_NU,
        "occupied_capture_floor": floor,
        "minimum_occupied_capture": capture,
        "heldout_q3_trace_log_error": _finite(
            candidate.get("global_weighted_relative_trace_log_error"),
            "held-out trace-log error",
        ),
        "heldout_q3_pi_error": _finite(
            candidate.get("global_weighted_relative_pi_error"),
            "held-out Pi error",
        ),
        "maximum_overlap_condition": overlap,
        "reference_overlap_condition": overlap_reference,
        "maximum_overlap_condition_ratio": overlap_ratio,
        "maximum_overlap_condition_ratio_limit": contract["overlap_factor"],
        "maximum_eigenvalue_ev": maximum_eigenvalue,
        "reference_maximum_eigenvalue_ev": eigenvalue_reference,
        "maximum_eigenvalue_ratio": eigenvalue_ratio,
        "maximum_eigenvalue_ratio_limit": contract["eigenvalue_factor"],
        "pre_sos_gate": "pass",
        "coefficients": str(coefficients),
        "coefficients_sha256": sha256(coefficients),
        "source_orbital": str(orbital_path),
        "orbital_filename": orbital_filename,
        "exported_orbital": str(staged_orbital),
        "exported_orbital_sha256": sha256(staged_orbital),
        "comparison": str(comparison_path),
        "comparison_sha256": sha256(comparison_path),
        "spectrum": str(spectrum_path),
        "spectrum_sha256": sha256(spectrum_path),
    }
    manifest = output_directory / "CANDIDATE.json"
    _atomic_json(manifest, payload)
    (output_directory / "provenance.txt").write_text(
        "status=success\n"
        "purpose=stage_named_candidate_for_all_band_sos_validation\n"
        f"label={label}\n"
        "pre_sos_gate=pass\n"
        f"candidate_manifest_sha256={sha256(manifest)}\n"
        f"exported_orbital_sha256={payload['exported_orbital_sha256']}\n",
        encoding="ascii",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--spectrum", required=True, type=Path)
    parser.add_argument("--orbital", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--label", default="joint-two-g")
    parser.add_argument("--occupied-capture-floor", required=True, type=float)
    parser.add_argument("--reference-overlap-condition", required=True, type=float)
    parser.add_argument("--reference-maximum-eigenvalue-ev", required=True, type=float)
    args = parser.parse_args()
    payload = prepare_candidate(
        comparison_path=args.comparison,
        spectrum_path=args.spectrum,
        orbital_path=args.orbital,
        output_directory=args.output_directory,
        label=args.label,
        occupied_capture_floor=args.occupied_capture_floor,
        reference_overlap_condition=args.reference_overlap_condition,
        reference_maximum_eigenvalue_ev=args.reference_maximum_eigenvalue_ev,
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
