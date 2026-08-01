import json
import math
from pathlib import Path
import re

from zero_order_audit import ZeroOrderAudit


_CHECKS = (
    "abacus_finish_marker",
    "charge_grid_exact",
    "final_total_energy_le_1e_12_ha",
    "nbands_exact",
    "new_scf_complete",
    "occupations_le_1e_14",
    "occupied_eigenvalues_le_1e_12_ha",
    "occupied_state_count_exact",
    "old_scf_complete",
    "wavefunction_grid_exact",
)
_FILES = (
    "new_eig_occ",
    "new_running_scf_log",
    "old_eig_occ",
    "old_running_scf_log",
)
_THRESHOLD_LIMITS = (
    ("final_total_energy_abs_diff_ha", 1.0e-12),
    ("occupation_abs_diff", 1.0e-14),
    ("occupied_eigenvalue_abs_diff_ha", 1.0e-12),
)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _read_grid(value, label, expected_case):
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(component) is not int or component <= 0 for component in value)
    ):
        raise ValueError(
            f"zero-order audit {expected_case} {label} is invalid"
        )
    return tuple(value)


def read_zero_order_audit(path, expected_case):
    """Return validated zero-order identity data or raise ValueError."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"zero-order audit {expected_case} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"zero-order audit {expected_case} must be an object")
    if payload.get("format") != "sternheimer_siab_zero_order_identity_v1":
        raise ValueError(f"zero-order audit {expected_case} has wrong format")
    if payload.get("case") != expected_case:
        raise ValueError(f"zero-order audit {expected_case} has wrong case")
    if payload.get("pass") is not True:
        raise ValueError(f"zero-order audit {expected_case} did not pass")

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise ValueError(f"zero-order audit {expected_case} checks are missing")
    for key in _CHECKS:
        if checks.get(key) is not True:
            raise ValueError(
                f"zero-order audit {expected_case} check failed: {key}"
            )

    eig = payload.get("eig_occ_comparison")
    running = payload.get("running_log_comparison")
    if not isinstance(eig, dict) or not isinstance(running, dict):
        raise ValueError(
            f"zero-order audit {expected_case} comparisons are missing"
        )
    occupation_difference = _finite_number(
        eig.get("max_occupation_abs_diff"),
        f"zero-order audit {expected_case} occupation difference",
    )
    eigenvalue_difference = _finite_number(
        eig.get("max_occupied_eigenvalue_abs_diff_ha"),
        f"zero-order audit {expected_case} eigenvalue difference",
    )
    energy_difference = _finite_number(
        running.get("final_total_energy_abs_diff_ha"),
        f"zero-order audit {expected_case} energy difference",
    )
    if occupation_difference > 1.0e-14:
        raise ValueError(
            f"zero-order audit {expected_case} occupation difference exceeds 1e-14"
        )
    if eigenvalue_difference > 1.0e-12:
        raise ValueError(
            f"zero-order audit {expected_case} eigenvalue difference exceeds 1e-12 Ha"
        )
    if energy_difference > 1.0e-12:
        raise ValueError(
            f"zero-order audit {expected_case} energy difference exceeds 1e-12 Ha"
        )
    occupied_count = eig.get("occupied_state_count")
    if type(occupied_count) is not int or occupied_count <= 0:
        raise ValueError(
            f"zero-order audit {expected_case} occupied_state_count is invalid"
        )

    charge_grid = _read_grid(running.get("charge_grid"), "charge_grid", expected_case)
    wavefunction_grid = _read_grid(
        running.get("wavefunction_grid"), "wavefunction_grid", expected_case
    )
    if charge_grid != wavefunction_grid:
        raise ValueError(
            f"zero-order audit {expected_case} charge and wavefunction grids differ"
        )

    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"zero-order audit {expected_case} files are missing")
    source_file_paths = []
    source_file_sha256 = []
    for key in _FILES:
        record = files.get(key)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not record["path"].strip()
            or not _HASH_PATTERN.fullmatch(str(record.get("sha256", "")))
        ):
            raise ValueError(
                f"zero-order audit {expected_case} file record is invalid: {key}"
            )
        source_file_paths.append((key, record["path"]))
        source_file_sha256.append((key, record["sha256"]))

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(
            f"zero-order audit {expected_case} thresholds are missing"
        )
    validated_thresholds = []
    for key, maximum in _THRESHOLD_LIMITS:
        value = _finite_number(
            thresholds.get(key),
            f"zero-order audit {expected_case} threshold {key}",
        )
        if value <= 0.0 or value > maximum:
            raise ValueError(
                f"zero-order audit {expected_case} threshold is too loose: {key}"
            )
        validated_thresholds.append((key, value))

    return ZeroOrderAudit(
        case=expected_case,
        passed=True,
        occupied_state_count=occupied_count,
        grid=charge_grid,
        max_occupation_abs_diff=occupation_difference,
        max_occupied_eigenvalue_abs_diff_ha=eigenvalue_difference,
        final_total_energy_abs_diff_ha=energy_difference,
        source_file_paths=tuple(source_file_paths),
        source_file_sha256=tuple(source_file_sha256),
        thresholds=tuple(validated_thresholds),
    )
