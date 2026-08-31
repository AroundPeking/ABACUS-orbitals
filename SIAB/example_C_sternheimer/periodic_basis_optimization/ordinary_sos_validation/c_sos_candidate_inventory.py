#!/usr/bin/env python3
"""Inventory completed C SOS candidates against one exact trust-region subspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


DEFAULT_CONTRACT = {
    "nu": [3, 3, 2, 0, 0],
    "profile": "interpolated_dzp",
    "primary_channel_alphas": [0.5, -1.0, -1.5, 0.0, 0.0],
    "original_coefficients_sha256":
        "bc5fd37b5e01b2745812e7583b54642c6489f986b7eaf097185dbe1bb8a62bbe",
    "primary_optimized_coefficients_sha256":
        "6a7545b4ac8a7b3004aeed8f274eeda8966c310125099d2783e5c191297ffe9c",
    "secondary_optimized_coefficients_sha256":
        "1a7730c6492810c5a01379badb5394df22bd720de7b19743135960135549bb6b",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _vector(value, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected vector of length {length}: {value!r}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"non-finite vector: {value!r}")
    return result


def _all_zero(values: list[float]) -> bool:
    return all(value == 0.0 for value in values)


def classify_candidate(candidate: dict, contract: dict = DEFAULT_CONTRACT) -> dict:
    if candidate.get("status") != "success":
        return {"in_current_subspace": False, "reason": "candidate_status_not_success"}
    if candidate.get("profile") != contract["profile"]:
        return {"in_current_subspace": False, "reason": "different_profile"}
    if list(candidate.get("nu", [])) != contract["nu"]:
        return {"in_current_subspace": False, "reason": "different_ao_layout"}
    if candidate.get("original_coefficients_sha256") != contract["original_coefficients_sha256"]:
        return {"in_current_subspace": False, "reason": "different_original_coefficients"}
    if candidate.get("optimized_coefficients_sha256") != contract[
        "primary_optimized_coefficients_sha256"
    ]:
        return {"in_current_subspace": False, "reason": "different_primary_direction"}

    direction = candidate.get("direction")
    if direction not in {
        "original_plus_channel_alpha_times_optimized_minus_original",
        "original_plus_two_channel_resolved_directions",
        "original_plus_channel_and_zeta_resolved_directions",
    }:
        return {"in_current_subspace": False, "reason": "unsupported_direction_definition"}
    if _vector(candidate.get("channel_alphas"), 5) != contract["primary_channel_alphas"]:
        return {"in_current_subspace": False, "reason": "different_primary_coordinates"}

    coordinates: dict[str, float] = {}
    if direction == "original_plus_channel_alpha_times_optimized_minus_original":
        return {"in_current_subspace": True, "reason": "exact_origin", "coordinates": coordinates}

    if candidate.get("secondary_optimized_coefficients_sha256") != contract[
        "secondary_optimized_coefficients_sha256"
    ]:
        return {"in_current_subspace": False, "reason": "different_secondary_direction"}

    if direction == "original_plus_two_channel_resolved_directions":
        secondary = _vector(candidate.get("secondary_channel_alphas"), 5)
        if secondary[0] != 0.0 or not _all_zero(secondary[3:]):
            return {"in_current_subspace": False, "reason": "unsupported_secondary_channel"}
        if secondary[1] != 0.0:
            coordinates["relaxed_p_all"] = secondary[1]
        if secondary[2] != 0.0:
            coordinates["relaxed_d_all"] = secondary[2]
        return {"in_current_subspace": True, "reason": "exact_secondary_channel", "coordinates": coordinates}

    zeta = candidate.get("secondary_zeta_alphas")
    if not isinstance(zeta, list) or len(zeta) != 5:
        return {"in_current_subspace": False, "reason": "invalid_secondary_zeta_layout"}
    expected_lengths = (3, 3, 2, 0, 0)
    parsed = [_vector(channel, length) for channel, length in zip(zeta, expected_lengths)]
    allowed = {(0, 2): "beta_s3", (1, 2): "beta_p3", (2, 1): "beta_d2"}
    for channel_index, channel in enumerate(parsed):
        for zeta_index, value in enumerate(channel):
            if value == 0.0:
                continue
            coordinate = allowed.get((channel_index, zeta_index))
            if coordinate is None:
                return {"in_current_subspace": False, "reason": "unsupported_secondary_zeta"}
            coordinates[coordinate] = value
    return {"in_current_subspace": True, "reason": "exact_zeta", "coordinates": coordinates}


def _selected_hash_from_provenance(path: Path) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("selected_orbital_sha256="):
            return line.split("=", 1)[1].strip()
    return ""


def _index_outputs(runs_root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    pbe_by_hash: dict[str, dict] = {}
    binding_by_hash: dict[str, dict] = {}
    for path in runs_root.glob("*/pbe-gate/PBE_GATE.json"):
        value = _read_json(path)
        digest = str(value.get("selected_orbital_sha256", "")) or _selected_hash_from_provenance(
            path.parent / "provenance.txt"
        )
        if digest:
            pbe_by_hash[digest] = {"path": str(path.resolve()), "sha256": _sha256(path), "value": value}
    for path in runs_root.glob("*/binding/RESULT.json"):
        value = _read_json(path)
        digest = str(value.get("selected_orbital_sha256", ""))
        if digest:
            binding_by_hash[digest] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "value": value,
            }
    return pbe_by_hash, binding_by_hash


def build_inventory(runs_root: Path, trust_result_path: Path) -> dict:
    runs_root = runs_root.resolve(strict=True)
    trust_result_path = trust_result_path.resolve(strict=True)
    trust = _read_json(trust_result_path)
    trusted_hashes = {
        str(candidate.get("selected_orbital_sha256", ""))
        for candidate in trust["candidates"]
        if candidate.get("selected_orbital_sha256")
    }
    pbe_by_hash, binding_by_hash = _index_outputs(runs_root)
    records = []
    for path in sorted(runs_root.glob("*-candidate-*/CANDIDATE.json")):
        candidate = _read_json(path)
        digest = str(candidate.get("orbital_sha256", ""))
        classification = classify_candidate(candidate)
        pbe = pbe_by_hash.get(digest)
        binding = binding_by_hash.get(digest)
        pbe_pass = bool(pbe and pbe["value"].get("status") == "success" and pbe["value"].get("pbe_gate") == "pass")
        binding_success = bool(binding and binding["value"].get("status") == "success")
        records.append(
            {
                "name": path.parent.name,
                "candidate_path": str(path.resolve()),
                "candidate_sha256": _sha256(path),
                "orbital_sha256": digest,
                **classification,
                "pbe_pass": pbe_pass,
                "pbe": pbe,
                "binding_success": binding_success,
                "binding": binding,
                "already_in_trust_dataset": digest in trusted_hashes,
                "eligible_extra_stable_point": bool(
                    classification["in_current_subspace"]
                    and pbe_pass
                    and binding_success
                    and digest not in trusted_hashes
                ),
            }
        )

    stable_coordinates = [
        candidate["coordinates"]
        for candidate in trust["candidates"]
        if candidate.get("stability") == "stable"
    ]
    coordinate_names = sorted({key for values in stable_coordinates for key in values})
    sampling = {
        name: sorted({float(values[name]) for values in stable_coordinates if values.get(name, 0.0) != 0.0})
        for name in coordinate_names
    }
    eligible = [record["name"] for record in records if record["eligible_extra_stable_point"]]
    return {
        "status": "success",
        "quantity": "c_sos_existing_candidate_inventory",
        "runs_root": str(runs_root),
        "trust_result_path": str(trust_result_path),
        "trust_result_sha256": _sha256(trust_result_path),
        "trusted_orbital_count": len(trusted_hashes),
        "coordinate_nonzero_amplitudes": sampling,
        "candidate_count": len(records),
        "eligible_extra_stable_points": eligible,
        "strict_redundancy_blocker": not eligible and trust["surrogate"]["model_gate"] != "pass",
        "records": records,
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


def write_inventory(output_root: Path, result: dict) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    _atomic_write(output_root / "RESULT.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    _atomic_write(output_root / "STATUS", "success\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--trust-result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = build_inventory(args.runs_root, args.trust_result)
    write_inventory(args.output_root, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
