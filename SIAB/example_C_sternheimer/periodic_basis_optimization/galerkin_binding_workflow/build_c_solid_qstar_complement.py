#!/usr/bin/env python3
"""Build a hash-locked contract for missing diamond-C q-star datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_c_solid_all_q_candidate import DIAMOND_QSTAR_CONTRACT  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_hash(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def build_complement_contract(
    inventory,
    *,
    inventory_sha256,
    source_commit,
    measured_q_wall_minutes,
    nodes_per_q,
    storage_gib_per_q,
    direct_reference=None,
):
    if not _valid_hash(inventory_sha256, 64) or not _valid_hash(source_commit, 40):
        raise ValueError("invalid source or inventory hash")
    if isinstance(inventory, dict):
        if (
            inventory.get("coverage") != "reduced"
            or inventory.get("dataset_contract_gate") != "pass"
            or inventory.get("physical_release_gate") != "hold"
            or inventory.get("q_count") != 64
            or not isinstance(inventory.get("datasets"), list)
        ):
            raise ValueError("invalid reduced dataset inventory envelope")
        inventory = inventory["datasets"]
    if not isinstance(inventory, list):
        raise ValueError("dataset inventory must be a list or validated envelope")
    known = {record["label"]: record for record in DIAMOND_QSTAR_CONTRACT}
    present = []
    for record in inventory:
        label = record.get("logical_qstar_label")
        expected = known.get(label)
        if expected is None or label in present:
            raise ValueError("inventory has an unknown or duplicate q-star label")
        if (
            record.get("selected_iq") != expected["selected_iq"]
            or record.get("multiplicity") != expected["multiplicity"]
            or not math.isclose(
                float(record.get("q_weight")),
                expected["multiplicity"] / 64.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not _valid_hash(record.get("physics_hash"), 64)
        ):
            raise ValueError("inventory does not match the diamond q-star contract")
        present.append(label)
    expected_order = [record["label"] for record in DIAMOND_QSTAR_CONTRACT]
    if present != [label for label in expected_order if label in present]:
        raise ValueError("inventory q-star labels are not in canonical order")

    wall = tuple(float(value) for value in measured_q_wall_minutes)
    if (
        not wall
        or any(not math.isfinite(value) or value <= 0.0 for value in wall)
        or type(nodes_per_q) is not int
        or nodes_per_q <= 0
        or not math.isfinite(storage_gib_per_q)
        or storage_gib_per_q <= 0.0
    ):
        raise ValueError("invalid measured resource calibration")

    missing = [record for record in DIAMOND_QSTAR_CONTRACT if record["label"] not in present]
    reference = Path(direct_reference) if direct_reference is not None else None
    reference_exists = reference is not None and reference.is_file()
    missing_count = len(missing)
    wall_min = min(wall)
    wall_max = max(wall)
    return {
        "status": "success",
        "scope": "diamond_c_solid_missing_qstar_contract",
        "source_commit": source_commit,
        "dataset_inventory_sha256": inventory_sha256,
        "present_logical_qstars": present,
        "present_multiplicity_sum": sum(known[label]["multiplicity"] for label in present),
        "missing_logical_qstars": [record["label"] for record in missing],
        "missing_selected_iq": [record["selected_iq"] for record in missing],
        "missing_multiplicity_sum": sum(record["multiplicity"] for record in missing),
        "missing_qstars": [
            {
                "logical_qstar_label": record["label"],
                "selected_iq": record["selected_iq"],
                "multiplicity": record["multiplicity"],
                "q_weight": record["multiplicity"] / 64.0,
            }
            for record in missing
        ],
        "resource_estimate": {
            "calibration_wall_minutes": list(wall),
            "nodes_per_q": nodes_per_q,
            "missing_q_count": missing_count,
            "concurrent_wall_hours_min": wall_min / 60.0,
            "concurrent_wall_hours_max": wall_max / 60.0,
            "node_hours_min": missing_count * nodes_per_q * wall_min / 60.0,
            "node_hours_max": missing_count * nodes_per_q * wall_max / 60.0,
            "storage_gib": missing_count * storage_gib_per_q,
        },
        "direct_solid_reference": {
            "status": "present" if reference_exists else "missing",
            "path": str(reference) if reference is not None else None,
            "sha256": sha256(reference) if reference_exists else None,
        },
        "physical_submission_gate": (
            "not_required"
            if not missing
            else "pending_input_freeze"
            if reference_exists
            else "hold"
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--measured-q-wall-minutes", type=float, action="append", required=True)
    parser.add_argument("--nodes-per-q", type=int, required=True)
    parser.add_argument("--storage-gib-per-q", type=float, required=True)
    parser.add_argument("--direct-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    inventory = json.loads(args.inventory.read_text(encoding="ascii"))
    result = build_complement_contract(
        inventory,
        inventory_sha256=sha256(args.inventory),
        source_commit=args.source_commit,
        measured_q_wall_minutes=args.measured_q_wall_minutes,
        nodes_per_q=args.nodes_per_q,
        storage_gib_per_q=args.storage_gib_per_q,
        direct_reference=args.direct_reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
