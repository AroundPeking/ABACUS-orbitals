#!/usr/bin/env python3
"""Classify existing diamond-C direct-response reference artifacts read-only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


from build_c_solid_all_q_candidate import DIAMOND_QSTAR_CONTRACT


STATUS_NAME = "STERNHEIMER_SIAB_STATUS.dat"
DATASET_NAME = "STERNHEIMER_BASIS_OPT_V1"


def _parse_status(path):
    fields = {}
    for line in Path(path).read_text(encoding="ascii", errors="strict").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        fields[parts[0]] = parts[1] if len(parts) == 2 else ""
    return fields


def _q_directory_count(root):
    return sum(
        child.is_dir() and re.fullmatch(r"q\d+", child.name) is not None
        for child in Path(root).iterdir()
    )


def _response_dataset(status_path, fields):
    output = Path(status_path).parent
    dataset = output / DATASET_NAME
    response_files = tuple(dataset.glob("response_ik_*_ifreq_*.bin"))
    valid = (
        fields.get("status") == "success"
        and fields.get("format") == "basis_opt_v1"
        and fields.get("all_converged") == "yes"
        and fields.get("nfreq") == "6"
        and (dataset / "status.dat").is_file()
        and bool(response_files)
    )
    if not valid:
        return None
    try:
        selected_iq = int(fields["sternheimer_q_index"])
    except (KeyError, ValueError):
        return None
    return {
        "path": str(dataset),
        "selected_iq": selected_iq,
        "response_file_count": len(response_files),
        "status_path": str(status_path),
    }


def _audit_candidate_root(root):
    root = Path(root)
    status_paths = tuple(root.rglob(STATUS_NAME))
    fields = tuple(_parse_status(path) for path in status_paths)
    status_values = Counter(record.get("status", "missing") for record in fields)
    datasets = []
    for status_path, record in zip(status_paths, fields):
        dataset = _response_dataset(status_path, record)
        if dataset is not None:
            datasets.append(dataset)
    coulomb_files = sum(1 for _ in root.rglob("v1_coulomb_full_iq_*_rank0.dat"))
    if status_paths and not datasets and set(status_values) == {"abfs_diag_only"}:
        classification = "coulomb_only_diagnostic"
    elif datasets:
        classification = "contains_converged_response"
    else:
        classification = "incomplete_or_incompatible"
    return {
        "path": str(root),
        "q_directory_count": _q_directory_count(root),
        "status_file_count": len(status_paths),
        "status_values": dict(sorted(status_values.items())),
        "full_coulomb_file_count": coulomb_files,
        "response_dataset_count": len(datasets),
        "classification": classification,
    }


def audit_direct_reference_candidates(*, candidate_roots, response_roots):
    candidate_records = [_audit_candidate_root(path) for path in candidate_roots]
    response_records = []
    for root in response_roots:
        for status_path in Path(root).rglob(STATUS_NAME):
            fields = _parse_status(status_path)
            dataset = _response_dataset(status_path, fields)
            if dataset is not None:
                response_records.append(dataset)

    contract_by_iq = {
        record["selected_iq"]: record for record in DIAMOND_QSTAR_CONTRACT
    }
    available_selected_iq = [
        record["selected_iq"]
        for record in DIAMOND_QSTAR_CONTRACT
        if any(item["selected_iq"] == record["selected_iq"] for item in response_records)
    ]
    available_logical_qstars = [
        contract_by_iq[index]["label"] for index in available_selected_iq
    ]
    available_multiplicity_sum = sum(
        contract_by_iq[index]["multiplicity"] for index in available_selected_iq
    )
    missing = [
        record
        for record in DIAMOND_QSTAR_CONTRACT
        if record["selected_iq"] not in available_selected_iq
    ]
    if missing:
        reference = {
            "status": "missing",
            "reason": (
                "converged_response_coverage_is_reduced"
                if response_records
                else "no_converged_response_dataset_found"
            ),
        }
    else:
        reference = {
            "status": "candidate_pending_physics_audit",
            "reason": "qstar_coverage_complete_but_shared_physics_not_audited",
        }
    return {
        "status": "success",
        "scope": "diamond_c_solid_direct_reference_audit",
        "candidate_roots": candidate_records,
        "converged_response_datasets": response_records,
        "available_selected_iq": available_selected_iq,
        "available_logical_qstars": available_logical_qstars,
        "available_multiplicity_sum": available_multiplicity_sum,
        "missing_selected_iq": [record["selected_iq"] for record in missing],
        "missing_logical_qstars": [record["label"] for record in missing],
        "missing_multiplicity_sum": sum(record["multiplicity"] for record in missing),
        "direct_solid_reference": reference,
        "physical_submission_gate": "hold",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument("--response-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_direct_reference_candidates(
        candidate_roots=args.candidate_root,
        response_roots=args.response_root,
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
