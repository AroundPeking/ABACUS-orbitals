#!/usr/bin/env python3
"""Audit existing diamond-C q-star Galerkin inputs without running physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PERIODIC_ROOT = HERE.parent
SIAB_ROOT = PERIODIC_ROOT.parents[1]
OPT_ROOT = SIAB_ROOT / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OPT_ROOT))
sys.path.insert(0, str(PERIODIC_ROOT))

from build_c_solid_all_q_candidate import (  # noqa: E402
    DIAMOND_QSTAR_CONTRACT,
    _parse_qstar,
    validate_qstar_datasets,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_qstar_inputs(
    qstar_paths,
    *,
    dataset_reader=read_periodic_galerkin_dataset,
    direct_reference=None,
):
    qstar_paths = dict(qstar_paths)
    known = {record["label"]: record for record in DIAMOND_QSTAR_CONTRACT}
    if not qstar_paths or not set(qstar_paths).issubset(known):
        raise ValueError("qstar inputs must use known diamond logical labels")
    present_labels = [
        record["label"]
        for record in DIAMOND_QSTAR_CONTRACT
        if record["label"] in qstar_paths
    ]
    present_contract = tuple(known[label] for label in present_labels)
    datasets = tuple(dataset_reader(qstar_paths[label]) for label in present_labels)
    coverage = "full" if tuple(present_labels) == tuple(known) else "reduced"
    validation = validate_qstar_datasets(
        datasets,
        qstar_contract=present_contract,
        q_count=64,
        coverage=coverage,
    )
    records = []
    for record, path, dataset in zip(
        present_contract,
        (qstar_paths[label] for label in present_labels),
        datasets,
    ):
        records.append(
            {
                "logical_qstar_label": record["label"],
                "selected_iq": dataset.selected_iq,
                "multiplicity": record["multiplicity"],
                "q_weight": float(dataset.q_weight),
                "physics_hash": dataset.physics_hash,
                "path": str(path),
            }
        )
    reference_path = Path(direct_reference) if direct_reference is not None else None
    reference_exists = reference_path is not None and reference_path.exists()
    missing_contract = tuple(
        record
        for record in DIAMOND_QSTAR_CONTRACT
        if record["label"] not in qstar_paths
    )
    direct_reference_record = {
        "exists": reference_exists,
        "path": str(reference_path) if reference_path is not None else None,
        "sha256": (
            sha256(reference_path)
            if reference_exists and reference_path.is_file()
            else None
        ),
        "status": "present" if reference_exists else "missing",
    }
    return {
        "status": "success",
        "scope": "read_only_solid_qstar_input_audit",
        "present_logical_qstars": present_labels,
        "present_selected_iq": validation["selected_iq"],
        "present_multiplicity_sum": validation["multiplicity_sum"],
        "missing_logical_qstars": [record["label"] for record in missing_contract],
        "missing_selected_iq": [
            record["selected_iq"] for record in missing_contract
        ],
        "missing_multiplicity_sum": sum(
            record["multiplicity"] for record in missing_contract
        ),
        "missing_qstars": [
            {
                "logical_qstar_label": record["label"],
                "selected_iq": record["selected_iq"],
                "multiplicity": record["multiplicity"],
                "q_weight": float(record["multiplicity"] / 64.0),
            }
            for record in missing_contract
        ],
        "datasets": records,
        "direct_solid_reference": direct_reference_record,
        "complement_submission_gate": (
            "not_required"
            if not missing_contract
            else "pending_input_freeze"
            if reference_exists
            else "hold"
        ),
        "physical_release_gate": (
            "pending_candidate"
            if coverage == "full" and reference_exists
            else "hold"
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qstar", type=_parse_qstar, action="append", required=True)
    parser.add_argument("--direct-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    qstar_paths = dict(args.qstar)
    if len(qstar_paths) != len(args.qstar):
        raise ValueError("qstar labels must be unique")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_qstar_inputs(
        qstar_paths,
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
