#!/usr/bin/env python3
"""Build and validate solid-only all-q Galerkin candidates for diamond C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch


HERE = Path(__file__).resolve().parent
PERIODIC_ROOT = HERE.parent
SIAB_ROOT = PERIODIC_ROOT.parents[1]
OPT_ROOT = SIAB_ROOT / "opt_orb_pytorch_dpsi"
sys.path.insert(0, str(OPT_ROOT))
sys.path.insert(0, str(PERIODIC_ROOT))

from optimize_periodic_basis import validate_dataset_contract  # noqa: E402
from periodic_galerkin_basis import (  # noqa: E402
    read_periodic_optimizer_coefficients,
    write_periodic_optimizer_coefficients,
)
from periodic_galerkin_candidates import (  # noqa: E402
    build_single_family_candidate,
    evaluate_candidate_family_losses,
    evaluate_family_gradients,
)
from periodic_galerkin_data import read_periodic_galerkin_dataset  # noqa: E402


FULL_QSTAR_LABELS = (1, 2, 3, 6, 7, 8, 11, 28)
DIAMOND_QSTAR_CONTRACT = (
    {"label": 1, "selected_iq": 1, "multiplicity": 1},
    {"label": 2, "selected_iq": 22, "multiplicity": 8},
    {"label": 3, "selected_iq": 43, "multiplicity": 4},
    {"label": 6, "selected_iq": 6, "multiplicity": 6},
    {"label": 7, "selected_iq": 7, "multiplicity": 24},
    {"label": 8, "selected_iq": 8, "multiplicity": 12},
    {"label": 11, "selected_iq": 11, "multiplicity": 3},
    {"label": 28, "selected_iq": 28, "multiplicity": 6},
)
SHARED_PROVENANCE_FIELDS = (
    "abacus_commit",
    "executable_sha256",
    "orbital_sha256",
    "pseudopotential_sha256",
    "auxiliary_basis_sha256",
    "primitive_blocks_sha256",
)
REQUIRED_CONFIG_FIELDS = {
    "ao_count_per_atom",
    "candidate_nu",
    "coverage",
    "element",
    "fixed_nu",
    "format_version",
    "occupied_capture_floor",
    "q_count",
    "qstar_contract",
    "radial_rows",
    "system",
    "trust_radius",
}
FREQUENCY_RTOL = 1.0e-12
FREQUENCY_ATOL = 1.0e-14


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_frequency_tensor(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or value.device.type != "cpu"
        or value.ndim != 1
        or value.numel() != 6
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a finite six-point CPU float64 tensor")
    return value


def _parse_qstar_contract(qstar_contract, *, coverage, q_count):
    contract = tuple(qstar_contract)
    if not contract:
        raise ValueError("q-star contract must be nonempty")
    labels = []
    selected_iq = []
    multiplicities = []
    for record in contract:
        if not isinstance(record, dict) or set(record) != {
            "label",
            "selected_iq",
            "multiplicity",
        }:
            raise ValueError("q-star contract records have invalid fields")
        labels.append(_positive_integer(record["label"], "logical q-star label"))
        selected_iq.append(_positive_integer(record["selected_iq"], "selected_iq"))
        multiplicities.append(
            _positive_integer(record["multiplicity"], "q-star multiplicity")
        )
    if len(set(labels)) != len(labels) or len(set(selected_iq)) != len(selected_iq):
        raise ValueError("logical q-star labels and selected_iq values must be unique")
    if coverage == "full" and tuple(labels) != FULL_QSTAR_LABELS:
        raise ValueError("full coverage requires the eight logical q stars")
    if coverage == "full" and sum(multiplicities) != q_count:
        raise ValueError("full q-star multiplicities must cover the q mesh")
    return contract, labels, selected_iq, multiplicities


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, name, *, positive=False):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (positive and value <= 0.0)
    ):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite")
    return float(value)


def load_config(path):
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    if not isinstance(payload, dict) or set(payload) != REQUIRED_CONFIG_FIELDS:
        raise ValueError("solid-only workflow config fields differ from the contract")
    if payload["format_version"] != 1 or payload["system"] != "C_diamond_solid":
        raise ValueError("unsupported solid-only C workflow config")
    if payload["element"] != "C":
        raise ValueError("solid-only C workflow element must be C")
    if payload["coverage"] not in {"full", "reduced"}:
        raise ValueError("coverage must be full or reduced")
    _positive_integer(payload["q_count"], "q_count")
    _positive_integer(payload["radial_rows"], "radial_rows")
    _positive_integer(payload["ao_count_per_atom"], "ao_count_per_atom")
    if payload["candidate_nu"] != [3, 3, 2, 0, 0] or payload["fixed_nu"] != [
        2,
        2,
        1,
        0,
        0,
    ]:
        raise ValueError("solid-only C workflow orbital profile differs")
    if payload["radial_rows"] != 31 or payload["ao_count_per_atom"] != 22:
        raise ValueError("solid-only C workflow dimensions differ")
    _finite(payload["trust_radius"], "trust_radius", positive=True)
    _finite(payload["occupied_capture_floor"], "occupied_capture_floor", positive=True)
    if not 0.0 < payload["occupied_capture_floor"] <= 1.0:
        raise ValueError("occupied capture floor must be in (0, 1]")
    _, labels, _, _ = _parse_qstar_contract(
        payload["qstar_contract"],
        coverage=payload["coverage"],
        q_count=payload["q_count"],
    )
    if payload["coverage"] == "reduced" and labels != [1, 2, 3]:
        raise ValueError("reduced replay must use logical q stars 1, 2, and 3")
    return payload


def _parse_qstar(value):
    try:
        label_text, path_text = value.split("=", 1)
        label = int(label_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("qstar must be LABEL=PATH") from error
    if label <= 0 or not path_text:
        raise argparse.ArgumentTypeError("qstar must be LABEL=PATH")
    return label, Path(path_text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qstar", type=_parse_qstar, action="append", required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--block-cache-workers", type=int, default=8)
    return parser.parse_args(argv)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def validate_qstar_datasets(
    datasets,
    *,
    qstar_contract,
    q_count,
    coverage,
):
    """Validate symmetry labels, weights, grids, and shared response provenance."""
    if coverage not in {"full", "reduced"}:
        raise ValueError("coverage must be full or reduced")
    _positive_integer(q_count, "q_count")
    datasets = tuple(datasets)
    contract = tuple(qstar_contract)
    if not datasets or len(datasets) != len(contract):
        raise ValueError("one dataset is required for every q-star contract record")
    contract, labels, selected_iq, multiplicities = _parse_qstar_contract(
        contract,
        coverage=coverage,
        q_count=q_count,
    )

    reference_frequency = None
    reference_weights = None
    reference_provenance = None
    for dataset, record in zip(datasets, contract):
        if getattr(dataset, "selected_iq", None) != record["selected_iq"]:
            raise ValueError("dataset selected_iq does not match the q-star contract")
        if getattr(dataset, "q_count", None) != q_count:
            raise ValueError("dataset q_count does not match the q-star contract")
        q_weight = getattr(dataset, "q_weight", None)
        expected_weight = record["multiplicity"] / q_count
        if (
            not isinstance(q_weight, (int, float))
            or isinstance(q_weight, bool)
            or not math.isfinite(q_weight)
            or abs(float(q_weight) - expected_weight) > 1.0e-12
        ):
            raise ValueError("dataset q weight does not match star multiplicity")

        frequency = _validate_frequency_tensor(
            getattr(dataset, "frequency_ha", None),
            "frequency grid",
        )
        weights = _validate_frequency_tensor(
            getattr(dataset, "frequency_weights_ha", None),
            "frequency weights",
        )
        provenance = tuple(
            getattr(dataset, field, None) for field in SHARED_PROVENANCE_FIELDS
        )
        if any(not isinstance(value, str) or not value for value in provenance):
            raise ValueError("shared provenance fields must be nonempty strings")
        if reference_frequency is None:
            reference_frequency = frequency
            reference_weights = weights
            reference_provenance = provenance
        else:
            if not torch.allclose(
                frequency,
                reference_frequency,
                rtol=FREQUENCY_RTOL,
                atol=FREQUENCY_ATOL,
            ):
                raise ValueError("frequency grid differs across q-star datasets")
            if not torch.allclose(
                weights,
                reference_weights,
                rtol=FREQUENCY_RTOL,
                atol=FREQUENCY_ATOL,
            ):
                raise ValueError("frequency weights differ across q-star datasets")
            if provenance != reference_provenance:
                raise ValueError("shared provenance differs across q-star datasets")

    return {
        "dataset_contract_gate": "pass",
        "coverage": coverage,
        "logical_qstar_labels": labels,
        "selected_iq": selected_iq,
        "multiplicities": multiplicities,
        "multiplicity_sum": sum(multiplicities),
        "q_count": q_count,
        "frequency_count": int(reference_frequency.numel()),
        "physical_release_gate": "pending_candidate" if coverage == "full" else "hold",
    }


def build_dataset_inventory(datasets, contract_result):
    records = []
    for label, multiplicity, dataset in zip(
        contract_result["logical_qstar_labels"],
        contract_result["multiplicities"],
        datasets,
    ):
        records.append(
            {
                "logical_qstar_label": label,
                "selected_iq": dataset.selected_iq,
                "multiplicity": multiplicity,
                "q_weight": float(dataset.q_weight),
                "physics_hash": dataset.physics_hash,
            }
        )
    return dict(contract_result, datasets=records)


def main(
    argv=None,
    *,
    dataset_reader=read_periodic_galerkin_dataset,
    dataset_contract_validator=validate_dataset_contract,
    coefficient_reader=read_periodic_optimizer_coefficients,
    gradient_evaluator=evaluate_family_gradients,
    candidate_builder=build_single_family_candidate,
    candidate_evaluator=evaluate_candidate_family_losses,
    coefficient_writer=write_periodic_optimizer_coefficients,
):
    args = parse_args(argv)
    config = load_config(args.config)
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git hash")
    if type(args.block_cache_workers) is not int or args.block_cache_workers <= 0:
        raise ValueError("block-cache workers must be positive")
    qstar_paths = dict(args.qstar)
    if len(qstar_paths) != len(args.qstar):
        raise ValueError("qstar labels must be unique")
    expected_labels = [record["label"] for record in config["qstar_contract"]]
    if set(qstar_paths) != set(expected_labels):
        raise ValueError("qstar paths do not match the configured logical labels")
    if not args.config.is_file() or not args.initial.is_file():
        raise ValueError("config and initial coefficient files must exist")
    for path in qstar_paths.values():
        if not path.is_dir() or path.is_symlink():
            raise ValueError("qstar inputs must be real directories")
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(output)

    ordered_paths = tuple(qstar_paths[label].resolve() for label in expected_labels)
    datasets = tuple(
        dataset_reader(
            path,
            include_reference_projection=False,
            verify_omitted_chunks=False,
        )
        for path in ordered_paths
    )
    dataset_contract_validator(datasets)
    contract_result = validate_qstar_datasets(
        datasets,
        qstar_contract=config["qstar_contract"],
        q_count=config["q_count"],
        coverage=config["coverage"],
    )
    inventory = build_dataset_inventory(datasets, contract_result)
    initial = coefficient_reader(
        args.initial,
        element=config["element"],
        radial_rows=config["radial_rows"],
        expected_nu=tuple(config["candidate_nu"]),
    )

    output.mkdir(parents=True)
    _write_json(
        output / "STATUS.json",
        {"status": "running", "source_commit": args.source_commit},
    )
    try:
        gradient = gradient_evaluator(
            datasets,
            initial,
            fixed_nu={"C": tuple(config["fixed_nu"])},
            dataset_families=("C_solid",) * len(datasets),
            occupied_capture_tolerance=1.0 - config["occupied_capture_floor"],
            block_cache_workers=args.block_cache_workers,
        )
        candidate = candidate_builder(
            gradient,
            fixed_nu={"C": tuple(config["fixed_nu"])},
            family="C_solid",
            trust_radius=config["trust_radius"],
        )
        evaluation = candidate_evaluator(gradient, candidate.coefficients)
        baseline_loss = float(gradient.family_losses["C_solid"])
        candidate_loss = float(evaluation["family_losses"]["C_solid"])
        capture = float(evaluation["minimum_occupied_capture"])
        candidate_generation_gate = (
            "pass"
            if math.isfinite(candidate_loss)
            and candidate_loss < baseline_loss
            and math.isfinite(capture)
            and capture >= config["occupied_capture_floor"]
            else "fail"
        )

        orbital_path = output / "ORBITAL_RESULTS.txt"
        coefficient_writer(orbital_path, candidate.coefficients)
        _write_json(output / "DATASET_INVENTORY.json", inventory)
        gradient_payload = {
            "family_order": list(gradient.family_order),
            "family_losses": gradient.family_losses,
            "gradient_norms": gradient.gradient_norms,
            "gradient_cosines": gradient.gradient_cosines,
            "minimum_occupied_capture": gradient.minimum_occupied_capture,
            "maximum_overlap_condition": gradient.maximum_overlap_condition,
        }
        _write_json(output / "GRADIENT.json", gradient_payload)
        candidate_payload = {
            "status": "success",
            "coverage": config["coverage"],
            "family": candidate.family,
            "trust_radius": candidate.trust_radius,
            "coefficients_sha256": candidate.coefficients_sha256,
            "orbital_sha256": sha256(orbital_path),
            "baseline_family_loss": baseline_loss,
            "candidate_family_loss": candidate_loss,
            "candidate_evaluation": evaluation,
            "candidate_generation_gate": candidate_generation_gate,
            "physical_release_gate": (
                "pending_validation"
                if config["coverage"] == "full" and candidate_generation_gate == "pass"
                else "hold"
            ),
        }
        _write_json(output / "CANDIDATE.json", candidate_payload)
        input_hashes = {
            "config": sha256(args.config),
            "initial": sha256(args.initial),
            **{
                f"qstar_{label}": dataset.physics_hash
                for label, dataset in zip(expected_labels, datasets)
            },
        }
        provenance = {
            "status": "success",
            "source_commit": args.source_commit,
            "script_sha256": sha256(Path(__file__)),
            "input_sha256": input_hashes,
            "dataset_inventory_sha256": sha256(output / "DATASET_INVENTORY.json"),
            "gradient_sha256": sha256(output / "GRADIENT.json"),
            "candidate_sha256": sha256(output / "CANDIDATE.json"),
            "orbital_sha256": sha256(orbital_path),
        }
        _write_json(output / "PROVENANCE.json", provenance)
        _write_json(
            output / "STATUS.json",
            {
                "status": "success",
                "coverage": config["coverage"],
                "candidate_generation_gate": candidate_generation_gate,
                "physical_release_gate": candidate_payload["physical_release_gate"],
            },
        )
        print(json.dumps(candidate_payload, indent=2, sort_keys=True, allow_nan=False))
        return candidate_payload
    except Exception as error:
        _write_json(
            output / "STATUS.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
