#!/usr/bin/env python3
"""Build and validate solid-only all-q Galerkin candidates for diamond C."""

from __future__ import annotations

import math

import torch


FULL_QSTAR_LABELS = (1, 2, 3, 6, 7, 8, 11, 28)
SHARED_PROVENANCE_FIELDS = (
    "abacus_commit",
    "executable_sha256",
    "orbital_sha256",
    "pseudopotential_sha256",
    "auxiliary_basis_sha256",
    "primitive_blocks_sha256",
)


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
            if not torch.equal(frequency, reference_frequency):
                raise ValueError("frequency grid differs across q-star datasets")
            if not torch.equal(weights, reference_weights):
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

