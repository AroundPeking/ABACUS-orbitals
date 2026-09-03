#!/usr/bin/env python3
"""Build and compare unitary-invariant basis-opt reference-response signatures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from validate_periodic_basis_opt_streaming import (
    parse_manifest,
    read_chunk_doubles,
    scan_chunk,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _response_record(path, entry, frequency, weight):
    chunk = scan_chunk(path, expected_sha256=entry["sha256"])
    values = read_chunk_doubles(path, chunk)
    matrix = np.frombuffer(values, dtype=np.float64).view(np.complex128).reshape(
        chunk["rows"], chunk["columns"]
    )
    hermitian = 0.5 * (matrix + matrix.conjugate().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    return {
        "ifrequency": int(entry["ifrequency"]),
        "frequency": float(frequency),
        "weight": float(weight),
        "trace_real": float(np.trace(matrix).real),
        "trace_imag": float(np.trace(matrix).imag),
        "frobenius_norm": float(np.linalg.norm(matrix)),
        "eigenvalues": [float(value) for value in eigenvalues],
        "chunk_sha256": chunk["sha256"],
    }


def _metadata_value(metadata, key, index=0):
    values = metadata.get(key)
    if values is None or len(values) <= index:
        raise RuntimeError("response manifest is missing metadata: " + key)
    return values[index]


def build_signature(dataset, expected_commit):
    dataset = Path(dataset)
    metadata, frequencies, kpoints, entries = parse_manifest(dataset / "manifest.dat")
    if metadata["abacus_commit"][0] != expected_commit:
        raise RuntimeError("ABACUS commit provenance mismatch")
    raw_dimension = int(metadata["raw_auxiliary_dimension"][0])
    retained_rank = int(metadata["whitened_auxiliary_rank"][0])
    primitive_count = int(metadata["primitive_count"][0])
    by_frequency = {
        entry["ifrequency"]: entry
        for entry in entries
        if entry["kind"] == 8 and entry["ik"] == 0
    }
    if len(by_frequency) != len(frequencies):
        raise RuntimeError("reference-response frequency set is incomplete")
    response = []
    for ifrequency, frequency, weight in frequencies:
        entry = by_frequency.get(ifrequency)
        if entry is None:
            raise RuntimeError("reference-response frequency set is incomplete")
        if entry["rows"] != retained_rank or entry["columns"] != retained_rank:
            raise RuntimeError("reference-response dimension mismatch")
        response.append(
            _response_record(dataset / entry["path"], entry, frequency, weight)
        )
    return {
        "status": "success",
        "signature_definition": "reference_response_hermitian_spectrum_v1",
        "dataset_root": str(dataset.resolve()),
        "manifest_sha256": _sha256(dataset / "manifest.dat"),
        "status_sha256": _sha256(dataset / "status.dat"),
        "protocol": {
            "abacus_commit": expected_commit,
            "executable_sha256": _metadata_value(metadata, "executable_sha256"),
            "orbital_sha256": _metadata_value(metadata, "orbital_sha256"),
            "pseudopotential_sha256": _metadata_value(
                metadata, "pseudopotential_sha256"
            ),
            "auxiliary_basis_source": _metadata_value(
                metadata, "auxiliary_basis_source"
            ),
            "auxiliary_basis_sha256": _metadata_value(
                metadata, "auxiliary_basis_sha256"
            ),
            "primitive_blocks_sha256": _metadata_value(
                metadata, "primitive_blocks_sha256"
            ),
            "physics_hash": _metadata_value(metadata, "physics_hash"),
            "kernel": _metadata_value(metadata, "kernel"),
            "q_count": int(_metadata_value(metadata, "q_count")),
            "selected_iq": int(_metadata_value(metadata, "selected_iq")),
            "qpoint": [float(value) for value in metadata["qpoint"]],
            "q_weight": float(_metadata_value(metadata, "q_weight")),
            "raw_auxiliary_dimension": raw_dimension,
            "whitened_auxiliary_rank": retained_rank,
            "primitive_count": primitive_count,
            "kpoints": len(kpoints),
            "frequencies": [list(item) for item in frequencies],
        },
        "reference_response": response,
    }


def _relative_vector_error(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise RuntimeError("response spectrum dimension mismatch")
    return float(
        np.linalg.norm(actual - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-15)
    )


def compare_signatures(actual, reference, relative_tolerance):
    if actual.get("status") != "success" or reference.get("status") != "success":
        raise RuntimeError("response signature is not successful")
    if actual.get("signature_definition") != reference.get("signature_definition"):
        raise RuntimeError("response signature definition mismatch")
    protocol_fields = (
        "abacus_commit",
        "orbital_sha256",
        "pseudopotential_sha256",
        "auxiliary_basis_source",
        "auxiliary_basis_sha256",
        "primitive_blocks_sha256",
        "kernel",
        "q_count",
        "selected_iq",
        "qpoint",
        "q_weight",
        "raw_auxiliary_dimension",
        "whitened_auxiliary_rank",
        "primitive_count",
        "kpoints",
        "frequencies",
    )
    for field in protocol_fields:
        if actual["protocol"].get(field) != reference["protocol"].get(field):
            raise RuntimeError("response protocol mismatch: " + field)
    actual_response = actual["reference_response"]
    reference_response = reference["reference_response"]
    if len(actual_response) != len(reference_response):
        raise RuntimeError("response spectrum dimension mismatch")

    comparisons = []
    maximum = 0.0
    for actual_record, reference_record in zip(actual_response, reference_response):
        if actual_record["ifrequency"] != reference_record["ifrequency"]:
            raise RuntimeError("response frequency ordering mismatch")
        spectrum_error = _relative_vector_error(
            actual_record["eigenvalues"], reference_record["eigenvalues"]
        )
        trace_error = abs(
            complex(actual_record["trace_real"], actual_record["trace_imag"])
            - complex(reference_record["trace_real"], reference_record["trace_imag"])
        ) / max(
            abs(
                complex(
                    reference_record["trace_real"], reference_record["trace_imag"]
                )
            ),
            1.0e-15,
        )
        frobenius_error = abs(
            actual_record["frobenius_norm"] - reference_record["frobenius_norm"]
        ) / max(abs(reference_record["frobenius_norm"]), 1.0e-15)
        maximum = max(maximum, spectrum_error)
        comparisons.append(
            {
                "ifrequency": actual_record["ifrequency"],
                "spectrum_relative_error": spectrum_error,
                "trace_relative_error": trace_error,
                "frobenius_relative_error": frobenius_error,
            }
        )
    if maximum > relative_tolerance:
        raise RuntimeError(
            "response spectrum mismatch: "
            + f"{maximum:.16e} > {relative_tolerance:.16e}"
        )
    return {
        "status": "success",
        "gate": "pass",
        "physics_hash_match": (
            actual["protocol"].get("physics_hash")
            == reference["protocol"].get("physics_hash")
        ),
        "actual_physics_hash": actual["protocol"].get("physics_hash"),
        "reference_physics_hash": reference["protocol"].get("physics_hash"),
        "executable_hash_match": (
            actual["protocol"].get("executable_sha256")
            == reference["protocol"].get("executable_sha256")
        ),
        "actual_executable_sha256": actual["protocol"].get(
            "executable_sha256"
        ),
        "reference_executable_sha256": reference["protocol"].get(
            "executable_sha256"
        ),
        "relative_tolerance": float(relative_tolerance),
        "max_response_spectrum_relative_error": maximum,
        "frequency_comparisons": comparisons,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("dataset")
    build.add_argument("--commit", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("actual")
    compare.add_argument("reference")
    compare.add_argument("--relative-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args(argv)

    if args.command == "build":
        payload = build_signature(args.dataset, args.commit)
    else:
        with open(args.actual, encoding="ascii") as handle:
            actual = json.load(handle)
        with open(args.reference, encoding="ascii") as handle:
            reference = json.load(handle)
        payload = compare_signatures(actual, reference, args.relative_tolerance)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return payload


if __name__ == "__main__":
    main()
