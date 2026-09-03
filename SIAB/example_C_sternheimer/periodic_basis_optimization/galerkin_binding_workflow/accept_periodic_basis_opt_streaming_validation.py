#!/usr/bin/env python3
"""Accept a completed streaming validation without rescanning its dataset."""

import argparse
import json
import math


EXACT_FIELDS = (
    "entries",
    "kpoints",
    "frequencies",
    "raw_auxiliary_dimension",
    "whitened_auxiliary_rank",
    "primitive_count",
)
DIAGNOSTIC_FIELDS = (
    "metric_hermitian_relative_error",
    "declared_whitening_max_error",
    "sampled_whitening_max_error",
    "sampled_whitening_limit",
    "reference_response_hermitian_relative_error",
    "overlap_hermitian_relative_error",
    "hamiltonian_hermitian_relative_error",
)


def accept_validation(actual, reference, max_rss_kb, max_rss_limit_kb):
    if actual.get("status") != "success":
        raise RuntimeError("streaming validation did not report success")
    for field in EXACT_FIELDS:
        if actual.get(field) != reference.get(field):
            raise RuntimeError("dimension mismatch: " + field)

    for field in DIAGNOSTIC_FIELDS:
        if field not in actual or not math.isfinite(float(actual[field])):
            raise RuntimeError("numerical gate failed: " + field)
    numerical_checks = {
        "metric_hermitian_relative_error": float(
            actual["metric_hermitian_relative_error"]
        )
        <= 1.0e-10,
        "declared_whitening_max_error": float(
            actual["declared_whitening_max_error"]
        )
        <= 1.0e-8,
        "sampled_whitening_max_error": float(
            actual["sampled_whitening_max_error"]
        )
        <= float(actual["sampled_whitening_limit"]),
        "reference_response_hermitian_relative_error": float(
            actual["reference_response_hermitian_relative_error"]
        )
        <= 1.0e-10,
        "overlap_hermitian_relative_error": float(
            actual["overlap_hermitian_relative_error"]
        )
        <= 1.0e-10,
        "hamiltonian_hermitian_relative_error": float(
            actual["hamiltonian_hermitian_relative_error"]
        )
        <= 1.0e-8,
    }
    failed_checks = [name for name, passed in numerical_checks.items() if not passed]
    if failed_checks:
        raise RuntimeError("numerical gate failed: " + ",".join(failed_checks))
    if max_rss_kb <= 0 or max_rss_kb > max_rss_limit_kb:
        raise RuntimeError("memory gate failed")

    differences = {
        field: float(actual[field]) - float(reference[field])
        for field in DIAGNOSTIC_FIELDS
    }
    return {
        "status": "success_recovered_from_completed_streaming_validation",
        "dimension_parity": "pass",
        "numerical_gate": "pass",
        "memory_gate": "pass",
        "max_rss_kb": int(max_rss_kb),
        "max_rss_limit_kb": int(max_rss_limit_kb),
        "cross_host_diagnostic_differences": differences,
        "validation": actual,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_json")
    parser.add_argument("reference_validation_json")
    parser.add_argument("--max-rss-kb", type=int, required=True)
    parser.add_argument("--max-rss-limit-kb", type=int, required=True)
    parser.add_argument("--source-validator-job-id", required=True)
    parser.add_argument("--physical-source-job-id", required=True)
    parser.add_argument("--siab-source-commit", required=True)
    parser.add_argument("--validator-sha256", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--status-sha256", required=True)
    args = parser.parse_args(argv)

    with open(args.validation_json, encoding="ascii") as handle:
        actual = json.load(handle)
    with open(args.reference_validation_json, encoding="ascii") as handle:
        reference = json.load(handle)
    payload = accept_validation(
        actual,
        reference,
        args.max_rss_kb,
        args.max_rss_limit_kb,
    )
    payload.update(
        {
            "source_validator_job_id": args.source_validator_job_id,
            "physical_source_job_id": args.physical_source_job_id,
            "siab_source_commit": args.siab_source_commit,
            "validator_sha256": args.validator_sha256,
            "dataset_root": args.dataset_root,
            "manifest_sha256": args.manifest_sha256,
            "status_sha256": args.status_sha256,
        }
    )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return payload


if __name__ == "__main__":
    main()
