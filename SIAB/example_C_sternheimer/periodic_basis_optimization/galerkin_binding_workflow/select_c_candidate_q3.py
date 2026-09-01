#!/usr/bin/env python3
"""Select one C Galerkin bank candidate using independent diamond q3 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_c_candidate_bank import load_config  # noqa: E402


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


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _indexed_candidates(records, *, name):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{name} candidates must be a nonempty list")
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{name} candidate must be a dictionary")
        label = record.get("name", record.get("label"))
        if not isinstance(label, str) or not label.strip() or label in indexed:
            raise ValueError(f"{name} candidate labels must be unique and nonempty")
        indexed[label] = record
    return indexed


def select_q3_candidate(*, config, bank, comparison):
    """Return a deterministic q3 gate and at most one selected candidate."""
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    if (
        not isinstance(bank, dict)
        or bank.get("format_version") != 1
        or bank.get("status") != "success"
        or bank.get("candidate_bank_gate") != "pass"
    ):
        raise ValueError("candidate bank must be a successful version-1 bank")
    if (
        not isinstance(comparison, dict)
        or comparison.get("format_version") != 1
        or "independent SOS validation required" not in comparison.get("scope", "")
    ):
        raise ValueError("q3 comparison must use the Galerkin screening contract")
    datasets = comparison.get("datasets")
    if (
        not isinstance(datasets, list)
        or len(datasets) != 1
        or datasets[0].get("selected_iq") != 43
    ):
        raise ValueError("q3 comparison must contain only selected_iq 43")
    floor = _finite(config.get("occupied_capture_floor"), "capture floor", positive=True)
    comparison_floor = _finite(
        comparison.get("occupied_capture_floor"),
        "comparison capture floor",
        positive=True,
    )
    if comparison_floor != floor:
        raise ValueError("q3 comparison occupied-capture floor differs")
    condition_ratio_limit = _finite(
        config.get("q3_maximum_condition_ratio"),
        "q3 maximum condition ratio",
        positive=True,
    )

    bank_candidates = _indexed_candidates(bank.get("candidates"), name="bank")
    comparison_candidates = _indexed_candidates(
        comparison.get("candidates"),
        name="comparison",
    )
    expected_names = {
        name
        for name, record in bank_candidates.items()
        if record.get("family_tradeoff_gate") == "pass"
    }
    if not expected_names:
        raise ValueError("candidate bank has no promotable candidate")
    if set(comparison_candidates) != expected_names | {"initial"}:
        raise ValueError("q3 comparison candidate set differs from the bank")
    initial_hash = _sha256(
        bank.get("input_sha256", {}).get("initial"),
        "initial input hash",
    )
    if (
        _sha256(
            comparison_candidates["initial"].get("coefficients_sha256"),
            "q3 initial hash",
        )
        != initial_hash
    ):
        raise ValueError("q3 initial candidate hash differs from the bank")
    for name in expected_names:
        bank_hash = _sha256(
            bank_candidates[name].get("orbital_sha256"),
            f"{name} orbital hash",
        )
        comparison_hash = _sha256(
            comparison_candidates[name].get("coefficients_sha256"),
            f"{name} q3 hash",
        )
        if bank_hash != comparison_hash:
            raise ValueError(f"{name} q3 candidate hash differs from the bank")

    baseline = comparison_candidates["initial"]
    baseline_pi = _finite(
        baseline.get("global_weighted_relative_pi_error"),
        "initial q3 Pi error",
        positive=True,
    )
    baseline_trace = _finite(
        baseline.get("global_weighted_relative_trace_log_error"),
        "initial q3 trace-log error",
        positive=True,
    )
    baseline_condition = _finite(
        baseline.get("maximum_overlap_condition"),
        "initial q3 overlap condition",
        positive=True,
    )

    records = []
    passing = []
    for name in sorted(expected_names):
        candidate = comparison_candidates[name]
        pi_error = _finite(
            candidate.get("global_weighted_relative_pi_error"),
            f"{name} q3 Pi error",
            positive=True,
        )
        trace_error = _finite(
            candidate.get("global_weighted_relative_trace_log_error"),
            f"{name} q3 trace-log error",
            positive=True,
        )
        capture = _finite(
            candidate.get("minimum_occupied_capture"),
            f"{name} q3 occupied capture",
            positive=True,
        )
        condition = _finite(
            candidate.get("maximum_overlap_condition"),
            f"{name} q3 overlap condition",
            positive=True,
        )
        pi_ratio = pi_error / baseline_pi
        trace_ratio = trace_error / baseline_trace
        condition_ratio = condition / baseline_condition
        failure_reasons = []
        if pi_ratio >= 1.0:
            failure_reasons.append("pi_error_not_improved")
        if trace_ratio >= 1.0:
            failure_reasons.append("trace_log_error_not_improved")
        if capture < floor:
            failure_reasons.append("occupied_capture_below_floor")
        if condition_ratio >= condition_ratio_limit:
            failure_reasons.append("overlap_condition_ratio_exceeds_limit")
        score = 0.5 * (pi_ratio + trace_ratio)
        record = {
            "name": name,
            "gate": "pass" if not failure_reasons else "fail",
            "failure_reasons": failure_reasons,
            "q3_pi_error": pi_error,
            "q3_trace_log_error": trace_error,
            "q3_pi_error_ratio": pi_ratio,
            "q3_trace_log_error_ratio": trace_ratio,
            "q3_score": score,
            "minimum_occupied_capture": capture,
            "maximum_overlap_condition": condition,
            "overlap_condition_ratio": condition_ratio,
            "orbital_sha256": bank_candidates[name]["orbital_sha256"],
        }
        records.append(record)
        if not failure_reasons:
            passing.append(record)
    selected = (
        min(passing, key=lambda record: (record["q3_score"], record["name"]))
        if passing
        else None
    )
    return {
        "format_version": 1,
        "status": "success",
        "scope": "independent q3 Galerkin screen; not a PBE or SOS acceptance",
        "gate": "pass" if selected is not None else "fail",
        "selected_candidate": selected["name"] if selected is not None else None,
        "selected_orbital_sha256": (
            selected["orbital_sha256"] if selected is not None else None
        ),
        "baseline": {
            "q3_pi_error": baseline_pi,
            "q3_trace_log_error": baseline_trace,
            "maximum_overlap_condition": baseline_condition,
        },
        "occupied_capture_floor": floor,
        "maximum_condition_ratio": condition_ratio_limit,
        "selection_metric": "mean(q3_pi_error_ratio,q3_trace_log_error_ratio)",
        "candidates": records,
    }


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git hash")
    for path in (args.config, args.bank, args.comparison):
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError("q3 selection inputs must be nonempty regular files")
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(args.config)
    bank = json.loads(args.bank.read_text(encoding="ascii"))
    comparison = json.loads(args.comparison.read_text(encoding="ascii"))
    result = select_q3_candidate(
        config=config,
        bank=bank,
        comparison=comparison,
    )
    output.mkdir(parents=True)
    shutil.copyfile(args.comparison, output / "COMPARISON_RESULT.json")
    _write_json(output / "SELECTION_RESULT.json", result)
    provenance = {
        "status": "success",
        "source_commit": args.source_commit,
        "script_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(args.config),
        "candidate_bank_sha256": sha256(args.bank),
        "comparison_sha256": sha256(args.comparison),
        "selection_sha256": sha256(output / "SELECTION_RESULT.json"),
    }
    _write_json(output / "PROVENANCE.json", provenance)
    _write_json(
        output / "STATUS.json",
        {
            "status": "success",
            "gate": result["gate"],
            "selected_candidate": result["selected_candidate"],
            "selection_sha256": provenance["selection_sha256"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
