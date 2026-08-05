#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


EXPECTED_ARTIFACTS = {
    "rpa_sensitive_frequency.pdf",
    "rpa_sensitive_frequency.png",
    "rpa_sensitive_ranking.json",
    "rpa_sensitive_ranking.md",
}
EXPECTED_ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]
EXPECTED_GATES = {
    "first_f_improves_two_d",
    "first_g_improves_first_f",
    "second_f_not_better",
    "second_g_not_better",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", required=True, type=Path)
    parser.add_argument("--run2", required=True, type=Path)
    parser.add_argument("--rc1", required=True, type=int)
    parser.add_argument("--rc2", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_json_equal(left, right, path="$", counts=None):
    if counts is None:
        counts = {"numbers": 0, "nodes": 0}
    counts["nodes"] += 1
    if type(left) is not type(right):
        raise ValueError(f"JSON type mismatch at {path}")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError(f"JSON key mismatch at {path}")
        for key in sorted(left):
            exact_json_equal(left[key], right[key], f"{path}.{key}", counts)
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError(f"JSON length mismatch at {path}")
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            exact_json_equal(lhs, rhs, f"{path}[{index}]", counts)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        counts["numbers"] += 1
        if isinstance(left, float) and not math.isfinite(left):
            raise ValueError(f"non-finite JSON number at {path}")
        if left != right:
            raise ValueError(f"JSON numeric mismatch at {path}: {left} != {right}")
    elif left != right:
        raise ValueError(f"JSON value mismatch at {path}: {left!r} != {right!r}")
    return counts


def validate_payload(payload, analyzer_rc):
    if payload["alphas"] != EXPECTED_ALPHAS:
        raise ValueError("frozen alpha grid changed")
    if len(payload["alpha_results"]) != len(EXPECTED_ALPHAS):
        raise ValueError("alpha result count is incomplete")
    for expected_alpha, result in zip(EXPECTED_ALPHAS, payload["alpha_results"]):
        if result["alpha"] != expected_alpha:
            raise ValueError("alpha result ordering changed")
        if set(result["gates"]) != EXPECTED_GATES:
            raise ValueError("historical gate set changed")
        if not all(type(value) is bool for value in result["gates"].values()):
            raise ValueError("historical gates must be boolean")
        if result["admissible"] is not all(result["gates"].values()):
            raise ValueError("admissibility does not match all four gates")
    for flag in (
        "uses_sos_energy_as_numeric_input",
        "uses_ghost_family",
        "new_candidate_was_evaluated",
    ):
        if payload[flag] is not False:
            raise ValueError(f"forbidden campaign flag is not false: {flag}")
    admissible = [
        result["alpha"]
        for result in payload["alpha_results"]
        if result["admissible"]
    ]
    expected_selected = max(admissible) if admissible else None
    if payload["selected_alpha"] != expected_selected:
        raise ValueError("selected alpha is not the maximum admissible alpha")
    expected_decision = "pass" if expected_selected is not None else "stop_galerkin_required"
    expected_rc = 0 if expected_selected is not None else 2
    if payload["decision"] != expected_decision or analyzer_rc != expected_rc:
        raise ValueError("decision, selected alpha, and analyzer exit are inconsistent")


def main():
    args = parse_arguments()
    if args.rc1 not in (0, 2) or args.rc2 != args.rc1:
        raise ValueError("analyzer exit codes are not matching scientific outcomes")
    for output_dir in (args.run1, args.run2):
        produced = {path.name for path in output_dir.iterdir()}
        if produced != EXPECTED_ARTIFACTS:
            raise ValueError(
                f"invalid artifact set in {output_dir}: {sorted(produced)}"
            )

    json1 = args.run1 / "rpa_sensitive_ranking.json"
    json2 = args.run2 / "rpa_sensitive_ranking.json"
    payload1 = json.loads(json1.read_text(encoding="utf-8"))
    payload2 = json.loads(json2.read_text(encoding="utf-8"))
    validate_payload(payload1, args.rc1)
    validate_payload(payload2, args.rc2)
    counts = exact_json_equal(payload1, payload2)

    hashes = {}
    all_artifact_hashes_equal = True
    for name in sorted(EXPECTED_ARTIFACTS):
        run1_hash = sha256(args.run1 / name)
        run2_hash = sha256(args.run2 / name)
        hashes[name] = {"run1": run1_hash, "run2": run2_hash}
        all_artifact_hashes_equal &= run1_hash == run2_hash
    if not all_artifact_hashes_equal:
        raise ValueError("deterministic artifact hashes differ between runs")

    summary = {
        "all_artifact_hashes_equal": all_artifact_hashes_equal,
        "analyzer_exit_codes": [args.rc1, args.rc2],
        "decision": payload1["decision"],
        "exact_four_artifacts_each_run": True,
        "flags": {
            "new_candidate_was_evaluated": False,
            "uses_ghost_family": False,
            "uses_sos_energy_as_numeric_input": False,
        },
        "job_id": os.environ["SLURM_JOB_ID"],
        "json_numeric_count": counts["numbers"],
        "json_node_count": counts["nodes"],
        "numeric_json_equality": True,
        "output_sha256": hashes,
        "selected_alpha": payload1["selected_alpha"],
        "structural_json_equality": True,
    }
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
