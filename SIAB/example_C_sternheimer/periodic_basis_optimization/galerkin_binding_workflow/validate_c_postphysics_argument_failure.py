"""Gate read-only recovery of the known post-physics frequency-source typo."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re


def validate_completion(job_id, accounting, stderr, abacus_output, abacus_time):
    rows = list(csv.DictReader(io.StringIO(accounting), delimiter="|"))
    parents = [row for row in rows if row["JobID"] == job_id]
    if (len(parents) != 1 or parents[0]["State"] != "FAILED"
            or parents[0]["ExitCode"] != "2:0"):
        raise ValueError("parent must be terminal FAILED/2:0")
    children = [row for row in rows if row["JobID"].startswith(job_id + ".")]
    mpi_steps = []
    for row in children:
        suffix = row["JobID"][len(job_id) + 1:]
        expected = ("FAILED", "2:0") if suffix == "batch" else ("COMPLETED", "0:0")
        if (row["State"], row["ExitCode"]) != expected:
            raise ValueError("non-successful source step: " + row["JobID"])
        if suffix.isdigit():
            mpi_steps.append(row)
    if not mpi_steps:
        raise ValueError("no completed MPI steps in scheduler evidence")
    marker = ("validate_c_solid_fd8_q_dataset.py: error: argument "
              "--frequency-grid-source: invalid choice: "
              "'frozen_df_q1_greenx_minimax'")
    if marker not in stderr:
        raise ValueError("not the known post-physics argument failure")
    for marker in ("Transport retry count exceeded", "Node failure on", "Traceback (", "MPI_Abort"):
        if marker in stderr:
            raise ValueError("additional failure beyond argument parsing: " + marker)
    if not (re.search(r"^ FINISH Time  :", abacus_output, re.M)
            and re.search(r"^ TOTAL  Time  :", abacus_output, re.M)):
        raise ValueError("ABACUS finish markers missing")
    if re.findall(r"^\s*Exit status:\s*(\d+)\s*$", abacus_time, re.M) != ["0"]:
        raise ValueError("ABACUS time exit evidence is not uniquely zero")
    return {
        "status": "pass",
        "recovery_reason": "post_physics_validator_frequency_source_argument_mismatch",
        "source_job_id": job_id,
        "source_scheduler_state": "FAILED",
        "source_scheduler_exit_code": "2:0",
        "completed_mpi_steps": len(mpi_steps),
        "scope": "completion_evidence_only_requires_streaming_and_standard_validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    for name in ("accounting", "stderr", "abacus-output", "abacus-time"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    files = {name: getattr(args, name) for name in (
        "accounting", "stderr", "abacus_output", "abacus_time")}
    result = validate_completion(args.job_id, **{
        name: path.read_text() for name, path in files.items()})
    result["evidence"] = {name: {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    } for name, path in files.items()}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
