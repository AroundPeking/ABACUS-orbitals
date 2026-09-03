#!/bin/bash
#SBATCH --job-name=c_fd8_q_validate
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --time=02:00:00

set -euo pipefail
trap 'status=$?; echo "C_SOLID_FD8_VALIDATION_RECOVERY_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${DATASET_ROOT:?missing existing basis-opt-v1 dataset}"
: "${OUTPUT_ROOT:?missing immutable validation output root}"
: "${SOURCE_JOB_ID:?missing physical source job id}"
: "${SIAB_SOURCE_ROOT:?missing immutable SIAB source deployment}"
: "${SIAB_SOURCE_COMMIT:?missing SIAB source commit}"
: "${BASIS_OPT_VALIDATOR_SHA256:?missing streaming validator SHA256}"
: "${EXPECTED_ABACUS_COMMIT:?missing dataset ABACUS commit}"
: "${REFERENCE_VALIDATION_JSON:?missing accepted legacy validation JSON}"

workflow=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
runner=$workflow/recover_c_solid_fd8_q_dataset_validation.slurm
validator=$workflow/validate_periodic_basis_opt_streaming.py
validator_python=${C_BASIS_PYTHON:-/data/home/df_iopcas_ghj/app/miniconda3/bin/python}
max_rss_kb=${MAX_VALIDATOR_RSS_KB:-4194304}

test -d "$DATASET_ROOT"
test -s "$DATASET_ROOT/manifest.dat"
test -s "$DATASET_ROOT/status.dat"
test -s "$REFERENCE_VALIDATION_JSON"
test -s "$runner"
test "$(sha256sum "$validator" | awk '{print $1}')" = "$BASIS_OPT_VALIDATOR_SHA256"
test -x "$validator_python"
"$validator_python" -c 'import numpy'
test "$max_rss_kb" -gt 0
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

manifest_sha_before=$(sha256sum "$DATASET_ROOT/manifest.dat" | awk '{print $1}')
status_sha_before=$(sha256sum "$DATASET_ROOT/status.dat" | awk '{print $1}')

/usr/bin/time -v -o "$OUTPUT_ROOT/validator.time" \
  "$validator_python" "$validator" \
    "$DATASET_ROOT" --commit "$EXPECTED_ABACUS_COMMIT" \
    > "$OUTPUT_ROOT/BASIS_OPT_VALIDATION.json"

"$validator_python" - \
  "$REFERENCE_VALIDATION_JSON" "$OUTPUT_ROOT/BASIS_OPT_VALIDATION.json" <<'PY'
import json
import math
import sys

reference = json.load(open(sys.argv[1], encoding="ascii"))
actual = json.load(open(sys.argv[2], encoding="ascii"))
exact = (
    "status",
    "entries",
    "kpoints",
    "frequencies",
    "raw_auxiliary_dimension",
    "whitened_auxiliary_rank",
    "primitive_count",
)
floating = (
    "metric_hermitian_relative_error",
    "declared_whitening_max_error",
    "sampled_whitening_max_error",
    "sampled_whitening_limit",
    "reference_response_hermitian_relative_error",
    "overlap_hermitian_relative_error",
    "hamiltonian_hermitian_relative_error",
)
for key in exact:
    if actual.get(key) != reference.get(key):
        raise RuntimeError("validation parity mismatch: " + key)
for key in floating:
    if not math.isclose(
        float(actual[key]), float(reference[key]), rel_tol=1.0e-12, abs_tol=1.0e-14
    ):
        raise RuntimeError("validation parity mismatch: " + key)
PY

max_rss_observed=$(sed -n \
  's/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*//p' \
  "$OUTPUT_ROOT/validator.time")
test -n "$max_rss_observed"
test "$max_rss_observed" -le "$max_rss_kb"

manifest_sha_after=$(sha256sum "$DATASET_ROOT/manifest.dat" | awk '{print $1}')
status_sha_after=$(sha256sum "$DATASET_ROOT/status.dat" | awk '{print $1}')
test "$manifest_sha_after" = "$manifest_sha_before"
test "$status_sha_after" = "$status_sha_before"

"$validator_python" - \
  "$OUTPUT_ROOT/BASIS_OPT_VALIDATION.json" \
  "$OUTPUT_ROOT/RECOVERY_ACCEPTANCE.json" \
  "$SOURCE_JOB_ID" "$SIAB_SOURCE_COMMIT" "$BASIS_OPT_VALIDATOR_SHA256" \
  "$DATASET_ROOT" "$manifest_sha_after" "$status_sha_after" \
  "$REFERENCE_VALIDATION_JSON" "$max_rss_observed" "$max_rss_kb" <<'PY'
import hashlib
import json
import sys

(
    validation_path,
    output_path,
    source_job_id,
    source_commit,
    validator_sha256,
    dataset_root,
    manifest_sha256,
    status_sha256,
    reference_path,
    max_rss_kb,
    max_rss_limit_kb,
) = sys.argv[1:]
validation = json.load(open(validation_path, encoding="ascii"))
with open(reference_path, "rb") as handle:
    reference_sha256 = hashlib.sha256(handle.read()).hexdigest()
payload = {
    "status": "success_recovered_without_physics_rerun",
    "source_job_id": source_job_id,
    "siab_source_commit": source_commit,
    "validator_sha256": validator_sha256,
    "dataset_root": dataset_root,
    "manifest_sha256": manifest_sha256,
    "status_sha256": status_sha256,
    "reference_validation_sha256": reference_sha256,
    "validation_parity": "pass",
    "max_rss_kb": int(max_rss_kb),
    "max_rss_limit_kb": int(max_rss_limit_kb),
    "validation": validation,
}
with open(output_path, "w", encoding="ascii") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

{
  echo status=success
  echo purpose=read_only_streaming_basis_opt_v1_validation_recovery
  echo source_job_id=$SOURCE_JOB_ID
  echo siab_source_commit=$SIAB_SOURCE_COMMIT
  echo expected_abacus_commit=$EXPECTED_ABACUS_COMMIT
  echo dataset_root=$DATASET_ROOT
  echo max_rss_kb=$max_rss_observed
  echo max_rss_limit_kb=$max_rss_kb
  echo manifest_sha256=$manifest_sha_after
  echo status_sha256=$status_sha_after
  sha256sum "$runner" "$validator" "$REFERENCE_VALIDATION_JSON" \
    "$OUTPUT_ROOT/BASIS_OPT_VALIDATION.json" \
    "$OUTPUT_ROOT/RECOVERY_ACCEPTANCE.json" "$OUTPUT_ROOT/validator.time"
} > "$OUTPUT_ROOT/PROVENANCE.txt"

echo "C_SOLID_FD8_VALIDATION_RECOVERY_OK output=$OUTPUT_ROOT"
