#!/bin/bash
#SBATCH --job-name=c_fd8_q_std_recover
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=00:15:00

set -euo pipefail
trap 'status=$?; echo "C_SOLID_FD8_STANDARD_RECOVERY_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${WORK_ROOT:?missing completed q work root}"
: "${DATASET_ROOT:?missing completed basis-opt-v1 dataset}"
: "${OUTPUT_ROOT:?missing immutable recovery output root}"
: "${SOURCE_JOB_ID:?missing physical source job id}"
: "${SOURCE_ARRAY_JOB_ID:?missing source array job id}"
: "${SOURCE_SCHEDULER_STATE:?missing verified source scheduler state}"
: "${SOURCE_SCHEDULER_EXIT_CODE:?missing verified source scheduler exit code}"
: "${SIAB_SOURCE_ROOT:?missing immutable SIAB source deployment}"
: "${SIAB_SOURCE_COMMIT:?missing SIAB source commit}"
: "${STANDARD_VALIDATOR_SHA256:?missing standard validator SHA256}"
: "${BASIS_OPT_VALIDATION_JSON:?missing completed streaming validation JSON}"
: "${EXPECTED_FREQUENCY_GRID:?missing frozen q1 frequency grid}"
: "${EXPECTED_FREQUENCY_GRID_SHA256:?missing frozen q1 frequency-grid SHA256}"
: "${EXPECTED_ABACUS_COMMIT:?missing dataset ABACUS commit}"
: "${EXPECTED_ABACUS_SHA256:?missing ABACUS executable SHA256}"
: "${EXPECTED_ORBITAL_SHA256:?missing orbital SHA256}"
: "${EXPECTED_PSEUDOPOTENTIAL_SHA256:?missing pseudopotential SHA256}"
: "${EXPECTED_IQ:?missing selected q index}"
: "${EXPECTED_Q_WEIGHT:?missing q-star weight}"

workflow=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
runner=$workflow/recover_c_solid_fd8_q_standard_validation_dfdcu.slurm
validator=$workflow/validate_c_solid_fd8_q_dataset.py
validator_python=${C_BASIS_PYTHON:-/public/home/ghj/.conda/envs/ds092/bin/python}
input=$WORK_ROOT/INPUT
abacus_output=$WORK_ROOT/abacus.out
abacus_time=$WORK_ROOT/abacus.time
source_provenance=$WORK_ROOT/RUN_PROVENANCE.txt

test "$SOURCE_SCHEDULER_STATE" = FAILED
test "$SOURCE_SCHEDULER_EXIT_CODE" = 1:0
test -d "$WORK_ROOT"
test -d "$DATASET_ROOT"
test -s "$DATASET_ROOT/manifest.dat"
test -s "$DATASET_ROOT/status.dat"
test -s "$BASIS_OPT_VALIDATION_JSON"
test -s "$EXPECTED_FREQUENCY_GRID"
test -s "$input"
test -s "$abacus_output"
test -s "$abacus_time"
test -s "$source_provenance"
test -s "$runner"
test "$(sha256sum "$validator" | awk '{print $1}')" = "$STANDARD_VALIDATOR_SHA256"
test "$(sha256sum "$EXPECTED_FREQUENCY_GRID" | awk '{print $1}')" = "$EXPECTED_FREQUENCY_GRID_SHA256"
test -x "$validator_python"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

grep -q '^ FINISH Time  :' "$abacus_output"
grep -q '^ TOTAL  Time  :' "$abacus_output"
grep -qx 'status success' "$DATASET_ROOT/status.dat"
grep -qx 'all_converged yes' "$DATASET_ROOT/status.dat"

manifest_sha_before=$(sha256sum "$DATASET_ROOT/manifest.dat" | awk '{print $1}')
status_sha_before=$(sha256sum "$DATASET_ROOT/status.dat" | awk '{print $1}')
streaming_validation_sha_before=$(sha256sum "$BASIS_OPT_VALIDATION_JSON" | awk '{print $1}')
input_sha_before=$(sha256sum "$input" | awk '{print $1}')
abacus_output_sha_before=$(sha256sum "$abacus_output" | awk '{print $1}')

"$validator_python" - "$BASIS_OPT_VALIDATION_JSON" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="ascii") as handle:
    payload = json.load(handle)
if payload.get("status") != "success":
    raise SystemExit("streaming validation did not succeed")
for key in (
    "entries",
    "kpoints",
    "frequencies",
    "raw_auxiliary_dimension",
    "whitened_auxiliary_rank",
):
    value = float(payload[key])
    if not math.isfinite(value) or value <= 0:
        raise SystemExit("invalid streaming validation field: " + key)
PY

expected_auxiliary_sha=$("$validator_python" - "$EXPECTED_ORBITAL_SHA256" <<'PY'
import hashlib
import struct
import sys

digest = hashlib.sha256()
digest.update(b"ABACUS_STERNHEIMER_AUXILIARY_BASIS_DEFINITION_V1")
digest.update(sys.argv[1].encode("ascii"))
digest.update(struct.pack(">d", 1.0e-6))
digest.update(struct.pack(">d", 4.0))
digest.update(struct.pack(">Q", 0))
print(digest.hexdigest())
PY
)

"$validator_python" "$validator" \
  --manifest "$DATASET_ROOT/manifest.dat" \
  --status "$DATASET_ROOT/status.dat" \
  --input "$input" \
  --frequency-grid-output "$OUTPUT_ROOT/FREQUENCY_GRID.tsv" \
  --frequency-grid-source frozen_q1_greenx_minimax \
  --expected-frequency-grid "$EXPECTED_FREQUENCY_GRID" \
  --expected-abacus-commit "$EXPECTED_ABACUS_COMMIT" \
  --expected-executable-sha256 "$EXPECTED_ABACUS_SHA256" \
  --expected-orbital-sha256 "$EXPECTED_ORBITAL_SHA256" \
  --expected-pseudopotential-sha256 "$EXPECTED_PSEUDOPOTENTIAL_SHA256" \
  --expected-auxiliary-basis-sha256 "$expected_auxiliary_sha" \
  --expected-iq "$EXPECTED_IQ" \
  --expected-q-weight "$EXPECTED_Q_WEIGHT" \
  > "$OUTPUT_ROOT/STANDARD_PROTOCOL_VALIDATION.json"

test "$(sha256sum "$OUTPUT_ROOT/FREQUENCY_GRID.tsv" | awk '{print $1}')" = "$EXPECTED_FREQUENCY_GRID_SHA256"
test "$(sha256sum "$DATASET_ROOT/manifest.dat" | awk '{print $1}')" = "$manifest_sha_before"
test "$(sha256sum "$DATASET_ROOT/status.dat" | awk '{print $1}')" = "$status_sha_before"
test "$(sha256sum "$BASIS_OPT_VALIDATION_JSON" | awk '{print $1}')" = "$streaming_validation_sha_before"
test "$(sha256sum "$input" | awk '{print $1}')" = "$input_sha_before"
test "$(sha256sum "$abacus_output" | awk '{print $1}')" = "$abacus_output_sha_before"

"$validator_python" - \
  "$OUTPUT_ROOT/STANDARD_PROTOCOL_VALIDATION.json" \
  "$BASIS_OPT_VALIDATION_JSON" \
  "$OUTPUT_ROOT/STANDARD_PROTOCOL_RECOVERY_ACCEPTANCE.json" \
  "$SOURCE_JOB_ID" "$SOURCE_ARRAY_JOB_ID" \
  "$SOURCE_SCHEDULER_STATE" "$SOURCE_SCHEDULER_EXIT_CODE" \
  "$SIAB_SOURCE_COMMIT" "$STANDARD_VALIDATOR_SHA256" \
  "$DATASET_ROOT" "$manifest_sha_before" "$status_sha_before" \
  "$streaming_validation_sha_before" "$abacus_output_sha_before" <<'PY'
import hashlib
import json
import sys

(
    standard_path,
    streaming_path,
    output_path,
    source_job_id,
    source_array_job_id,
    source_scheduler_state,
    source_scheduler_exit_code,
    source_commit,
    validator_sha256,
    dataset_root,
    manifest_sha256,
    status_sha256,
    streaming_validation_sha256,
    abacus_output_sha256,
) = sys.argv[1:]
with open(standard_path, encoding="ascii") as handle:
    standard = json.load(handle)
with open(streaming_path, encoding="ascii") as handle:
    streaming = json.load(handle)
if standard.get("status") != "success" or streaming.get("status") != "success":
    raise SystemExit("recovered validations are not successful")
with open(standard_path, "rb") as handle:
    standard_sha256 = hashlib.sha256(handle.read()).hexdigest()
payload = {
    "status": "success_recovered_without_physics_rerun",
    "recovery_reason": "post_physics_validator_interpreter_mismatch",
    "physical_abacus_completion": "pass",
    "source_job_id": source_job_id,
    "source_array_job_id": source_array_job_id,
    "source_scheduler_state": source_scheduler_state,
    "source_scheduler_exit_code": source_scheduler_exit_code,
    "siab_source_commit": source_commit,
    "standard_validator_sha256": validator_sha256,
    "dataset_root": dataset_root,
    "manifest_sha256": manifest_sha256,
    "status_sha256": status_sha256,
    "streaming_validation_sha256": streaming_validation_sha256,
    "abacus_output_sha256": abacus_output_sha256,
    "standard_protocol_validation_sha256": standard_sha256,
    "standard_protocol_validation": standard,
}
with open(output_path, "w", encoding="ascii") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

{
  echo status=success
  echo purpose=read_only_standard_protocol_validation_recovery
  echo source_job_id=$SOURCE_JOB_ID
  echo source_array_job_id=$SOURCE_ARRAY_JOB_ID
  echo source_scheduler_state=$SOURCE_SCHEDULER_STATE
  echo source_scheduler_exit_code=$SOURCE_SCHEDULER_EXIT_CODE
  echo siab_source_commit=$SIAB_SOURCE_COMMIT
  echo dataset_root=$DATASET_ROOT
  echo manifest_sha256=$manifest_sha_before
  echo status_sha256=$status_sha_before
  echo streaming_validation_sha256=$streaming_validation_sha_before
  echo abacus_output_sha256=$abacus_output_sha_before
  sha256sum "$runner" "$validator" "$EXPECTED_FREQUENCY_GRID" \
    "$OUTPUT_ROOT/FREQUENCY_GRID.tsv" \
    "$OUTPUT_ROOT/STANDARD_PROTOCOL_VALIDATION.json" \
    "$OUTPUT_ROOT/STANDARD_PROTOCOL_RECOVERY_ACCEPTANCE.json"
} > "$OUTPUT_ROOT/PROVENANCE.txt"

echo "C_SOLID_FD8_STANDARD_RECOVERY_OK output=$OUTPUT_ROOT"
