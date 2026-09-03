#!/bin/bash
#SBATCH --job-name=c_fd8_q_accept
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=00:15:00

set -euo pipefail
trap 'status=$?; echo "C_SOLID_FD8_VALIDATION_ACCEPTANCE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${DATASET_ROOT:?missing existing basis-opt-v1 dataset}"
: "${OUTPUT_ROOT:?missing immutable acceptance output root}"
: "${PHYSICAL_SOURCE_JOB_ID:?missing physical source job id}"
: "${SOURCE_VALIDATOR_JOB_ID:?missing completed validator job id}"
: "${SIAB_SOURCE_ROOT:?missing immutable SIAB source deployment}"
: "${SIAB_SOURCE_COMMIT:?missing SIAB source commit}"
: "${BASIS_OPT_VALIDATOR_SHA256:?missing streaming validator SHA256}"
: "${BASIS_OPT_ACCEPTOR_SHA256:?missing acceptance script SHA256}"
: "${BASIS_OPT_VALIDATION_JSON:?missing completed validation JSON}"
: "${BASIS_OPT_VALIDATION_SHA256:?missing completed validation SHA256}"
: "${VALIDATOR_TIME_FILE:?missing completed validator time file}"
: "${VALIDATOR_TIME_SHA256:?missing completed validator time SHA256}"
: "${REFERENCE_VALIDATION_JSON:?missing cross-host reference validation JSON}"
: "${EXPECTED_MANIFEST_SHA256:?missing dataset manifest SHA256}"
: "${EXPECTED_STATUS_SHA256:?missing dataset status SHA256}"

workflow=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
runner=$workflow/accept_c_solid_fd8_q_streaming_validation_recovery.slurm
validator=$workflow/validate_periodic_basis_opt_streaming.py
acceptor=$workflow/accept_periodic_basis_opt_streaming_validation.py
python=${C_BASIS_PYTHON:-/data/home/df_iopcas_ghj/app/miniconda3/bin/python}
max_rss_limit_kb=${MAX_VALIDATOR_RSS_KB:-4194304}

test -d "$DATASET_ROOT"
test -s "$DATASET_ROOT/manifest.dat"
test -s "$DATASET_ROOT/status.dat"
test -s "$BASIS_OPT_VALIDATION_JSON"
test -s "$VALIDATOR_TIME_FILE"
test -s "$REFERENCE_VALIDATION_JSON"
test -s "$runner"
test "$(sha256sum "$validator" | awk '{print $1}')" = "$BASIS_OPT_VALIDATOR_SHA256"
test "$(sha256sum "$acceptor" | awk '{print $1}')" = "$BASIS_OPT_ACCEPTOR_SHA256"
test "$(sha256sum "$BASIS_OPT_VALIDATION_JSON" | awk '{print $1}')" = "$BASIS_OPT_VALIDATION_SHA256"
test "$(sha256sum "$VALIDATOR_TIME_FILE" | awk '{print $1}')" = "$VALIDATOR_TIME_SHA256"
test "$(sha256sum "$DATASET_ROOT/manifest.dat" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256"
test "$(sha256sum "$DATASET_ROOT/status.dat" | awk '{print $1}')" = "$EXPECTED_STATUS_SHA256"
test -x "$python"
test "$max_rss_limit_kb" -gt 0
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

max_rss_observed=$(sed -n \
  's/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*//p' \
  "$VALIDATOR_TIME_FILE")
test -n "$max_rss_observed"

"$python" "$acceptor" \
  "$BASIS_OPT_VALIDATION_JSON" "$REFERENCE_VALIDATION_JSON" \
  --max-rss-kb "$max_rss_observed" \
  --max-rss-limit-kb "$max_rss_limit_kb" \
  --source-validator-job-id "$SOURCE_VALIDATOR_JOB_ID" \
  --physical-source-job-id "$PHYSICAL_SOURCE_JOB_ID" \
  --siab-source-commit "$SIAB_SOURCE_COMMIT" \
  --validator-sha256 "$BASIS_OPT_VALIDATOR_SHA256" \
  --dataset-root "$DATASET_ROOT" \
  --manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --status-sha256 "$EXPECTED_STATUS_SHA256" \
  > "$OUTPUT_ROOT/RECOVERY_ACCEPTANCE.json"

{
  echo status=success
  echo purpose=accept_completed_streaming_validation_without_dataset_rescan
  echo physical_source_job_id=$PHYSICAL_SOURCE_JOB_ID
  echo source_validator_job_id=$SOURCE_VALIDATOR_JOB_ID
  echo siab_source_commit=$SIAB_SOURCE_COMMIT
  echo dataset_root=$DATASET_ROOT
  echo max_rss_kb=$max_rss_observed
  echo max_rss_limit_kb=$max_rss_limit_kb
  echo manifest_sha256=$EXPECTED_MANIFEST_SHA256
  echo status_sha256=$EXPECTED_STATUS_SHA256
  sha256sum "$runner" "$validator" "$acceptor" \
    "$BASIS_OPT_VALIDATION_JSON" "$VALIDATOR_TIME_FILE" \
    "$REFERENCE_VALIDATION_JSON" "$OUTPUT_ROOT/RECOVERY_ACCEPTANCE.json"
} > "$OUTPUT_ROOT/PROVENANCE.txt"

echo "C_SOLID_FD8_VALIDATION_ACCEPTANCE_OK output=$OUTPUT_ROOT"
