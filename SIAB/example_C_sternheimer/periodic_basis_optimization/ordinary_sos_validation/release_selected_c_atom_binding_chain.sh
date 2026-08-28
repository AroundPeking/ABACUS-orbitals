#!/bin/bash

set -euo pipefail
trap 'status=$?; echo "C_SELECTED_ATOM_RELEASE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
base=$campaign/runs/product-pca-20260825
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
validation=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
producer_script=$validation/run_selected_c_atom_response_pca_producer_55d25e3c9.slurm
sos_script=$validation/run_selected_c_atom_sos_d4810f73.slurm
delta_script=$validation/run_selected_c_atom_matched_delta_55d25e3c9.slurm
reader_script=$validation/run_selected_c_atom_matched_delta_reader_d4810f73.slurm
collector=$validation/collect_selected_c_binding_energy.py
atom_root=$base/selected-c-atom-response-aware-product-pca-55d25e3c9
failed_producer=$atom_root/producer
producer=$atom_root/producer-retry1
sos=$atom_root/sos-nfreq6-d4810f73
delta=$atom_root/matched-delta-nfreq6-55d25e3c9
reader=$atom_root/matched-delta-reader-nfreq6-d4810f73
binding=$atom_root/binding-body
solid_sos=$base/ordinary-sos-response-aware-full-bz-nfreq6-d4810f73
solid_delta=$base/matched-delta-response-aware-fixed-prefix-reader-nfreq6-d4810f73
solid_reference=$base/response-aware-product-pca-q1-55d25e3c9
solid_delta_q1=$base/matched-delta-response-aware-fixed-prefix-nfreq6-55d25e3c9/q1
expected_orbital_sha=3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python

for file in "$producer_script" "$sos_script" "$delta_script" "$reader_script" "$collector"; do
  test -s "$file"
done
test -x "$python"
mkdir -p "$atom_root"
exec 9>"$atom_root/.release.lock"
if ! flock -n 9; then
  echo "refusing duplicate: selected C atom release lock is held" >&2
  exit 2
fi

job_state() {
  local job_id=$1 active
  active=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -1 || true)
  if test -n "$active"; then
    printf '%s active\n' "$active"
    return
  fi
  sacct -n -X -j "$job_id" --format=State,ExitCode -P | awk -F'|' 'NF>=2 && $1!="" {print $1, $2; exit}'
}

require_completed_job() {
  local label=$1 record=$2 job_id state exit_code
  test -s "$record"
  job_id=$(tr -d '[:space:]' < "$record")
  test "$job_id" -gt 0
  read -r state exit_code < <(job_state "$job_id")
  test "$state" = COMPLETED
  test "$exit_code" = 0:0
  echo "$label scheduler COMPLETED/0:0 job=$job_id"
}

refuse_named_active_job() {
  local job_name=$1
  if squeue -h -o '%j|%T' | awk -F'|' -v name="$job_name" '$1==name {found=1} END {exit !found}'; then
    echo "refusing duplicate: active scheduler job $job_name already exists" >&2
    exit 2
  fi
}

submit_unique_stage() {
  local label=$1 job_name=$2 script=$3 result_dir=$4 record=$5 output job_id temporary
  if test -e "$result_dir" || test -e "$record"; then
    echo "refusing duplicate: $label directory or job record already exists" >&2
    exit 2
  fi
  refuse_named_active_job "$job_name"
  sbatch --test-only --export=ALL,SIAB_SOURCE_ROOT="$code" "$script" > "$atom_root/${label}_TEST_ONLY.txt" 2>&1
  output=$(sbatch --parsable --export=ALL,SIAB_SOURCE_ROOT="$code" "$script")
  job_id=${output%%;*}
  test "$job_id" -gt 0
  temporary=$record.pending.$$
  printf '%s\n' "$job_id" > "$temporary"
  mv "$temporary" "$record"
  printf 'status submitted\nstage %s\njob_id %s\nscript_sha256 %s\n' \
    "$label" "$job_id" "$(sha256sum "$script" | awk '{print $1}')" \
    > "$atom_root/${label}_SUBMISSION.txt"
  echo "C_SELECTED_ATOM_STAGE_SUBMITTED stage=$label job=$job_id"
  exit 0
}

if ! test -e "$producer"; then
  test -d "$failed_producer"
  test -s "$failed_producer/OUT.C_ATOM_SELECTED_RESPONSE_PCA/STERNHEIMER_SIAB_STATUS.dat"
  grep -qx 'status failed' "$failed_producer/OUT.C_ATOM_SELECTED_RESPONSE_PCA/STERNHEIMER_SIAB_STATUS.dat"
  grep -Fqx 'reason sternheimer_mpi_layout=global_equation requires frequency MPI and channel MPI.' "$failed_producer/OUT.C_ATOM_SELECTED_RESPONSE_PCA/STERNHEIMER_SIAB_STATUS.dat"
  read -r failed_state failed_exit < <(job_state "$(tr -d '[:space:]' < "$atom_root/PRODUCER_JOB_ID.txt")")
  test "$failed_state" = FAILED && test "$failed_exit" = 1:0
  submit_unique_stage producer_retry1 c_atom_pca_retry1 "$producer_script" "$producer" "$atom_root/PRODUCER_RETRY1_JOB_ID.txt"
fi
grep -qx 'status=success' "$producer/provenance.txt"
grep -qx 'naux=261' "$producer/provenance.txt"
require_completed_job producer_retry1 "$atom_root/PRODUCER_RETRY1_JOB_ID.txt"

if ! test -e "$sos"; then
  submit_unique_stage sos c_atom_selected_sos "$sos_script" "$sos" "$atom_root/SOS_JOB_ID.txt"
fi
grep -qx 'status success' "$sos/RESULT_SUMMARY.txt"
grep -qx 'side atom' "$sos/RESULT_SUMMARY.txt"
grep -qx 'method sos' "$sos/RESULT_SUMMARY.txt"
require_completed_job sos "$atom_root/SOS_JOB_ID.txt"

if ! test -e "$delta"; then
  submit_unique_stage delta c_atom_selected_delta "$delta_script" "$delta" "$atom_root/DELTA_JOB_ID.txt"
fi
grep -qx 'status=success' "$delta/provenance.txt"
grep -qx 'all_converged=yes' "$delta/provenance.txt"
require_completed_job delta "$atom_root/DELTA_JOB_ID.txt"

if ! test -e "$reader"; then
  submit_unique_stage reader c_atom_delta_rpa "$reader_script" "$reader" "$atom_root/DELTA_READER_JOB_ID.txt"
fi
grep -qx 'status success' "$reader/RESULT_SUMMARY.txt"
grep -qx 'side atom' "$reader/RESULT_SUMMARY.txt"
grep -qx 'method delta_st' "$reader/RESULT_SUMMARY.txt"
require_completed_job reader "$atom_root/DELTA_READER_JOB_ID.txt"

if test -e "$binding"; then
  echo "refusing duplicate: binding result directory already exists" >&2
  exit 2
fi
test -s "$solid_sos/POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt"
test -s "$solid_sos/SELECTED_SOS_FREQUENCY_GRID.dat"
test -s "$solid_delta/RESULT_SUMMARY.txt"
test -s "$solid_reference/abacus.out"
test -s "$solid_delta_q1/abacus.out"
grep -qx 'status success' "$solid_sos/POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt"
grep -qx 'scope body_only_no_analytic_headwing' "$solid_sos/POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt"
grep -qx 'status success' "$solid_delta/RESULT_SUMMARY.txt"
grep -qx 'scope body_only_no_analytic_headwing' "$solid_delta/RESULT_SUMMARY.txt"

mkdir "$binding"
solid_sos_ec=$(awk '$1=="ecrpa_ha" {print $2}' "$solid_sos/POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt")
solid_delta_ec=$(awk '$1=="delta_st_ecrpa_ha" {print $2}' "$solid_delta/RESULT_SUMMARY.txt")
solid_reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' "$solid_reference/abacus.out")
solid_delta_reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' "$solid_delta_q1/abacus.out")
solid_frequency_sha=$(sha256sum "$solid_sos/SELECTED_SOS_FREQUENCY_GRID.dat" | awk '{print $1}')
"$python" - "$solid_sos_ec" "$solid_delta_ec" "$solid_reference_ha" "$solid_delta_reference_ha" <<'PY'
import math
import sys
values = list(map(float, sys.argv[1:]))
assert all(math.isfinite(value) for value in values)
assert math.isclose(values[2], values[3], rel_tol=0.0, abs_tol=1.0e-10), (values[2], values[3])
PY

cat > "$binding/SOLID_SOS_ENDPOINT.txt" <<EOF
status success
side solid
method sos
scope body_only_no_analytic_headwing
coulomb_kernel full_periodic_poisson
selected_orbital_sha256 $expected_orbital_sha
frequency_grid_sha256 $solid_frequency_sha
naux 522
reference_ha $solid_reference_ha
ecrpa_ha $solid_sos_ec
EOF
cat > "$binding/SOLID_DELTA_ENDPOINT.txt" <<EOF
status success
side solid
method delta_st
scope body_only_no_analytic_headwing
coulomb_kernel full_periodic_poisson
selected_orbital_sha256 $expected_orbital_sha
frequency_grid_sha256 $solid_frequency_sha
naux 522
reference_ha $solid_delta_reference_ha
ecrpa_ha $solid_delta_ec
EOF

"$python" "$collector" \
  --atom-sos "$sos/RESULT_SUMMARY.txt" \
  --atom-delta "$reader/RESULT_SUMMARY.txt" \
  --solid-sos "$binding/SOLID_SOS_ENDPOINT.txt" \
  --solid-delta "$binding/SOLID_DELTA_ENDPOINT.txt" \
  --output "$binding/SELECTED_C_BINDING_BODY_RESULT.json" \
  > "$binding/collector.out"
{
  echo status success
  echo scope body_only_no_analytic_headwing
  sha256sum "$collector" "$sos/RESULT_SUMMARY.txt" "$reader/RESULT_SUMMARY.txt" "$binding/SOLID_SOS_ENDPOINT.txt" "$binding/SOLID_DELTA_ENDPOINT.txt" "$binding/SELECTED_C_BINDING_BODY_RESULT.json"
} > "$binding/provenance.txt"

echo "C_SELECTED_ATOM_BINDING_BODY_OK result=$binding/SELECTED_C_BINDING_BODY_RESULT.json"
