#!/bin/bash

set -euo pipefail
trap 'status=$?; echo "C24_RESPONSE_PCA_MATCHED_DELTA_RELEASE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
validation=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
response_script=$validation/run_selected_candidate_response_pca_matched_delta_q1_55d25e3c9.slurm
consumer_script=$validation/run_selected_candidate_response_pca_matched_delta_reader_d4810f73.slurm
qavg_script=$validation/run_selected_candidate_response_pca_matched_headwing_qavg_d4810f73.slurm
root=$campaign/runs/product-pca-20260825/matched-delta-response-aware-fixed-prefix-nfreq6-55d25e3c9
q1=$root/q1
sos_root=$campaign/runs/product-pca-20260825/ordinary-sos-response-aware-full-bz-nfreq6-d4810f73
frequency=$sos_root/SELECTED_SOS_FREQUENCY_GRID.dat
sos_summary=$sos_root/POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt
chain=$root/MATCHED_DELTA_CHAIN.txt
pending=$root/MATCHED_DELTA_CHAIN.pending
lock=$root/.release.lock
q1_job=${Q1_RESPONSE_JOB_ID:?missing completed q1 response job id}
headwing_job=${HEADWING_JOB_ID:?missing completed selected headwing job id}
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
runtime_gate_hours=18

test -s "$response_script" && test -s "$consumer_script" && test -s "$qavg_script"
test "$headwing_job" -gt 0
test -s "$frequency" && test -s "$sos_summary" && test -x "$python"
grep -qx 'status success' "$sos_summary"
test -d "$q1"
grep -qx 'status=success' "$q1/RUN_PROVENANCE.txt"
grep -qx 'expected_naux=522' "$q1/RUN_PROVENANCE.txt"
grep -qx 'all_converged yes' "$(find "$q1" -path '*/OUT.*/STERNHEIMER_CHI0.dat' -type f -print -quit)"
test ! -e "$chain" && test ! -e "$pending"
mkdir "$lock"

read -r q1_state q1_exit < <(sacct -n -X -j "$q1_job" --format=State,ExitCode -P | awk -F'|' 'NF>=2 && $1!="" {print $1, $2; exit}')
test "$q1_state" = COMPLETED && test "$q1_exit" = 0:0
read -r headwing_state headwing_exit < <(sacct -n -X -j "$headwing_job" --format=State,ExitCode -P | awk -F'|' 'NF>=2 && $1!="" {print $1, $2; exit}')
test "$headwing_state" = COMPLETED && test "$headwing_exit" = 0:0
test -s "$q1/abacus.time"
q1_wall_seconds=$("$python" - "$q1/abacus.time" "$runtime_gate_hours" <<'PY'
import pathlib
import re
import sys
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
limit_hours = float(sys.argv[2])
match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\s]+)", text)
assert match is not None
parts = [float(value) for value in match.group(1).split(":")]
assert len(parts) in (2, 3)
seconds = parts[-1] + 60.0 * parts[-2] + (3600.0 * parts[0] if len(parts) == 3 else 0.0)
assert seconds <= limit_hours * 3600.0, (seconds, limit_hours)
print(f"{seconds:.6f}")
PY
)

report=$(find "$q1" -path '*/OUT.*/STERNHEIMER_CHI0.dat' -type f -print -quit)
mapfile -t representatives < <(awk '$1=="sternheimer_canonical_q_indices" {for(i=2;i<=NF;i++) print $i}' "$report")
test "${#representatives[@]}" -gt 1
declare -A seen=()
remaining=()
for iq in "${representatives[@]}"; do
  test "$iq" -ge 1 && test "$iq" -le 64
  test -z "${seen[$iq]:-}"
  seen[$iq]=1
  if test "$iq" != 1; then test ! -e "$root/q${iq}"; remaining+=("$iq"); fi
done
test "${seen[1]:-0}" = 1 && test "${#remaining[@]}" -gt 0
remaining_csv=$(IFS=,; echo "${remaining[*]}")
canonical_csv=$(IFS=,; echo "${representatives[*]}")
{
  echo status=pending
  echo q1_response_job=$q1_job
  echo headwing_job=$headwing_job
  echo canonical_q_indices=$canonical_csv
  echo remaining_q_indices=$remaining_csv
  echo runtime_gate_hours=$runtime_gate_hours
  echo q1_wall_seconds=$q1_wall_seconds
  echo frequency_grid_sha256=$(sha256sum "$frequency" | awk '{print $1}')
  sha256sum "$response_script" "$consumer_script" "$qavg_script" "$frequency" "$sos_summary" "$report"
} > "$pending"

sbatch --test-only --array="$remaining_csv%1" --export=ALL,SIAB_SOURCE_ROOT="$code" "$response_script" > "$root/REMAINING_Q_TEST_ONLY.txt"
response_job=$(sbatch --parsable --array="$remaining_csv%1" --export=ALL,SIAB_SOURCE_ROOT="$code" "$response_script")
response_job=${response_job%%;*}
test "$response_job" -gt 0
echo response_array_job=$response_job >> "$pending"
sbatch --test-only --export=ALL,SIAB_SOURCE_ROOT="$code" "$consumer_script" > "$root/CONSUMER_TEST_ONLY.txt"
consumer_job=$(sbatch --parsable --dependency=afterok:"$response_job" --export=ALL,SIAB_SOURCE_ROOT="$code" "$consumer_script")
consumer_job=${consumer_job%%;*}
test "$consumer_job" -gt 0
echo consumer_job=$consumer_job >> "$pending"
sbatch --test-only --dependency=afterok:"$consumer_job":"$headwing_job" \
  --export=ALL,SIAB_SOURCE_ROOT="$code" "$qavg_script" > "$root/QAVG_TEST_ONLY.txt"
qavg_job=$(sbatch --parsable --dependency=afterok:"$consumer_job":"$headwing_job" \
  --export=ALL,SIAB_SOURCE_ROOT="$code" "$qavg_script")
qavg_job=${qavg_job%%;*}
test "$qavg_job" -gt 0
echo qavg_job=$qavg_job >> "$pending"
echo status=submitted >> "$pending"
mv "$pending" "$chain"
echo "C24_RESPONSE_PCA_MATCHED_DELTA_RELEASE_OK response_job=$response_job consumer_job=$consumer_job qavg_job=$qavg_job"
