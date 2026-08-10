#!/bin/bash

set -euo pipefail

campaign_root=${1:?campaign root is required}
mpi_ranks=${2:-1}
max_parallel=${3:-6}
script_dir=$(cd "$(dirname "$0")" && pwd)
runner="$script_dir/run_case_66.sh"
test -x "$runner"
test "$((max_parallel * mpi_ranks))" -le "$(nproc)"

pairs=(
  baseline_tzdp:H
  baseline_tzdp:H2
  baseline_tzdp:H_ghost
  optimized_3s3p2d:H
  optimized_3s3p2d:H2
  optimized_3s3p2d:H_ghost
)
status=0
for ((start = 0; start < ${#pairs[@]}; start += max_parallel)); do
  pids=()
  labels=()
  for ((offset = 0; offset < max_parallel; offset += 1)); do
    index=$((start + offset))
    test "$index" -lt "${#pairs[@]}" || break
    pair=${pairs[$index]}
    lane=${pair%%:*}
    case_name=${pair#*:}
    "$runner" "$campaign_root" "$lane" "$case_name" "$mpi_ranks" \
      > "$campaign_root/driver.$lane.$case_name.stdout" \
      2> "$campaign_root/driver.$lane.$case_name.stderr" &
    pids+=("$!")
    labels+=("$pair")
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "failed: ${labels[$index]}" >&2
      status=1
    fi
  done
done
exit "$status"
