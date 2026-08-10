#!/bin/bash

set -euo pipefail

campaign_root=${1:?campaign root is required}
threads=${2:-10}
script_dir=$(cd "$(dirname "$0")" && pwd)
runner="$script_dir/run_case_66.sh"
test -x "$runner"
test "$((6 * threads))" -le "$(nproc)"

pairs=(
  baseline_tzdp:H
  baseline_tzdp:H2
  baseline_tzdp:H_ghost
  optimized_3s3p2d:H
  optimized_3s3p2d:H2
  optimized_3s3p2d:H_ghost
)
pids=()
labels=()
for pair in "${pairs[@]}"; do
  lane=${pair%%:*}
  case_name=${pair#*:}
  "$runner" "$campaign_root" "$lane" "$case_name" "$threads" \
    > "$campaign_root/driver.$lane.$case_name.stdout" \
    2> "$campaign_root/driver.$lane.$case_name.stderr" &
  pids+=("$!")
  labels+=("$pair")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "failed: ${labels[$index]}" >&2
    status=1
  fi
done
exit "$status"

