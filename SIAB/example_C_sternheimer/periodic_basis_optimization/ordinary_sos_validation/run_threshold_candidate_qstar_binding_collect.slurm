#!/bin/bash
#SBATCH --job-name=c_thr_bind
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-threshold-binding-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-threshold-binding-%j.err

set -euo pipefail
trap 'status=$?; echo "C_THRESHOLD_BINDING_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

run_root=${RUN_ROOT:?missing immutable run root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
collector=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/collect_threshold_candidate_qstar_binding.py
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$run_root/binding
atom=$run_root/atom/RESULT_SUMMARY.txt
solid_root=${SOLID_SOS_ROOT:-$run_root/solid-sos}
solid=$solid_root/RESULT_SUMMARY.txt

test ! -e "$work"
test -s "$collector" && test -x "$python"
grep -qx 'success' "$run_root/atom/STATUS"
grep -qx 'success' "$solid_root/STATUS"
test -s "$atom" && test -s "$solid"
mkdir -p "$work"
"$python" "$collector" \
  --atom-summary "$atom" \
  --solid-summary "$solid" \
  --delta-reference-ev-per-c 6.902326 \
  --tolerance-ev-per-c 0.1 \
  --output "$work/RESULT.json" > "$work/collect.out"
"$python" - "$work/RESULT.json" <<'PY'
import json, math, sys
payload = json.load(open(sys.argv[1], encoding="ascii"))
assert payload["status"] == "success"
assert math.isfinite(payload["sos_total_binding_ev_per_c"])
assert payload["binding_tolerance_ev_per_c"] == 0.1
PY
{
  echo status=success
  echo atom_job=${ATOM_JOB_ID:?missing atom job id}
  echo qstar_array_job=${QSTAR_ARRAY_JOB_ID:?missing q-star array job id}
  echo solid_sos_job=${SOLID_SOS_JOB_ID:?missing solid SOS job id}
  echo solid_sos_root=$solid_root
  echo delta_st_reference_ev_per_c=6.902326
  echo acceptance_tolerance_ev_per_c=0.1
  sha256sum "$collector" "$atom" "$solid" "$work/RESULT.json"
} > "$work/provenance.txt"
printf 'success\n' > "$work/STATUS"
cat "$work/RESULT.json"
