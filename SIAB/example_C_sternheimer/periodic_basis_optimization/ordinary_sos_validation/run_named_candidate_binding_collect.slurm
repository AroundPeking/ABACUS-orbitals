#!/bin/bash
#SBATCH --job-name=c2g_bind
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=00:15:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c2g-binding-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c2g-binding-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NAMED_BINDING_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
manifest=$candidate_root/CANDIDATE.json
collector=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/collect_named_candidate_sos_binding.py
atom=$run_root/atom-sos/RESULT_SUMMARY.txt
solid=$run_root/solid-sos/RESULT_SUMMARY.txt
work=$run_root/binding
reference_ev_per_c=6.902326

test ! -e "$work"
test -s "$manifest" && test -s "$collector" && test -s "$atom" && test -s "$solid"
grep -qx 'status success' "$atom"
grep -qx 'status success' "$solid"
mkdir -p "$work"
python3 "$collector" --atom-sos "$atom" --solid-sos "$solid" --delta-reference-ev-per-c "$reference_ev_per_c" --output "$work/RESULT.json" > "$work/collector.out"
python3 - "$work/RESULT.json" <<'PY' > "$work/RESULT_SUMMARY.txt"
import json, math, sys
payload = json.load(open(sys.argv[1], encoding="ascii"))
assert payload["status"] == "success"
for key in ("sos_total_binding_ev_per_c", "difference_from_delta_ev_per_c"):
    assert math.isfinite(payload[key])
print("status success")
print("quantity solid_binding_energy_per_carbon_atom")
print(f"sos_total_binding_ev_per_c {payload['sos_total_binding_ev_per_c']:.12f}")
print(f"delta_st_reference_ev_per_c {payload['delta_st_reference_ev_per_c']:.12f}")
print(f"difference_from_delta_ev_per_c {payload['difference_from_delta_ev_per_c']:.12f}")
print(f"difference_from_delta_kcal_mol_per_c {payload['difference_from_delta_kcal_mol_per_c']:.12f}")
print(f"basis_full_body_gate {payload['basis_full_body_gate']}")
PY
{
  echo status=success
  echo fixed_reference_ev_per_c=$reference_ev_per_c
  sha256sum "$0" "$manifest" "$collector" "$atom" "$solid" "$work/RESULT.json" "$work/RESULT_SUMMARY.txt"
} > "$work/provenance.txt"
echo "C_NAMED_BINDING_OK result=$work/RESULT.json"
