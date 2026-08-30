#!/bin/bash
#SBATCH --job-name=c_thr_qrecover
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-threshold-qstar-recovery-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-threshold-qstar-recovery-%j.err

set -euo pipefail
trap 'status=$?; echo "C_THRESHOLD_QSTAR_RECOVERY_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

run_root=${RUN_ROOT:?missing immutable run root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
source_commit=${SIAB_SOURCE_COMMIT:?missing exact SIAB source commit}
failed=${FAILED_SOLID_SOS_ROOT:?missing failed solid SOS root}
failed_job=${FAILED_SOLID_SOS_JOB_ID:?missing failed solid SOS job id}
work=${ACCEPTED_SOLID_SOS_ROOT:?missing accepted solid SOS root}
qstar_gate=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/sparse_qstar_sos_gate.py
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
q1=$run_root/solid-grid/q1

[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test "${source_commit:0:8}" = "${source_root##*-}"
test "$failed" = "$run_root/solid-sos"
test "$work" != "$failed" && test ! -e "$work"
test "$(sacct -n -X -j "$failed_job" --format=State,ExitCode -P | sed -n '1p')" = 'FAILED|2:0'
grep -qx 'status=running' "$failed/provenance.txt"
grep -q 'libRPA finished successfully' "$failed/librpa.out"
grep -q '^| Total EcRPA:' "$failed/librpa.out"
test -s "$failed/SOLID_SOS_FREQUENCY_GRID.dat"
test -s "$failed/SOLID_SOS_FREQUENCY_GRID.json"
test -s "$qstar_gate" && test -s "$q1/abacus.out"

mkdir -p "$work"
"$python" "$qstar_gate" \
  --sparse-output "allband=$failed/librpa.out" \
  --output-root "$work/qstar-gate" \
  --binding-tolerance-kcal-mol-per-c 0.1 \
  --provenance "source_commit=$source_commit" \
  --provenance "failed_solid_sos_job=$failed_job" > "$work/qstar_gate.out"
grep -qx 'success' "$work/qstar-gate/STATUS"
ecrpa_ha=$("$python" - "$work/qstar-gate/RESULT.json" <<'PY'
import json, math, sys
payload = json.load(open(sys.argv[1], encoding="ascii"))
assert payload["status"] == "success" and payload["sparse_qstar_gate"] == "pass"
assert len(payload["rows"]) == 1
value = float(payload["rows"][0]["qstar_reconstruction_ha"])
assert math.isfinite(value)
print(f"{value:.17g}")
PY
)
reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' "$q1/abacus.out")
orbital_sha=$(sed -n 's/^selected_orbital_sha256=//p' "$failed/provenance.txt")
naux=$(sed -n 's/^naux=//p' "$failed/provenance.txt")
frequency_sha=$(sha256sum "$failed/SOLID_SOS_FREQUENCY_GRID.dat" | awk '{print $1}')
"$python" - "$reference_ha" "$ecrpa_ha" "$naux" <<'PY'
import math, sys
assert all(math.isfinite(float(value)) for value in sys.argv[1:])
PY
cat > "$work/RESULT_SUMMARY.txt" <<EOF
status success
side solid
method sos
scope body_only_no_analytic_headwing_qstar_reconstructed
coulomb_kernel full_periodic_poisson
selected_orbital_sha256 $orbital_sha
frequency_grid_sha256 $frequency_sha
naux $naux
reference_ha $reference_ha
ecrpa_ha $ecrpa_ha
EOF
{
  echo status=success
  echo purpose=accept_completed_solid_qstar_librpa_after_provenance_wrapper_failure
  echo source_commit=$source_commit
  echo failed_solid_sos_job=$failed_job
  echo failed_solid_sos_root=$failed
  echo reference_ha=$reference_ha
  echo ecrpa_qstar_reconstructed_ha=$ecrpa_ha
  echo naux=$naux
  echo frequency_grid_sha256=$frequency_sha
  sha256sum "$0" "$qstar_gate" "$failed/librpa.in" "$failed/librpa.out" \
    "$failed/SOLID_SOS_FREQUENCY_GRID.dat" "$failed/SOLID_SOS_FREQUENCY_GRID.json" \
    "$work/RESULT_SUMMARY.txt" "$work/qstar-gate/RESULT.json"
} > "$work/provenance.txt"
printf 'success\n' > "$work/STATUS"
echo "C_THRESHOLD_QSTAR_RECOVERY_OK root=$work ecrpa_ha=$ecrpa_ha"
