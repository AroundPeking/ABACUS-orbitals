#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=siab_compact
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00

set -euo pipefail

source_root=/work1/ghj/sternheimer_abacus_tests/siab_compact_response_source_v1_20260729
asset_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_selection_assets_v1_20260722
target_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_h2_channel_mpi_prod_v1_20260726
campaign_root=/work1/ghj/sternheimer_abacus_tests/siab_compact_response_campaign_v1_20260729
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

compact=$source_root/SIAB/example_H_sternheimer/compact_response_selection
greedy=$source_root/SIAB/example_H_sternheimer/greedy_response_selection
optimizer=$source_root/SIAB/opt_orb_pytorch_dpsi/main.py
driver=$greedy/run_response_selection.py
template=$greedy/optimizer_template.json
baseline=$asset_root/H_TZDP_8au_ORBITAL_RESULTS.txt

lanes=(tail_0p00 tail_0p10 tail_0p30)
configs=(
  selection_config_tail_0p00.json
  selection_config_tail_0p10.json
  selection_config_tail_0p30.json
)
task=${SLURM_ARRAY_TASK_ID:?}
lane=${lanes[$task]}
config=$compact/${configs[$task]}
run_root=$campaign_root/$lane
output_root=$run_root/campaign

atom_dir=$target_root/producer_atom
h2_dir=$target_root/producer_h2
atom_validation=$atom_dir/target_validation.json
h2_validation=$h2_dir/target_validation.json
atom_target=$atom_dir/OUT.H_SIAB_GREEDY_ATOM_WFULL_NF16_E50/sternheimer_matrix.dat
h2_target=$h2_dir/OUT.H2_SIAB_GREEDY_R074085_WFULL_NF16_E50/sternheimer_matrix.dat
atom_status=$atom_dir/OUT.H_SIAB_GREEDY_ATOM_WFULL_NF16_E50/STERNHEIMER_SIAB_STATUS.dat
h2_status=$h2_dir/OUT.H2_SIAB_GREEDY_R074085_WFULL_NF16_E50/STERNHEIMER_SIAB_STATUS.dat

test ! -e "$run_root"
mkdir -p "$run_root"
for path in \
  "$driver" "$optimizer" "$config" "$template" "$baseline" \
  "$atom_validation" "$h2_validation" "$atom_target" "$h2_target" \
  "$atom_status" "$h2_status"; do
  test -s "$path"
done
test -s "$source_root/SOURCE_COMMIT"
test -s "$source_root/SOURCE_SHA256SUMS"

cd "$source_root"
sha256sum -c SOURCE_SHA256SUMS
printf '%s  %s\n' \
  8caf4b153232fbc6f6104ba434643bc9b0f481fd6b3d7af43cb091957418cae9 \
  "$baseline" | sha256sum -c -
printf '%s  %s\n' \
  6286069e96003c996c9e3231d4b700488a93edc7f5fe07f58becca44ad9c6081 \
  "$atom_target" | sha256sum -c -
printf '%s  %s\n' \
  1f4263864a20bf0e10f48f562dbe1af12a44fb607406275910165e508a778804 \
  "$h2_target" | sha256sum -c -

for status in "$atom_status" "$h2_status"; do
  grep -q '^status success$' "$status"
  grep -q '^nfreq 16$' "$status"
  grep -q '^sternheimer_frequency_mpi yes$' "$status"
  grep -q '^sternheimer_channel_mpi yes$' "$status"
  grep -q '^all_converged yes$' "$status"
done

export PYTHONPATH=$source_root/SIAB/opt_orb_pytorch_dpsi:$greedy
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_DYNAMIC=FALSE
export MPLCONFIGDIR=$run_root/.mplconfig

echo "date_start=$(date +%F_%T)"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "lane=$lane"
echo "omp_threads=$OMP_NUM_THREADS"
echo "source_commit=$(cat "$source_root/SOURCE_COMMIT")"
sha256sum "$driver" "$optimizer" "$config" "$template" "$baseline" \
  "$atom_target" "$h2_target"

cd "$source_root/SIAB/tests"
"$python" -m unittest discover -v > "$run_root/unit_tests.log" 2>&1
grep -Eq '^Ran [0-9]+ tests in ' "$run_root/unit_tests.log"
grep -q '^OK$' "$run_root/unit_tests.log"

cd "$source_root"
sha256sum -c SOURCE_SHA256SUMS
/usr/bin/time -v "$python" "$driver" \
  --config "$config" \
  --optimizer-template "$template" \
  --asset-root "$asset_root" \
  --baseline "$baseline" \
  --atom-target "$atom_target" \
  --multicenter-target "$h2_target" \
  --optimizer "$optimizer" \
  --python "$python" \
  --output "$output_root"

"$python" - "$output_root" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
campaign = json.loads((root / "campaign_manifest.json").read_text())
manifest = json.loads((root / "frozen/selection_manifest.json").read_text())
if campaign["status"] != "ao_budget_reached":
    raise SystemExit("compact campaign did not stop at its AO budget")
if campaign["targets"] != [
    {"family": "atom", "path": campaign["targets"][0]["path"], "role": "physical"},
    {"family": "multicenter", "path": campaign["targets"][1]["path"], "role": "physical"},
]:
    raise SystemExit("compact campaign used a nonphysical target")
if manifest["config"]["max_ao_per_atom"] != 48 or not manifest["steps"]:
    raise SystemExit("compact manifest has no frozen AO-budget frontier")
required = {
    "sternheimer", "radial_tail", "regularization_locality", "total",
    "max_st_condition", "max_locality_condition",
}
for step in manifest["steps"]:
    if step["ao_function_count"] > 48:
        raise SystemExit("frozen compact step exceeds 48 AO/H")
    if set(step.get("optimization_metrics", {})) != required:
        raise SystemExit("frozen compact step is missing optimization metrics")
print(json.dumps({
    "status": campaign["status"],
    "steps": len(manifest["steps"]),
    "last_ao": manifest["steps"][-1]["ao_function_count"],
}, sort_keys=True))
PY

sha256sum "$output_root/campaign_manifest.json" \
  "$output_root/frozen/selection_manifest.json"
echo "date_end=$(date +%F_%T)"
