#!/bin/bash

set -euo pipefail
trap 'status=$?; echo "C24_SELECTED_VALIDATION_RELEASE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
root=$campaign/runs/product-pca-20260825/selected-full-validation-release
chain=$root/SELECTED_VALIDATION_CHAIN.txt
pending=$root/SELECTED_VALIDATION_CHAIN.pending
lock=$root/.release.lock
optimizer_job=${OPTIMIZER_ARRAY_JOB_ID:?missing optimizer array job id}
heldout_job=${HELDOUT_JOB_ID:?missing heldout q3 job id}
export_script=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/export_selected_product_pca_candidate.sh
validation=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
grid_script=$validation/run_selected_candidate_full_bz_grid_coulomb.slurm
sos_script=$validation/run_selected_candidate_full_bz_reader_d4810f73.slurm
headwing_script=$validation/run_selected_candidate_headwing_input.slurm
q1_script=$validation/run_selected_candidate_matched_delta_response.slurm
heldout=$campaign/runs/product-pca-20260825/heldout-q3-fixed-prefix-layout-checkpoint-500
comparison=$heldout/COMPARISON_RESULT.json
selection_root=$campaign/runs/product-pca-20260825/selected-fixed-prefix-checkpoint-500
grid_root=$campaign/runs/product-pca-20260825/grid-coulomb-selected-fixed-prefix-full-bz-ad29464fd
sos_root=$campaign/runs/product-pca-20260825/ordinary-sos-selected-grid-full-bz-reader-fractional-nfreq6-d4810f73
headwing_root=$campaign/runs/product-pca-20260825/headwing-selected-fixed-prefix-ad29464fd
response_root=$campaign/runs/product-pca-20260825/matched-delta-selected-fixed-prefix-nfreq6-ad29464fd

for script in "$export_script" "$grid_script" "$sos_script" "$headwing_script" "$q1_script"; do
  test -s "$script"
done
read -r optimizer_state optimizer_exit < <(sacct -n -X -j "$optimizer_job" --format=State,ExitCode -P | awk -F'|' 'NF>=2 && $1!="" {print $1, $2; exit}')
read -r heldout_state heldout_exit < <(sacct -n -X -j "$heldout_job" --format=State,ExitCode -P | awk -F'|' 'NF>=2 && $1!="" {print $1, $2; exit}')
test "$optimizer_state" = COMPLETED
test "$optimizer_exit" = 0:0
test "$heldout_state" = COMPLETED
test "$heldout_exit" = 0:0
test -s "$comparison"
grep -qx 'status=success' "$heldout/provenance.txt"
test ! -e "$selection_root"
test ! -e "$grid_root"
test ! -e "$sos_root"
test ! -e "$headwing_root"
test ! -e "$response_root"

mkdir -p "$root"
test ! -e "$chain"
test ! -e "$pending"
mkdir "$lock"
{
  echo status=preparing
  echo optimizer_array_job=$optimizer_job
  echo heldout_job=$heldout_job
  echo siab_source_root=$code
  sha256sum "$export_script" "$grid_script" "$sos_script" "$headwing_script" "$q1_script" "$comparison"
} > "$pending"

OPTIMIZER_ARRAY_JOB_ID=$optimizer_job HELDOUT_JOB_ID=$heldout_job bash "$export_script" > "$root/export.out" 2>&1
selection=$selection_root/SELECTED_CANDIDATE.json
orbital=$selection_root/C_gga_10au_100Ry_selected_product_pca.orb
test -s "$selection"
test -s "$orbital"
grep -qx 'status=success' "$selection_root/provenance.txt"
grep -qx "optimizer_array_job=$optimizer_job" "$selection_root/provenance.txt"
grep -qx "heldout_job=$heldout_job" "$selection_root/provenance.txt"
sha256sum "$selection" "$orbital" >> "$pending"

sbatch --test-only --array=1-64%8 \
  --export=ALL,SIAB_SOURCE_ROOT="$code" \
  "$grid_script" > "$root/GRID_TEST_ONLY.txt"
grid_job=$(sbatch --parsable --array=1-64%8 \
  --export=ALL,SIAB_SOURCE_ROOT="$code" \
  "$grid_script")
grid_job=${grid_job%%;*}
test "$grid_job" -gt 0
echo grid_coulomb_array_job=$grid_job >> "$pending"

sbatch --test-only \
  --export=ALL,SIAB_SOURCE_ROOT="$code" \
  "$headwing_script" > "$root/HEADWING_TEST_ONLY.txt"
headwing_job=$(sbatch --parsable \
  --export=ALL,SIAB_SOURCE_ROOT="$code" \
  "$headwing_script")
headwing_job=${headwing_job%%;*}
test "$headwing_job" -gt 0
echo headwing_job=$headwing_job >> "$pending"

sbatch --test-only \
  --export=ALL,SIAB_SOURCE_ROOT="$code",GRID_COULOMB_ARRAY_JOB_ID="$grid_job",HELDOUT_JOB_ID="$heldout_job" \
  "$sos_script" > "$root/SOS_TEST_ONLY.txt"
selected_sos_job=$(sbatch --parsable --dependency=afterok:"$grid_job" \
  --export=ALL,SIAB_SOURCE_ROOT="$code",GRID_COULOMB_ARRAY_JOB_ID="$grid_job",HELDOUT_JOB_ID="$heldout_job" \
  "$sos_script")
selected_sos_job=${selected_sos_job%%;*}
test "$selected_sos_job" -gt 0
echo selected_sos_job=$selected_sos_job >> "$pending"

sbatch --test-only \
  --export=ALL,SIAB_SOURCE_ROOT="$code",SELECTED_SOS_JOB_ID="$selected_sos_job",GRID_COULOMB_ARRAY_JOB_ID="$grid_job" \
  "$q1_script" > "$root/Q1_DELTA_TEST_ONLY.txt"
q1_response_job=$(sbatch --parsable --dependency=afterok:"$selected_sos_job" \
  --export=ALL,SIAB_SOURCE_ROOT="$code",SELECTED_SOS_JOB_ID="$selected_sos_job",GRID_COULOMB_ARRAY_JOB_ID="$grid_job" \
  "$q1_script")
q1_response_job=${q1_response_job%%;*}
test "$q1_response_job" -gt 0
echo q1_response_job=$q1_response_job >> "$pending"
echo status=submitted >> "$pending"
mv "$pending" "$chain"

echo "C24_SELECTED_VALIDATION_RELEASE_OK grid_job=$grid_job headwing_job=$headwing_job selected_sos_job=$selected_sos_job q1_response_job=$q1_response_job"
