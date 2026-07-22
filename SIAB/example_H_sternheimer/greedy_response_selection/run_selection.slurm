#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=siab_select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00

set -euo pipefail

source_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_selection_source_v1_20260722
asset_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_selection_assets_v1_20260722
target_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_global_whitening_v2_20260722
output_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_selection_campaign_v1_20260722
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

greedy=$source_root/SIAB/example_H_sternheimer/greedy_response_selection
optimizer=$source_root/SIAB/opt_orb_pytorch_dpsi/main.py
driver=$greedy/run_response_selection.py
config=$greedy/selection_config.json
template=$greedy/optimizer_template.json
baseline=$asset_root/H_TZDP_8au_ORBITAL_RESULTS.txt

atom_dir=$target_root/producer_atom
h3_dir=$target_root/producer_h3
ghost_dir=$target_root/producer_fragment_ghost
atom_validation=$atom_dir/target_validation.json
h3_validation=$h3_dir/target_validation.json
ghost_validation=$ghost_dir/target_validation.json
atom_target=$atom_dir/OUT.H_SIAB_GREEDY_ATOM_WFULL_NF16_E50/sternheimer_matrix.dat
h3_target=$h3_dir/OUT.H3_SIAB_GREEDY_EQ090_WFULL_NF16_E50/sternheimer_matrix.dat
ghost_target=$ghost_dir/OUT.H_GHOST2_SIAB_GREEDY_EQ090_WFULL_NF16_E50/sternheimer_matrix.dat

test ! -e "$output_root"
test -s "$driver"
test -s "$optimizer"
test -s "$config"
test -s "$template"
test -s "$baseline"
for path in \
  "$atom_validation" "$h3_validation" "$ghost_validation" \
  "$atom_target" "$h3_target" "$ghost_target"; do
  test -s "$path"
done

cd "$source_root"
sha256sum -c SOURCE_SHA256SUMS
cd "$asset_root"
sha256sum -c ASSET_SHA256SUMS
"$python" - "$atom_validation" "$h3_validation" "$ghost_validation" <<'PY'
import json
import pathlib
import sys
for value in sys.argv[1:]:
    payload = json.loads(pathlib.Path(value).read_text())
    if payload.get("abfs_source") != "explicit_abfs":
        raise SystemExit("target validation did not preserve explicit ABFS")
    if payload.get("frequencies") != 16:
        raise SystemExit("target validation has the wrong frequency count")
PY

export PYTHONPATH=$source_root/SIAB/opt_orb_pytorch_dpsi:$greedy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_DYNAMIC=FALSE

echo "date_start=$(date +%F_%T)"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "omp_threads=$OMP_NUM_THREADS"
sha256sum "$driver" "$optimizer" "$config" "$template" "$baseline" \
  "$atom_target" "$h3_target" "$ghost_target"

/usr/bin/time -v "$python" "$driver" \
  --config "$config" \
  --optimizer-template "$template" \
  --asset-root "$asset_root" \
  --baseline "$baseline" \
  --atom-target "$atom_target" \
  --multicenter-target "$h3_target" \
  --ghost-target "$ghost_target" \
  --optimizer "$optimizer" \
  --python "$python" \
  --output "$output_root"

test -s "$output_root/campaign_manifest.json"
test -s "$output_root/frozen/selection_manifest.json"
sha256sum "$output_root/campaign_manifest.json" \
  "$output_root/frozen/selection_manifest.json"
echo "date_end=$(date +%F_%T)"
