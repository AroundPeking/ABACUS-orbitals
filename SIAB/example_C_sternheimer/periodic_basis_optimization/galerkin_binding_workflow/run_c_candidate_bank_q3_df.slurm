#!/bin/bash
#SBATCH --job-name=c_galerkin_q3
#SBATCH --partition=p1,48cp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=04:00:00

set -euo pipefail
trap 'status=$?; echo "C_GALERKIN_Q3_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${SIAB_SOURCE_ROOT:?missing immutable source deployment}"
: "${SIAB_SOURCE_COMMIT:?missing source commit}"
: "${BANK_ROOT:?missing successful candidate bank root}"
: "${INITIAL_COEFFICIENTS:?missing accepted starting coefficients}"
: "${OUTPUT_ROOT:?missing immutable q3 output root}"

test "$(cat "$SIAB_SOURCE_ROOT/.git/HEAD")" = "$SIAB_SOURCE_COMMIT"
output=$OUTPUT_ROOT
work=${output}.work
test ! -e "$output"
test ! -e "$work"
module load python/3.9.22
runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
python_exe=$(command -v python3)
export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages:$SIAB_SOURCE_ROOT/SIAB/opt_orb_pytorch_dpsi
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

workflow=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
config=$workflow/c_diamond.json
selector=$workflow/select_c_candidate_q3.py
comparator=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/compare_periodic_candidates.py
producer=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q3=$producer/c-diamond-g24-fd8-nfreq6-q3-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q3_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
bank=$BANK_ROOT/CANDIDATE_BANK.json

for path in "$config" "$selector" "$comparator" "$bank" "$INITIAL_COEFFICIENTS"; do
  test -s "$path"
done
grep -q '"status": "success"' "$BANK_ROOT/STATUS.json"
grep -q '"candidate_bank_gate": "pass"' "$BANK_ROOT/STATUS.json"
test -s "$q3/manifest.dat"
test -s "$q3/status.dat"
grep -qx 'status success' "$q3/status.dat"
grep -qx 'all_converged yes' "$q3/status.dat"
grep -qx 'selected_iq 43' "$q3/manifest.dat"
grep -qx 'kernel full_coulomb' "$q3/manifest.dat"

mkdir -p "$work"
comparison=$work/COMPARISON_RESULT.json
/usr/bin/time -v -o "$work/compare.time" \
  "$python_exe" -u "$comparator" \
    --dataset "$q3" \
    --candidate initial:"$INITIAL_COEFFICIENTS":3,3,2,0,0 \
    --candidate pareto_w0p25:"$BANK_ROOT/candidates/pareto_w0p25/ORBITAL_RESULTS.txt":3,3,2,0,0 \
    --candidate pareto_w0p50:"$BANK_ROOT/candidates/pareto_w0p50/ORBITAL_RESULTS.txt":3,3,2,0,0 \
    --candidate pareto_w0p75:"$BANK_ROOT/candidates/pareto_w0p75/ORBITAL_RESULTS.txt":3,3,2,0,0 \
    --occupied-capture-floor 0.9998982409775239 \
    --output "$comparison" \
    > "$work/compare.stdout" 2> "$work/compare.stderr"

"$python_exe" -u "$selector" \
  --config "$config" \
  --bank "$bank" \
  --comparison "$comparison" \
  --output-directory "$output" \
  --source-commit "$SIAB_SOURCE_COMMIT" \
  > "$work/select.stdout" 2> "$work/select.stderr"

test -s "$output/COMPARISON_RESULT.json"
test -s "$output/SELECTION_RESULT.json"
grep -q '"status": "success"' "$output/STATUS.json"
test -s "$output/PROVENANCE.json"
sha256sum "$output/COMPARISON_RESULT.json" "$output/SELECTION_RESULT.json" "$output/STATUS.json" "$output/PROVENANCE.json"
