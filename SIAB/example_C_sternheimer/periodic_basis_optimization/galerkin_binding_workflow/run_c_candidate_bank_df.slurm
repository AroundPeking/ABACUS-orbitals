#!/bin/bash
#SBATCH --job-name=c_galerkin_bank
#SBATCH --partition=p1,48cp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=08:00:00

set -euo pipefail
trap 'status=$?; echo "C_GALERKIN_BANK_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${SIAB_SOURCE_ROOT:?missing immutable source deployment}"
: "${SIAB_SOURCE_COMMIT:?missing source commit}"
: "${INITIAL_COEFFICIENTS:?missing accepted starting coefficients}"
: "${OUTPUT_ROOT:?missing immutable output root}"

test "$(cat "$SIAB_SOURCE_ROOT/.git/HEAD")" = "$SIAB_SOURCE_COMMIT"
output=$OUTPUT_ROOT
test ! -e "$output"
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

script=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/build_c_candidate_bank.py
config=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/c_diamond.json
producer=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q1=$producer/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
q2=$producer/c-diamond-g24-fd8-nfreq6-q2-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
atomic_response=/data/home/df_iopcas_ghj/gw/260823/c-atom-siab-reference-fixedocc-p1-8cf890e8/OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat
atomic_source=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/runs/product-pca-20260825/c-atom-source-only-9bd0795d/OUT.C_ATOM_SIAB_SOURCE/STERNHEIMER_SIAB_SOURCE_V1.dat

for path in "$script" "$config" "$INITIAL_COEFFICIENTS" "$atomic_response" "$atomic_source"; do
  test -s "$path"
done
for dataset in "$q1" "$q2"; do
  test -s "$dataset/manifest.dat"
  grep -qx 'status success' "$dataset/status.dat"
  grep -qx 'all_converged yes' "$dataset/status.dat"
done

mkdir -p "$(dirname "$output")"
/usr/bin/time -v -o "${output}.time" \
  "$python_exe" -u "$script" \
    --config "$config" \
    --dataset "$q1" --dataset-family C_solid \
    --dataset "$q2" --dataset-family C_solid \
    --atomic-response "$atomic_response" \
    --atomic-source "$atomic_source" \
    --initial "$INITIAL_COEFFICIENTS" \
    --output-directory "$output" \
    --source-commit "$SIAB_SOURCE_COMMIT" \
    --block-cache-workers 8 \
    > "${output}.stdout" 2> "${output}.stderr"

test -s "$output/CANDIDATE_BANK.json"
grep -q '"status": "success"' "$output/STATUS.json"
test -s "$output/PROVENANCE.json"
sha256sum "$output/CANDIDATE_BANK.json" "$output/STATUS.json" "$output/PROVENANCE.json"
