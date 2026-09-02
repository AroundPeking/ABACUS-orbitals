#!/bin/bash
#SBATCH --job-name=c_solid_q123
#SBATCH --partition=p1,48cp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=08:00:00

set -euo pipefail
trap 'status=$?; echo "C_SOLID_Q123_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${SIAB_SOURCE_ROOT:?missing immutable source deployment}"
: "${SIAB_SOURCE_COMMIT:?missing source commit}"
: "${INITIAL_COEFFICIENTS:?missing accepted starting coefficients}"
: "${INITIAL_SHA256:?missing initial coefficient SHA256}"
: "${Q1_PHYSICS_HASH:?missing q1 physics hash}"
: "${Q2_PHYSICS_HASH:?missing q2 physics hash}"
: "${Q3_PHYSICS_HASH:?missing q3 physics hash}"
: "${OUTPUT_ROOT:?missing immutable output root}"

test "$(cat "$SIAB_SOURCE_ROOT/.git/HEAD")" = "$SIAB_SOURCE_COMMIT"
test "$(sha256sum "$INITIAL_COEFFICIENTS" | awk '{print $1}')" = "$INITIAL_SHA256"
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

workflow=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
script=$workflow/build_c_solid_all_q_candidate.py
config=$workflow/c_diamond_solid_q123_reduced.json
producer=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q1=$producer/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
q2=$producer/c-diamond-g24-fd8-nfreq6-q2-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
q3=$producer/c-diamond-g24-fd8-nfreq6-q3-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q3_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1

for path in "$script" "$config" "$INITIAL_COEFFICIENTS"; do
  test -s "$path"
done
for specification in "$q1:$Q1_PHYSICS_HASH" "$q2:$Q2_PHYSICS_HASH" "$q3:$Q3_PHYSICS_HASH"; do
  dataset=${specification%%:*}
  expected=${specification##*:}
  test -s "$dataset/manifest.dat"
  grep -qx 'status success' "$dataset/status.dat"
  grep -qx 'all_converged yes' "$dataset/status.dat"
  grep -qx "physics_hash $expected" "$dataset/manifest.dat"
done

mkdir -p "$(dirname "$output")"
/usr/bin/time -v -o "${output}.time" \
  "$python_exe" -u "$script" \
    --config "$config" \
    --qstar "1=$q1" \
    --qstar "2=$q2" \
    --qstar "3=$q3" \
    --initial "$INITIAL_COEFFICIENTS" \
    --output-directory "$output" \
    --source-commit "$SIAB_SOURCE_COMMIT" \
    --block-cache-workers 8 \
    > "${output}.stdout" 2> "${output}.stderr"

test -s "$output/CANDIDATE.json"
grep -q '"status": "success"' "$output/STATUS.json"
grep -q '"coverage": "reduced"' "$output/STATUS.json"
grep -q '"physical_release_gate": "hold"' "$output/STATUS.json"
test -s "$output/PROVENANCE.json"
sha256sum "$output/CANDIDATE.json" "$output/STATUS.json" "$output/PROVENANCE.json"
