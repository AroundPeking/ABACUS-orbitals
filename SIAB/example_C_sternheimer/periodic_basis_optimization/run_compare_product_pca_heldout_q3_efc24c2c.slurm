#!/bin/bash
#SBATCH --job-name=c24_pca_holdout
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-holdout-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-holdout-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_PRODUCT_PCA_HELDOUT_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
siab_commit=efc24c2c02a460cc08ffdd6bc5c016b5ccf29ba1
producer_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q3=$producer_root/c-diamond-g24-fd8-nfreq6-q3-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q3_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
original=$code/seeds/C_TZDP_original.txt
seed_two_g=$code/seeds/C_3s3p2d1f2g_q1_lowfreq.txt
seed_three_g=$code/seeds/C_3s3p2d1f3g_q1_lowfreq.txt
joint_two_g=$code/runs/product-pca-20260825/joint-2g-nfreq6-q1-q2/result/ORBITAL_RESULTS.txt
joint_three_g=$code/runs/product-pca-20260825/joint-3g-nfreq6-q1-q2/result/ORBITAL_RESULTS.txt
comparator=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/compare_periodic_candidates.py
output=$code/runs/product-pca-20260825/heldout-q3-comparison

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(cat "$code/.git/HEAD")" = "$siab_commit"
test ! -e "$output"
test -s "$q3/manifest.dat"
grep -qx 'status success' "$q3/status.dat"
grep -qx 'all_converged yes' "$q3/status.dat"
grep -qx 'auxiliary_basis_source product_pca' "$q3/manifest.dat"
grep -qx 'frequency_count 6' "$q3/manifest.dat"
grep -qx 'selected_iq 43' "$q3/manifest.dat"
grep -qx 'kernel full_coulomb' "$q3/manifest.dat"
for path in "$original" "$seed_two_g" "$seed_three_g" \
  "$joint_two_g" "$joint_three_g" "$comparator"; do
  test -s "$path"
done
test "$(sha256sum "$original" | awk '{print $1}')" = \
  b58a2183c3028e46e6f4bc55b0f21531f1253275d5c2f2c4ee4e27676c1b55f4
test "$(sha256sum "$seed_two_g" | awk '{print $1}')" = \
  835ed71ccc3d590f60fac3257ee613260b80d11c88528fda1d3bf5355e81c059
test "$(sha256sum "$seed_three_g" | awk '{print $1}')" = \
  40ef9cb46c3f1d33b634b4ae0b601a712d1fd9ab14bc619e0593f24d8a0cca58

export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages:$code
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p "$output"
{
  echo status=running
  echo purpose=periodic_c_product_pca_independent_q3_validation
  echo siab_commit=$siab_commit
  echo producer_job=${Q3_PRODUCER_JOB_ID:?missing q3 producer job id}
  echo optimizer_job=${OPTIMIZER_JOB_ID:-unknown}
  echo dataset=q3
  echo full_q_index=43
  echo nfreq=6
  echo auxiliary_basis_source=product_pca
  echo candidates=original_tzdp,seed_two_g,joint_two_g,seed_three_g,joint_three_g
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$q3/manifest.dat" "$q3/status.dat" \
    "$original" "$seed_two_g" "$joint_two_g" "$seed_three_g" "$joint_three_g" \
    "$comparator" "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_data.py" \
    "$runtime/site-packages/torch/lib/libtorch_cpu.so"
} > "$output/provenance.txt"

/usr/bin/time -v -o "$output/compare.time" \
  python3 "$comparator" \
    --dataset "$q3" \
    --candidate "original-tzdp:$original:3,3,2,0,0" \
    --candidate "seed-two-g:$seed_two_g:3,3,2,1,2" \
    --candidate "joint-two-g:$joint_two_g:3,3,2,1,2" \
    --candidate "seed-three-g:$seed_three_g:3,3,2,1,3" \
    --candidate "joint-three-g:$joint_three_g:3,3,2,1,3" \
    --output "$output/COMPARISON_RESULT.json" \
    > "$output/compare.log" 2>&1

test -s "$output/COMPARISON_RESULT.json"
python3 - "$output/COMPARISON_RESULT.json" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1], encoding="ascii"))
assert report["format_version"] == 1
assert len(report["datasets"]) == 1
assert len(report["candidates"]) == 5
for candidate in report["candidates"]:
    for key in (
        "global_weighted_relative_pi_error",
        "global_weighted_relative_trace_log_error",
        "minimum_occupied_capture",
    ):
        assert math.isfinite(candidate[key])
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' "$output/compare.time"
  sha256sum "$output/COMPARISON_RESULT.json"
} >> "$output/provenance.txt"

echo "C24_PRODUCT_PCA_HELDOUT_OK output=$output"
