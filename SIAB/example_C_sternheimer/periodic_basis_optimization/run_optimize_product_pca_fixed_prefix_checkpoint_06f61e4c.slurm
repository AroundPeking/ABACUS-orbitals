#!/bin/bash
# Checkpointed 500-step segment pinned to SIAB commit 06f61e4c.
#SBATCH --job-name=c24_pca_fp_ckp
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=2,3
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-fp-opt-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-fp-opt-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C24_PRODUCT_PCA_FIXED_PREFIX_OPTIMIZATION_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-06f61e4c
campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
siab_commit=06f61e4c5d8aa514a391b268206d99a2e70c6075
producer_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q1=$producer_root/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
q2=$producer_root/c-diamond-g24-fd8-nfreq6-q2-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
optimizer=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/optimize_periodic_basis.py
ng=${SLURM_ARRAY_TASK_ID:?missing g multiplicity}

case "$ng" in
  2)
    initial=$campaign/seeds/C_3s3p2d1f2g_q1_lowfreq.txt
    initial_sha=835ed71ccc3d590f60fac3257ee613260b80d11c88528fda1d3bf5355e81c059
    nu=3,3,2,1,2
    ;;
  3)
    initial=$campaign/seeds/C_3s3p2d1f3g_q1_lowfreq.txt
    initial_sha=40ef9cb46c3f1d33b634b4ae0b601a712d1fd9ab14bc619e0593f24d8a0cca58
    nu=3,3,2,1,3
    ;;
  *)
    echo "unsupported g multiplicity: $ng" >&2
    exit 2
    ;;
esac

output=$campaign/runs/product-pca-20260825/joint-fixed-prefix-layout-checkpoint-500-${ng}g-nfreq6-q1-q2

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(cat "$code/.git/HEAD")" = "$siab_commit"
test "$(cat "$code/SIAB_COMMIT")" = "$siab_commit"
test "$(sha256sum "$initial" | awk '{print $1}')" = "$initial_sha"
test -s "$optimizer"
test ! -e "$output"

for dataset in "$q1" "$q2"; do
  test -s "$dataset/manifest.dat"
  grep -qx 'status success' "$dataset/status.dat"
  grep -qx 'all_converged yes' "$dataset/status.dat"
  grep -qx 'auxiliary_basis_source product_pca' "$dataset/manifest.dat"
  grep -qx 'frequency_count 6' "$dataset/manifest.dat"
  grep -qx 'kernel full_coulomb' "$dataset/manifest.dat"
done
grep -qx 'selected_iq 1' "$q1/manifest.dat"
grep -qx 'selected_iq 22' "$q2/manifest.dat"
test "$(awk '$1 == "auxiliary_basis_sha256" {print $2}' "$q1/manifest.dat")" = \
  "$(awk '$1 == "auxiliary_basis_sha256" {print $2}' "$q2/manifest.dat")"

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
  echo purpose=periodic_c_product_pca_fixed_prefix_multifrequency_multiq_checkpointed_optimization
  echo siab_commit=$siab_commit
  echo producer_jobs=3118459_1+3119622,3119900_2
  echo datasets=q1,q2
  echo full_q_indices=1,22
  echo nfreq=6
  echo initial_basis=3s3p2d1f${ng}g
  echo fixed_basis=2s2p1d
  echo occupied_capture_reference=fixed_prefix
  echo omitted_reference_projection_validation=layout
  echo maximum_steps=500
  echo best_checkpoint=each_strict_loss_improvement
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$q1/manifest.dat" "$q1/status.dat" \
    "$q2/manifest.dat" "$q2/status.dat" "$initial" "$optimizer" \
    "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_basis.py" \
    "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_data.py" \
    "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_fit.py" \
    "$runtime/site-packages/torch/lib/libtorch_cpu.so"
} > "$output/provenance.txt"

/usr/bin/time -v -o "$output/optimize.time" \
  python3 -u "$optimizer" \
    --dataset "$q1" \
    --dataset "$q2" \
    --initial "$initial" \
    --output-directory "$output/result" \
    --siab-commit "$siab_commit" \
    --element C \
    --nu "$nu" \
    --fixed-nu 2,2,1,0,0 \
    --radial-rows 31 \
    --max-l 4 \
    --learning-rate 1e-3 \
    --max-steps 500 \
    --minimum-steps 200 \
    --plateau-patience 250 \
    --plateau-relative-improvement 1e-7 \
    --maximum-backtracks 40 \
    --block-cache-workers 8 \
    --occupied-capture-reference fixed_prefix \
    --omitted-reference-projection-validation layout \
    > "$output/optimize.log" 2>&1

grep -q '"status": "success"' "$output/result/STATUS.json"
test -s "$output/result/OPTIMIZATION_RESULT.json"
test -s "$output/result/ORBITAL_RESULTS.txt"
test -s "$output/result/BEST_CHECKPOINT.json"
test -s "$output/result/BEST_ORBITAL_CHECKPOINT.txt"
python3 - "$output/result/OPTIMIZATION_RESULT.json" "$output/result/BEST_CHECKPOINT.json" \
  "$output/result/BEST_ORBITAL_CHECKPOINT.txt" "$output/result/ORBITAL_RESULTS.txt" <<'PY'
import hashlib
import json
import math
import sys

result = json.load(open(sys.argv[1], encoding="ascii"))
checkpoint = json.load(open(sys.argv[2], encoding="ascii"))
digest = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
assert result["occupied_capture_reference"] == "fixed_prefix"
assert result["omitted_reference_projection_validation"] == "layout"
assert result["block_cache_workers"] == 8
assert result["steps_completed"] >= 200
assert result["steps_completed"] <= 500
assert math.isfinite(result["best_loss"])
assert result["best_loss"] < result["initial_loss"]
assert checkpoint["step"] == result["best_step"]
assert checkpoint["loss"] == result["best_loss"]
assert checkpoint["orbital_sha256"] == digest(sys.argv[3])
assert digest(sys.argv[3]) == digest(sys.argv[4])
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' "$output/optimize.time"
  sha256sum "$output/result/OPTIMIZATION_RESULT.json" \
    "$output/result/ORBITAL_RESULTS.txt" \
    "$output/result/BEST_CHECKPOINT.json" \
    "$output/result/BEST_ORBITAL_CHECKPOINT.txt"
} >> "$output/provenance.txt"

echo "C24_PRODUCT_PCA_FIXED_PREFIX_OPTIMIZATION_OK g=$ng output=$output"
