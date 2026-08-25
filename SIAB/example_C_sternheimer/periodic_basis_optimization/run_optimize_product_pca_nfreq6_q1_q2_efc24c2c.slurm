#!/bin/bash
#SBATCH --job-name=c24_pca_opt_g
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=2,3
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-opt-g-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-opt-g-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C24_PRODUCT_PCA_JOINT_OPTIMIZATION_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
siab_commit=efc24c2c02a460cc08ffdd6bc5c016b5ccf29ba1
producer_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q1=$producer_root/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
q2=$producer_root/c-diamond-g24-fd8-nfreq6-q2-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
optimizer=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/optimize_periodic_basis.py
ng=${SLURM_ARRAY_TASK_ID:?missing g multiplicity}

case "$ng" in
  2)
    initial=$code/seeds/C_3s3p2d1f2g_q1_lowfreq.txt
    initial_sha=835ed71ccc3d590f60fac3257ee613260b80d11c88528fda1d3bf5355e81c059
    nu=3,3,2,1,2
    ;;
  3)
    initial=$code/seeds/C_3s3p2d1f3g_q1_lowfreq.txt
    initial_sha=40ef9cb46c3f1d33b634b4ae0b601a712d1fd9ab14bc619e0593f24d8a0cca58
    nu=3,3,2,1,3
    ;;
  *)
    echo "unsupported g multiplicity: $ng" >&2
    exit 2
    ;;
esac

output=$code/runs/product-pca-20260825/joint-${ng}g-nfreq6-q1-q2

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(git -C "$code" rev-parse HEAD)" = "$siab_commit"
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
grep -qx 'selected_iq 2' "$q2/manifest.dat"
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
  echo purpose=periodic_c_product_pca_multifrequency_multiq_joint_optimization
  echo siab_commit=$siab_commit
  echo producer_job=3118459
  echo producer_abacus_commit=711af860c125b9757c344a1961b63524c550cfe4
  echo datasets=q1,q2
  echo nfreq=6
  echo initial_basis=3s3p2d1f${ng}g
  echo fixed_basis=2s2p1d
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
  python3 "$optimizer" \
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
    --max-steps 4000 \
    --minimum-steps 500 \
    --plateau-patience 600 \
    --plateau-relative-improvement 1e-7 \
    --maximum-backtracks 40 \
    > "$output/optimize.log" 2>&1

grep -q '"status": "success"' "$output/result/STATUS.json"
test -s "$output/result/OPTIMIZATION_RESULT.json"
test -s "$output/result/ORBITAL_RESULTS.txt"
{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' "$output/optimize.time"
  sha256sum "$output/result/OPTIMIZATION_RESULT.json" "$output/result/ORBITAL_RESULTS.txt"
} >> "$output/provenance.txt"

echo "C24_PRODUCT_PCA_JOINT_OPTIMIZATION_OK g=$ng output=$output"
