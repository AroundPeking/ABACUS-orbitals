#!/bin/bash
#SBATCH --job-name=c24_opt_nf6q
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-14f5f64d/logs/c24-opt-nf6q-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-14f5f64d/logs/c24-opt-nf6q-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_OPTIMIZE_NFREQ6_Q_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-14f5f64d
base=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825
q1=$base/c-diamond-g24-fd8-nfreq6-q1-9ee4e4e0b-kpar8/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1/STERNHEIMER_BASIS_OPT_V1
q2=$base/c-diamond-g24-fd8-nfreq6-q2-9ee4e4e0b-kpar8/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2/STERNHEIMER_BASIS_OPT_V1
initial=$base/screen-fourth-g-q1-lowfreq-ed35f1ad/result/ORBITAL_RESULTS.txt
optimizer=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/optimize_periodic_basis.py
output=$base/optimize-three-g-nfreq6-q1-q2-14f5f64d

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$output"
for dataset in "$q1" "$q2"; do
  test -s "$dataset/manifest.dat"
  grep -qx 'status success' "$dataset/status.dat"
  grep -qx 'all_converged yes' "$dataset/status.dat"
done
for path in "$initial" "$optimizer"; do
  test -s "$path"
done

export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages
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
  echo purpose=periodic_c_multifrequency_multiq_joint_optimization
  echo siab_commit=14f5f64d0044d99ba50fe74d08d4be013270a0f9
  echo producer_job=3113575
  echo datasets=q1,q2
  echo nfreq=6
  echo initial_basis=3s3p2d1f3g
  echo fixed_basis=2s2p1d
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$q1/manifest.dat" "$q1/status.dat" \
    "$q2/manifest.dat" "$q2/status.dat" \
    "$initial" "$optimizer" \
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
    --siab-commit 14f5f64d0044d99ba50fe74d08d4be013270a0f9 \
    --element C \
    --nu 3,3,2,1,3 \
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

echo "C24_OPTIMIZE_NFREQ6_Q_OK output=$output"
