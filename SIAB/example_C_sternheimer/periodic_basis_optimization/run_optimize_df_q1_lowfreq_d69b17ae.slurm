#!/bin/bash
#SBATCH --job-name=c24_q1_siab_opt
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-d69b17ae/logs/c24-q1-opt-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-d69b17ae/logs/c24-q1-opt-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_Q1_SIAB_OPT_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-d69b17ae
dataset=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/c-diamond-g24-fd8-q1-lowfreq-9ee4e4e0b-kpar8/OUT.C_DIAMOND_G24_FD8_Q1_LOWFREQ/STERNHEIMER_BASIS_OPT_V1
initial=$code/SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt
optimizer=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/optimize_periodic_basis.py
siab_commit=d69b17aea815235793a9c87f0bf83705140cc194
output=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/optimize-c-tzdp-q1-lowfreq-d69b17ae

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test -s "$dataset/manifest.dat"
test -s "$dataset/status.dat"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
test -s "$initial"
test -s "$optimizer"
test ! -e "$output"
mkdir -p "$output"

export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

{
  echo status=running
  echo purpose=periodic_c_fixed_dzp_tzdp_q1_lowfreq_screen
  echo siab_commit=$siab_commit
  echo dataset=$dataset
  echo nu=3,3,2,0,0
  echo fixed_nu=2,2,1,0,0
  echo objective=exact_delta_st_pi
  echo promotion_scope=screening_only
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$dataset/manifest.dat" "$dataset/status.dat" "$initial" \
    "$optimizer" \
    "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_sternheimer.py" \
    "$runtime/site-packages/torch/lib/libtorch_cpu.so"
} > "$output/provenance.txt"

/usr/bin/time -v -o "$output/optimize.time" \
  python3 "$optimizer" \
    --dataset "$dataset" \
    --initial "$initial" \
    --output-directory "$output/result" \
    --siab-commit "$siab_commit" \
    --element C \
    --nu 3,3,2,0,0 \
    --fixed-nu 2,2,1,0,0 \
    --radial-rows 31 \
    --max-l 4 \
    --learning-rate 1e-3 \
    --max-steps 2000 \
    --minimum-steps 200 \
    --plateau-patience 300 \
    --plateau-relative-improvement 1e-6 \
    > "$output/optimize.log" 2>&1

test "$(python3 - "$output/result/STATUS.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="ascii"))["status"])
PY
)" = success

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p' \
    "$output/optimize.time"
  sha256sum "$output/result/OPTIMIZATION_RESULT.json" \
    "$output/result/OPTIMIZATION_HISTORY.jsonl" \
    "$output/result/ORBITAL_RESULTS.txt"
} >> "$output/provenance.txt"
