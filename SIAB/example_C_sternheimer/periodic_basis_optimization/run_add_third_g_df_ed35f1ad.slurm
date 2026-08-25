#!/bin/bash
#SBATCH --job-name=c24_add_third_g
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-ed35f1ad/logs/c24-add-third-g-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-ed35f1ad/logs/c24-add-third-g-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_ADD_THIRD_G_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-ed35f1ad
dataset=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/c-diamond-g24-fd8-q1-lowfreq-9ee4e4e0b-kpar8/OUT.C_DIAMOND_G24_FD8_Q1_LOWFREQ/STERNHEIMER_BASIS_OPT_V1
third_stage=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/screen-third-g-q1-lowfreq-ed35f1ad/result/ORBITAL_RESULTS.txt
expander=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/expand_periodic_coefficients.py
optimizer=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/optimize_periodic_basis.py
siab_commit=ed35f1ad548d03416ad3811d6eb8c08b9eb84e41
output=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/screen-fourth-g-q1-lowfreq-ed35f1ad
expanded=$output/INITIAL_EXPANDED_ORBITAL_RESULTS.txt

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test -s "$dataset/manifest.dat"
test -s "$dataset/status.dat"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
test -s "$third_stage"
test -s "$expander"
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

python3 "$expander" \
  --input "$third_stage" \
  --output "$expanded" \
  --element C \
  --radial-rows 31 \
  --input-nu 3,3,2,1,2 \
  --target-nu 3,3,2,1,3

{
  echo status=running
  echo purpose=periodic_c_third_g_response_channel_screen
  echo siab_commit=$siab_commit
  echo dataset=$dataset
  echo third_stage=$third_stage
  echo added_channel=g
  echo nu=3,3,2,1,3
  echo fixed_nu=2,2,1,0,0
  echo seed=deterministic_bessel_complement
  echo objective=exact_delta_st_pi
  echo promotion_scope=screening_only
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$dataset/manifest.dat" "$dataset/status.dat" "$third_stage" \
    "$expanded" "$expander" "$optimizer" \
    "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_fit.py" \
    "$runtime/site-packages/torch/lib/libtorch_cpu.so"
} > "$output/provenance.txt"

/usr/bin/time -v -o "$output/optimize.time" \
  python3 "$optimizer" \
    --dataset "$dataset" \
    --initial "$expanded" \
    --output-directory "$output/result" \
    --siab-commit "$siab_commit" \
    --element C \
    --nu 3,3,2,1,3 \
    --fixed-nu 2,2,1,0,0 \
    --radial-rows 31 \
    --max-l 4 \
    --learning-rate 1e-3 \
    --max-steps 1000 \
    --minimum-steps 100 \
    --plateau-patience 200 \
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
