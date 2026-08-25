#!/bin/bash
#SBATCH --job-name=c24_holdout_2g3g
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-b9a4ec3c/logs/c24-holdout-2g3g-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-b9a4ec3c/logs/c24-holdout-2g3g-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_HELDOUT_TWO_G_THREE_G_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-b9a4ec3c
base=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825
q3=$base/c-diamond-g24-fd8-nfreq6-q3-9ee4e4e0b-kpar8/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q3/STERNHEIMER_BASIS_OPT_V1
original=/data/home/df_iopcas_ghj/app/siab/periodic-c-d5eee384/SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt
q1_two_g=$base/screen-third-g-q1-lowfreq-ed35f1ad/result/ORBITAL_RESULTS.txt
joint_two_g=$base/optimize-two-g-nfreq6-q1-q2-b9a4ec3c/result/ORBITAL_RESULTS.txt
joint_three_g=$base/optimize-three-g-nfreq6-q1-q2-14f5f64d/result/ORBITAL_RESULTS.txt
comparator=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/compare_periodic_candidates.py
output=$base/compare-heldout-q3-joint-two-g-three-g-b9a4ec3c

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$output"
test -s "$q3/manifest.dat"
grep -qx 'status success' "$q3/status.dat"
grep -qx 'all_converged yes' "$q3/status.dat"
for path in "$original" "$q1_two_g" "$joint_two_g" "$joint_three_g" "$comparator"; do
  test -s "$path"
done
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
  echo purpose=periodic_c_independent_q3_minimal_two_g_three_g_validation
  echo siab_code_commit=b9a4ec3c79536f66d82ce85d8642d46423e58ed7
  echo producer_job=3116059
  echo two_g_optimizer_job=${TWO_G_OPTIMIZER_JOB_ID:-unknown}
  echo three_g_optimizer_job=3116015
  echo dataset=q3
  echo nfreq=6
  echo candidates=original_tzdp,q1_two_g,joint_two_g,joint_three_g
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$q3/manifest.dat" "$q3/status.dat" \
    "$original" "$q1_two_g" "$joint_two_g" "$joint_three_g" "$comparator" \
    "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_data.py" \
    "$code/SIAB/example_C_sternheimer/periodic_basis_optimization/optimize_periodic_basis.py" \
    "$runtime/site-packages/torch/lib/libtorch_cpu.so"
} > "$output/provenance.txt"

/usr/bin/time -v -o "$output/compare.time" \
  python3 "$comparator" \
    --dataset "$q3" \
    --candidate "original-tzdp:$original:3,3,2,0,0" \
    --candidate "q1-two-g:$q1_two_g:3,3,2,1,2" \
    --candidate "joint-two-g:$joint_two_g:3,3,2,1,2" \
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
assert len(report["candidates"]) == 4
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

echo "C24_HELDOUT_TWO_G_THREE_G_OK output=$output"
