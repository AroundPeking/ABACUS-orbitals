#!/bin/bash
#SBATCH --job-name=c24_compare_nf6q
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-14f5f64d/logs/c24-compare-nf6q-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-14f5f64d/logs/c24-compare-nf6q-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_COMPARE_NFREQ6_Q_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-14f5f64d
base=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825
q1=$base/c-diamond-g24-fd8-nfreq6-q1-9ee4e4e0b-kpar8/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q1/STERNHEIMER_BASIS_OPT_V1
q2=$base/c-diamond-g24-fd8-nfreq6-q2-9ee4e4e0b-kpar8/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q2/STERNHEIMER_BASIS_OPT_V1
original=/data/home/df_iopcas_ghj/app/siab/periodic-c-d5eee384/SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt
optimized_tzdp=$base/optimize-c-tzdp-q1-lowfreq-d5eee384/result/ORBITAL_RESULTS.txt
f_only=$base/screen-add-f-q1-lowfreq-ed35f1ad/result/ORBITAL_RESULTS.txt
one_g=$base/screen-second-g-q1-lowfreq-ed35f1ad/result/ORBITAL_RESULTS.txt
two_g=$base/screen-third-g-q1-lowfreq-ed35f1ad/result/ORBITAL_RESULTS.txt
three_g=$base/screen-fourth-g-q1-lowfreq-ed35f1ad/result/ORBITAL_RESULTS.txt
comparator=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/compare_periodic_candidates.py
output=$base/compare-nfreq6-q1-q2-14f5f64d

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$output"
for dataset in "$q1" "$q2"; do
  test -s "$dataset/manifest.dat"
  grep -qx 'status success' "$dataset/status.dat"
  grep -qx 'all_converged yes' "$dataset/status.dat"
done
for path in "$original" "$optimized_tzdp" "$f_only" "$one_g" "$two_g" "$three_g" "$comparator"; do
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
  echo purpose=periodic_c_multifrequency_multiq_candidate_comparison
  echo siab_commit=14f5f64d0044d99ba50fe74d08d4be013270a0f9
  echo producer_job=3113575
  echo datasets=q1,q2
  echo nfreq=6
  echo candidates=original_tzdp,optimized_tzdp,f_only,one_g,two_g,three_g
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$q1/manifest.dat" "$q1/status.dat" \
    "$q2/manifest.dat" "$q2/status.dat" \
    "$original" "$optimized_tzdp" "$f_only" "$one_g" "$two_g" "$three_g" \
    "$comparator" "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_data.py" \
    "$code/SIAB/example_C_sternheimer/periodic_basis_optimization/optimize_periodic_basis.py" \
    "$runtime/site-packages/torch/lib/libtorch_cpu.so"
} > "$output/provenance.txt"

/usr/bin/time -v -o "$output/compare.time" \
  python3 "$comparator" \
    --dataset "$q1" \
    --dataset "$q2" \
    --candidate "original-tzdp:$original:3,3,2,0,0" \
    --candidate "optimized-tzdp:$optimized_tzdp:3,3,2,0,0" \
    --candidate "f-only:$f_only:3,3,2,1,0" \
    --candidate "one-g:$one_g:3,3,2,1,1" \
    --candidate "two-g:$two_g:3,3,2,1,2" \
    --candidate "three-g:$three_g:3,3,2,1,3" \
    --output "$output/COMPARISON_RESULT.json" \
    > "$output/compare.log" 2>&1

test -s "$output/COMPARISON_RESULT.json"
python3 - "$output/COMPARISON_RESULT.json" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1], encoding="ascii"))
assert report["format_version"] == 1
assert len(report["datasets"]) == 2
assert len(report["candidates"]) == 6
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

echo "C24_COMPARE_NFREQ6_Q_OK output=$output"
