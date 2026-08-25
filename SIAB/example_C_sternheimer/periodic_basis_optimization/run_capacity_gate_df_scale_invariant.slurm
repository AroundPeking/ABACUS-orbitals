#!/bin/bash
#SBATCH --job-name=c24_scale_capacity
#SBATCH --partition=p1
#SBATCH --array=4,6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-scale-invariant-abf7326d/logs/c24-scale-capacity-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-scale-invariant-abf7326d/logs/c24-scale-capacity-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C24_SCALE_CAPACITY_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-scale-invariant-abf7326d
coefficients=$code/SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt
evaluator=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/evaluate_periodic_basis.py
base_commit=40b10ef8cd4fb8c7b5645c30f940afc3a20dea6f
source_diff_sha256=abf7326d075c02971dd7c1ad6ba8442c54a0f9b1a04548d99d387ad98394123f

case "$SLURM_ARRAY_TASK_ID" in
  4)
    dataset=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/c-diamond-g24-fd8-q1-lowfreq-9ee4e4e0b-kpar8/OUT.C_DIAMOND_G24_FD8_Q1_LOWFREQ/STERNHEIMER_BASIS_OPT_V1
    nu=3,3,2,0,0
    ;;
  6)
    dataset=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/c-diamond-g24-fd8-q1-lowfreq-lmax6-973db989f-kpar8/OUT.C_DIAMOND_G24_FD8_Q1_LOWFREQ_LMAX6/STERNHEIMER_BASIS_OPT_V1
    nu=3,3,2,0,0,0,0
    ;;
  *)
    echo "unsupported lmax: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

lmax=$SLURM_ARRAY_TASK_ID
output=/ssd/work/df_iopcas_ghj/c-periodic-basis-opt-20260825/capacity-g24-q1-lowfreq-lmax${lmax}-scale-invariant-abf7326d

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(cat "$code/SIAB_COMMIT")" = "$base_commit"
test -s "$dataset/manifest.dat"
test -s "$dataset/status.dat"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
test -s "$coefficients"
test -s "$evaluator"
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

python3 - <<'PY'
import numpy
import torch

assert numpy.__version__ == "1.23.3"
assert torch.__version__ == "1.9.1"
value = torch.tensor([[2.0, 0.1], [0.1, 1.0]], dtype=torch.float64)
assert torch.isfinite(torch.linalg.eigvalsh(value)).all()
PY

{
  echo status=running
  echo purpose=periodic_c_scale_invariant_mother_capacity_gate
  echo base_commit=$base_commit
  echo source_diff_sha256=$source_diff_sha256
  echo dataset=$dataset
  echo candidate_nu=$nu
  echo mother_response_tolerance=1e-3
  echo occupied_capture_tolerance=1e-6
  echo relative_rank_tolerance=1e-12
  echo condition_limit=1e12
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$dataset/manifest.dat" "$dataset/status.dat" "$coefficients" \
    "$code/SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_sternheimer.py" \
    "$runtime/site-packages/torch/lib/libtorch_cpu.so"
} > "$output/provenance.txt"

/usr/bin/time -v -o "$output/evaluate.time" \
  python3 "$evaluator" \
    --dataset "$dataset" \
    --coefficients "$coefficients" \
    --output "$output/capacity.json" \
    --nu "$nu" \
    --max-l "$lmax" \
    --radial-rows 31 \
    --mother-response-tolerance 1e-3 \
    --relative-rank-tolerance 1e-12 \
    --condition-limit 1e12 \
    --occupied-capture-tolerance 1e-6 \
    > "$output/evaluate.log" 2>&1

python3 - "$output/capacity.json" "$output/physics_gate.txt" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
mother = report["mother"]["capacity_gate"]
candidate = report["candidate"]["evaluation_gate"]
allowed = "yes" if report["optimization_allowed"] else "no"
Path(sys.argv[2]).write_text(
    f"mother_capacity_gate {mother}\n"
    f"candidate_evaluation_gate {candidate}\n"
    f"optimization_allowed {allowed}\n",
    encoding="ascii",
)
print(
    "C24_SCALE_CAPACITY_COMPLETE "
    f"mother={mother} candidate={candidate} optimization_allowed={allowed}"
)
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p' \
    "$output/evaluate.time"
  sha256sum "$output/capacity.json" "$output/physics_gate.txt"
} >> "$output/provenance.txt"
