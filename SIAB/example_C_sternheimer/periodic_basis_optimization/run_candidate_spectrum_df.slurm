#!/bin/bash
#SBATCH --job-name=c_basis_spectrum
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000M
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

dataset=${DATASET:?missing DATASET}
coefficients=${COEFFICIENTS:?missing COEFFICIENTS}
nu=${NU:?missing NU}
label=${LABEL:?missing LABEL}
output=${OUTPUT:?missing OUTPUT}
siab_deployment=${SIAB_DEPLOYMENT:?missing SIAB_DEPLOYMENT}
radial_rows=${RADIAL_ROWS:-31}
rank_tolerance=${RANK_TOLERANCE:-1e-12}
analyzer=$siab_deployment/SIAB/example_C_sternheimer/periodic_basis_optimization/analyze_periodic_candidate_spectrum.py

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_JOB_NUM_NODES" = 1
test -d "$dataset"
test -s "$coefficients"
test -s "$analyzer"
test ! -e "$output"
mkdir -p "$(dirname "$output")"

module load python/3.9.22
python_root=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
export PYTHONPATH=$python_root/lib-dynload:$python_root/site-packages
export LD_LIBRARY_PATH=$python_root/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
ulimit -s unlimited

time_file=${output%.json}.time
stdout_file=${output%.json}.stdout
/usr/bin/time -v -o "$time_file" \
  python3 "$analyzer" \
    --dataset "$dataset" \
    --coefficients "$coefficients" \
    --output "$output" \
    --label "$label" \
    --element C \
    --nu "$nu" \
    --radial-rows "$radial_rows" \
    --relative-rank-tolerance "$rank_tolerance" \
    > "$stdout_file"

python3 - "$output" <<'PY'
import json
import math
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
for key in (
    "maximum_overlap_condition",
    "maximum_eigenvalue_ev",
    "maximum_transition_energy_ev",
):
    if not math.isfinite(float(report[key])):
        raise SystemExit("non-finite spectrum diagnostic: " + key)
if not report.get("kpoints"):
    raise SystemExit("spectrum diagnostic contains no k points")
PY

{
  echo "status=success"
  echo "job_id=$SLURM_JOB_ID"
  echo "label=$label"
  echo "nu=$nu"
  echo "dataset=$dataset"
  echo "coefficients=$coefficients"
  echo "siab_deployment=$siab_deployment"
  sha256sum "$analyzer" "$coefficients" "$output"
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p' "$time_file"
  sed -n 's/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' "$time_file"
} > "${output%.json}.provenance.txt"

echo "CANDIDATE_SPECTRUM_OK label=$label output=$output"
