#!/bin/bash
#SBATCH --job-name=siab_pi_rank
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00

set -euo pipefail

case_root=${CASE_ROOT:?CASE_ROOT must name the staged feasibility directory}
code_root="$case_root/code"
input_root="$case_root/inputs"
result_root="$case_root/projected_pi_result"
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python
matplotlib_root=/work1/ghj/runtime/siab-projected-pi-mpl-20260801
source_commit=ecedcd8044df298e8693cfe19aa0b039cd3b5e07

export PYTHONPATH="$matplotlib_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "source_commit=$source_commit"
echo "host=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-none}"
echo "omp_num_threads=$OMP_NUM_THREADS"
echo "mkl_num_threads=$MKL_NUM_THREADS"
echo "openblas_num_threads=$OPENBLAS_NUM_THREADS"
"$python" - <<'PY'
import matplotlib
import numpy
import torch
import sys

print(f"python_version={sys.version.split()[0]}")
print(f"numpy_version={numpy.__version__}")
print(f"torch_version={torch.__version__}")
print(f"matplotlib_version={matplotlib.__version__}")
PY

sha256sum \
    "$input_root/H_sternheimer_matrix.dat" \
    "$input_root/H_STERNHEIMER_SIAB_SOURCE_V1.dat" \
    "$input_root/H_zero_order_identity.json" \
    "$input_root/H2_sternheimer_matrix.dat" \
    "$input_root/H2_STERNHEIMER_SIAB_SOURCE_V1.dat" \
    "$input_root/H2_zero_order_identity.json" \
    "$input_root/initial_tzdp_ORBITAL_RESULTS.txt" \
    "$input_root/fixed_dzp_joint_ORBITAL_RESULTS.txt" \
    "$input_root/low_frequency_guarded_ORBITAL_RESULTS.txt"

set +e
/usr/bin/time -v "$python" \
    "$code_root/SIAB/example_H_sternheimer/projected_pi_loss/analyze_projected_pi.py" \
    --h-response "$input_root/H_sternheimer_matrix.dat" \
    --h-source "$input_root/H_STERNHEIMER_SIAB_SOURCE_V1.dat" \
    --h-audit "$input_root/H_zero_order_identity.json" \
    --h2-response "$input_root/H2_sternheimer_matrix.dat" \
    --h2-source "$input_root/H2_STERNHEIMER_SIAB_SOURCE_V1.dat" \
    --h2-audit "$input_root/H2_zero_order_identity.json" \
    --initial "$input_root/initial_tzdp_ORBITAL_RESULTS.txt" \
    --joint "$input_root/fixed_dzp_joint_ORBITAL_RESULTS.txt" \
    --guarded "$input_root/low_frequency_guarded_ORBITAL_RESULTS.txt" \
    --output-dir "$result_root"
analysis_status=$?
set -e

echo "analysis_exit_status=$analysis_status"
exit "$analysis_status"
