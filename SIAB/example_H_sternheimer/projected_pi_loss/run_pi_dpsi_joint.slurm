#!/bin/bash
#SBATCH --job-name=siab_pi_joint
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --output=pi_joint.%j.out
#SBATCH --error=pi_joint.%j.err

set -euo pipefail

case_root=${CASE_ROOT:?CASE_ROOT must name the immutable campaign directory}
source_root="$case_root/code"
input_root="$case_root/inputs"
run_root="$case_root/run"
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python
matplotlib_root=/work1/ghj/runtime/siab-projected-pi-mpl-20260801

test -d "$source_root/.git"
test -d "$input_root"
test -s "$case_root/SOURCE_COMMIT"
test -s "$case_root/SOURCE_MANIFEST.sha256"
test -s "$case_root/INPUTS.sha256"
test ! -e "$run_root"

source_commit=$(cat "$case_root/SOURCE_COMMIT")
test ${#source_commit} -eq 40
test "$(git -C "$source_root" rev-parse HEAD)" = "$source_commit"
git -C "$source_root" diff --quiet
git -C "$source_root" diff --cached --quiet

(
  cd "$source_root"
  sha256sum -c "$case_root/SOURCE_MANIFEST.sha256"
)
(
  cd "$case_root"
  sha256sum -c INPUTS.sha256
)

mkdir "$run_root"
install -m 0644 \
  "$source_root/SIAB/example_H_sternheimer/projected_pi_loss/INPUT.pi_dpsi_joint" \
  "$run_root/INPUT"

export PYTHONPATH="$matplotlib_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "date_start=$(date +%F_%T)"
echo "host=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:?}"
echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:?}"
echo "source_commit=$source_commit"
echo "omp_num_threads=$OMP_NUM_THREADS"
"$python" - <<'PY'
import matplotlib
import numpy
import sys
import torch

print(f"python_version={sys.version.split()[0]}")
print(f"numpy_version={numpy.__version__}")
print(f"torch_version={torch.__version__}")
print(f"matplotlib_version={matplotlib.__version__}")
print(f"torch_threads={torch.get_num_threads()}")
PY

cd "$run_root"
/usr/bin/time -v -o time_verbose.txt \
  "$python" "$source_root/SIAB/opt_orb_pytorch_dpsi/main.py" \
  > run.log 2>&1

test -s ORBITAL_RESULTS.txt
test -s ORBITAL_1U.dat
test -s Spillage.dat
test -s PROJECTED_PI_METADATA.json
grep -q '^Mode = pi_dpsi_joint$' ORBITAL_RESULTS.txt
grep -q 'projected_pi' Spillage.dat
"$python" -m json.tool PROJECTED_PI_METADATA.json > /dev/null
sha256sum INPUT ORBITAL_RESULTS.txt ORBITAL_1U.dat Spillage.dat \
  PROJECTED_PI_METADATA.json run.log time_verbose.txt > OUTPUTS.sha256
echo "date_end=$(date +%F_%T)"
