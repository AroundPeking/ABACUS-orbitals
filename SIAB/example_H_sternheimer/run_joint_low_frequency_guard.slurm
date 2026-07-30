#!/bin/bash
#SBATCH -J h_siab_lf_guard
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH -t 1-00:00:00
#SBATCH -o low_frequency_guard.%j.out
#SBATCH -e low_frequency_guard.%j.err

set -euo pipefail

script_dir=${SLURM_SUBMIT_DIR:?}
source_root=$(cd "$script_dir/../.." && pwd)
optimizer="$source_root/SIAB/opt_orb_pytorch_dpsi/main.py"

test -s "$script_dir/INPUT"
test -s "$script_dir/SOURCE_COMMIT"
test -s "$script_dir/SOURCE_MANIFEST.sha256"
test -s "$optimizer"
test ! -e "$script_dir/ORBITAL_RESULTS.txt"
test ! -e "$script_dir/ORBITAL_1U.dat"
test ! -e "$script_dir/Spillage.dat"

cd "$source_root"
sha256sum -c "$script_dir/SOURCE_MANIFEST.sha256"

check_hash() {
  local expected=$1
  local path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

data="$script_dir/data"
check_hash bed58ebf61cb513da892658b848f881f724feba7f50fe64f7a0b6252bb8e0c8c \
  "$data/OUT.H-atom-ST/sternheimer_matrix.dat"
check_hash 7d02cc86c20bfb34bd1efa3ca8f1a09f70aec0fbf7079d954d15e1bacc875505 \
  "$data/OUT.H-STRU2-8-0.7/orb_matrix.0.dat"
check_hash 80712780be91a38e562d931028ff29053ceaa7ffbb6860ad0d0c5b96e059e015 \
  "$data/OUT.H-STRU2-8-0.7/orb_matrix.1.dat"
check_hash 834741f0a3fe69fb9c53ac0c2264e2f1c4d1827e26bf87df66bd14e0271c0648 \
  "$data/OUT.H-STRU2-8-0.9/orb_matrix.0.dat"
check_hash 391e876b9fec0c3834a5a6cf5e479d58a53b4ca6e539b8f951bf8f76f9a875e7 \
  "$data/OUT.H-STRU2-8-0.9/orb_matrix.1.dat"
check_hash 23326839e68011d8877677f5021da4be032641aa263abe0ab02d40a561f9edad \
  "$data/OUT.H-STRU2-8-1.3/orb_matrix.0.dat"
check_hash d00fbea91f2dddb346aee7fd5c406743a4f74c51bd49fb5b0009f2776f2622ab \
  "$data/OUT.H-STRU2-8-1.3/orb_matrix.1.dat"

source /etc/profile.d/modules.sh
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load apps/PyTorch/2.1.0/pytorch-2.1.0-dtk2310

export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export MKL_DYNAMIC=FALSE

cd "$script_dir"
echo "date_start=$(date +%F_%T)"
echo "pwd=$PWD"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "source_commit=$(cat SOURCE_COMMIT)"
echo "omp_threads=$OMP_NUM_THREADS"
python -c 'import sys, torch; print("python=" + sys.version.replace("\n", " ")); print("torch=" + torch.__version__); print("torch_threads=" + str(torch.get_num_threads()))'
sha256sum INPUT SOURCE_COMMIT SOURCE_MANIFEST.sha256

/usr/bin/time -v python "$optimizer" > run.log 2>&1

test -s ORBITAL_RESULTS.txt
test -s ORBITAL_1U.dat
test -s Spillage.dat
grep -q '^Mode = st_dpsi_joint$' ORBITAL_RESULTS.txt
grep -q '^Low-frequency guard weight = 1.0000000000e+01$' ORBITAL_RESULTS.txt
grep -q '^Low-frequency guard tolerance = 0.0000000000e+00$' ORBITAL_RESULTS.txt
grep -q 'sternheimer_lowest_frequency' Spillage.dat
grep -q 'regularization_low_frequency' Spillage.dat
python - <<'PY'
from pathlib import Path

values = {}
for line in Path("ORBITAL_RESULTS.txt").read_text().splitlines():
    if " = " not in line:
        continue
    key, value = line.split(" = ", 1)
    try:
        values[key] = float(value)
    except ValueError:
        pass

assert values["Final lowest-frequency ST loss"] <= (
    values["Initial lowest-frequency ST loss"] * (1.0 + 1.0e-12)
)
PY
sha256sum ORBITAL_RESULTS.txt ORBITAL_1U.dat Spillage.dat run.log
echo "date_end=$(date +%F_%T)"
