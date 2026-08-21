#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 TRAINING_INPUT_DIR NEW_OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
input_dir=$(cd "$1" && pwd)
output_dir=$2
python=${SIAB_PYTHON:-python3}

if [[ -e "$output_dir" ]]; then
    echo "output path already exists: $output_dir" >&2
    exit 2
fi

mkdir -p "$output_dir/run"
ln -s "$input_dir" "$output_dir/inputs"
cp "$script_dir/SIAB_INPUT.json" "$output_dir/run/INPUT"

cd "$output_dir/run"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
"$python" "$repo_root/SIAB/opt_orb_pytorch_dpsi/main.py"

test -s ORBITAL_RESULTS.txt
test -s ORBITAL_1U.dat
test -s Spillage.dat
