#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 DATASET COEFFICIENTS OUTPUT_DIRECTORY" >&2
    exit 2
fi

dataset=$(realpath "$1")
coefficients=$(realpath "$2")
output_directory=$3
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
evaluator=$script_directory/evaluate_periodic_basis.py

test -d "$dataset"
test ! -L "$dataset"
test -s "$dataset/manifest.dat"
test -s "$dataset/status.dat"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
test -f "$coefficients"
test ! -L "$coefficients"
test -s "$coefficients"
test ! -e "$output_directory"
mkdir -p "$output_directory"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
export OMP_DYNAMIC=FALSE

{
    echo status=running
    echo purpose=periodic_c_bessel_mother_capacity_gate
    echo dataset=$dataset
    echo coefficients=$coefficients
    echo omp_threads=$OMP_NUM_THREADS
    echo mkl_threads=$MKL_NUM_THREADS
    sha256sum "$dataset/manifest.dat" "$dataset/status.dat" "$coefficients"
} > "$output_directory/provenance.txt"

/usr/bin/time -v -o "$output_directory/evaluate.time" \
    python3 "$evaluator" \
        --dataset "$dataset" \
        --coefficients "$coefficients" \
        --output "$output_directory/capacity.json" \
        --nu 3,3,2,0,0 \
        --mother-response-tolerance 1e-3 \
        --occupied-capture-tolerance 1e-6 \
        > "$output_directory/evaluate.log" 2>&1

python3 - "$output_directory/capacity.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
if report["mother"]["capacity_gate"] != "PASS":
    raise SystemExit("mother-space capacity gate failed")
if not report["optimization_allowed"]:
    raise SystemExit("optimization is not allowed")
if report["candidate"]["evaluation_gate"] != "PASS":
    raise SystemExit("initial candidate basis gate failed")
PY

{
    echo status=success
    sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p' \
        "$output_directory/evaluate.time"
    sha256sum "$output_directory/capacity.json"
} >> "$output_directory/provenance.txt"

echo "PERIODIC_C_CAPACITY_GATE_OK output=$output_directory"
