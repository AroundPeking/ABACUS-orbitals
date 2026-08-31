#!/bin/bash
#SBATCH --job-name=c_ctrl_f_gate
#SBATCH --partition=48cp2,p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-controlled-f-gate-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-controlled-f-gate-%j.err

set -euo pipefail
trap 'status=$?; echo "C_CONTROLLED_F_GATE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${REPO_ROOT:?missing immutable source deployment}"
: "${EXPECTED_SIAB_COMMIT:?missing source commit}"
: "${BASE_CANDIDATE_ROOT:?missing accepted 3s3p2d base candidate}"
: "${CANDIDATE_ROOT:?missing controlled f candidate root}"
: "${GATE_ROOT:?missing controlled f gate root}"
second_primitive_amplitude=${SECOND_PRIMITIVE_AMPLITUDE:-0}

case "${SLURM_JOB_PARTITION:?}" in
  p1|48cp2) ;;
  *) echo "unsupported df partition: $SLURM_JOB_PARTITION" >&2; exit 1 ;;
esac
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test "$(cat "$REPO_ROOT/SOURCE_COMMIT")" = "$EXPECTED_SIAB_COMMIT"
test ! -e "$CANDIDATE_ROOT"
test ! -e "$GATE_ROOT"
grep -qx 'status=success' "$BASE_CANDIDATE_ROOT/provenance.txt"

module load python/3.9.22
runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
python_exe=$(command -v python3)
code=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization
preparer=$code/prepare_controlled_f_candidate.py
comparator=$code/compare_periodic_candidates.py
analyzer=$code/analyze_periodic_candidate_spectrum.py
original=$REPO_ROOT/SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt
producer_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q3=$producer_root/c-diamond-g24-fd8-nfreq6-q3-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q3_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
base_coefficients=$(
  "$python_exe" - "$BASE_CANDIDATE_ROOT/CANDIDATE.json" "$BASE_CANDIDATE_ROOT" <<'PY'
import json, pathlib, sys
manifest = json.load(open(sys.argv[1], encoding="ascii"))
path = pathlib.Path(sys.argv[2]) / manifest["coefficients_filename"]
print(path.resolve(strict=True))
PY
)

for path in "$preparer" "$comparator" "$analyzer" "$original" "$base_coefficients" "$q3/manifest.dat" "$q3/status.dat"; do
  test -s "$path"
done
grep -qx 'status success' "$q3/status.dat"
grep -qx 'all_converged yes' "$q3/status.dat"
grep -qx 'selected_iq 43' "$q3/manifest.dat"
grep -qx 'kernel full_coulomb' "$q3/manifest.dat"

export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages:$REPO_ROOT:$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p "$GATE_ROOT"
{
  echo status=running
  echo purpose=controlled_lowest_f_q3_and_spectrum_gate
  echo source_commit=$EXPECTED_SIAB_COMMIT
  echo base_candidate_root=$BASE_CANDIDATE_ROOT
  echo selected_iq=43
  echo second_primitive_amplitude=$second_primitive_amplitude
  sha256sum "$preparer" "$comparator" "$analyzer" "$original" "$base_coefficients" "$q3/manifest.dat" "$q3/status.dat"
} > "$GATE_ROOT/provenance.txt"

"$python_exe" "$preparer" --source "$base_coefficients" --root "$CANDIDATE_ROOT" \
  --second-primitive-amplitude "$second_primitive_amplitude" \
  > "$GATE_ROOT/prepare.out"

candidate_coefficients=$(
  "$python_exe" - "$CANDIDATE_ROOT/CANDIDATE.json" "$CANDIDATE_ROOT" <<'PY'
import json, pathlib, sys
manifest = json.load(open(sys.argv[1], encoding="ascii"))
print((pathlib.Path(sys.argv[2]) / manifest["coefficients_filename"]).resolve(strict=True))
PY
)

/usr/bin/time -v -o "$GATE_ROOT/compare.time" \
  "$python_exe" -u "$comparator" \
    --dataset "$q3" \
    --candidate original-tzdp:"$original":3,3,2,0,0 \
    --candidate base-p3:"$base_coefficients":3,3,2,0,0 \
    --candidate controlled-f:"$candidate_coefficients":3,3,2,1,0 \
    --occupied-capture-floor 0.9998982409775239 \
    --output "$GATE_ROOT/COMPARISON_RESULT.json" \
    > "$GATE_ROOT/compare.log" 2>&1

"$python_exe" "$analyzer" \
  --dataset "$q3" --coefficients "$original" --nu 3,3,2,0,0 \
  --label original-tzdp --output "$GATE_ROOT/ORIGINAL_SPECTRUM.json" >/dev/null
"$python_exe" "$analyzer" \
  --dataset "$q3" --coefficients "$candidate_coefficients" --nu 3,3,2,1,0 \
  --label controlled-f --output "$GATE_ROOT/CANDIDATE_SPECTRUM.json" >/dev/null

"$python_exe" - "$CANDIDATE_ROOT/CANDIDATE.json" "$GATE_ROOT/COMPARISON_RESULT.json" \
  "$GATE_ROOT/ORIGINAL_SPECTRUM.json" "$GATE_ROOT/CANDIDATE_SPECTRUM.json" \
  "$GATE_ROOT/GATE_RESULT.json" <<'PY'
import json, math, pathlib, sys
candidate, comparison, original, selected = (
    json.load(open(path, encoding="ascii")) for path in sys.argv[1:5]
)
records = {record["label"]: record for record in comparison["candidates"]}
assert set(records) == {"original-tzdp", "base-p3", "controlled-f"}
controlled = records["controlled-f"]
assert controlled["nu"] == [3, 3, 2, 1, 0]
assert controlled["ao_count_cell"] == 58
capture = float(controlled["minimum_occupied_capture"])
condition_ratio = float(selected["maximum_overlap_condition"]) / float(original["maximum_overlap_condition"])
virtual_ratio = float(selected["maximum_eigenvalue_ev"]) / float(original["maximum_eigenvalue_ev"])
metrics = {
    "q3_pi_error": float(controlled["global_weighted_relative_pi_error"]),
    "q3_trace_log_error": float(controlled["global_weighted_relative_trace_log_error"]),
    "minimum_occupied_capture": capture,
    "maximum_overlap_condition_ratio": condition_ratio,
    "maximum_eigenvalue_ratio": virtual_ratio,
}
assert all(math.isfinite(value) for value in metrics.values())
assert capture >= 0.9998982409775239
assert condition_ratio < 3.0
assert virtual_ratio < 1.5
diagnostics = candidate["f_diagnostics"]
assert diagnostics["interior_node_count"] == 0
assert abs(float(diagnostics["radial_norm"]) - 1.0) <= 1.0e-10
result = {
    "status": "success",
    "gate": "pass",
    "scope": "q3 capture and Galerkin spectrum stability; not an SOS energy gate",
    "candidate_orbital_sha256": candidate["orbital_sha256"],
    **metrics,
}
pathlib.Path(sys.argv[5]).write_text(
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="ascii",
)
PY

{
  echo status=success
  sha256sum "$CANDIDATE_ROOT/CANDIDATE.json" "$CANDIDATE_ROOT/provenance.txt" \
    "$GATE_ROOT/COMPARISON_RESULT.json" "$GATE_ROOT/ORIGINAL_SPECTRUM.json" \
    "$GATE_ROOT/CANDIDATE_SPECTRUM.json" "$GATE_ROOT/GATE_RESULT.json"
} >> "$GATE_ROOT/provenance.txt"
printf 'success\n' > "$GATE_ROOT/STATUS"
echo "C_CONTROLLED_F_GATE_OK candidate_root=$CANDIDATE_ROOT gate_root=$GATE_ROOT"
