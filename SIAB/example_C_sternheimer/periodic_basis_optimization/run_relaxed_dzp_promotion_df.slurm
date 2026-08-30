#!/bin/bash
#SBATCH --job-name=c_relaxed_promote
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-relaxed-promote-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-relaxed-promote-%j.err

set -euo pipefail
trap 'status=$?; echo "C_RELAXED_DZP_PROMOTION_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${REPO_ROOT:?missing immutable source deployment}"
: "${OPTIMIZER_ROOT:?missing successful optimizer root}"
: "${CANDIDATE_ROOT:?missing immutable candidate root}"
: "${EXPECTED_SIAB_COMMIT:?missing source commit}"

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test "$(cat "$REPO_ROOT/.git/HEAD")" = "$EXPECTED_SIAB_COMMIT"
test ! -e "$CANDIDATE_ROOT"
gate_root=${CANDIDATE_ROOT}.gates
test ! -e "$gate_root"

module load python/3.9.22
runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
python_exe=$(command -v python3)
optimization=$OPTIMIZER_ROOT/result/OPTIMIZATION_RESULT.json
candidate=$OPTIMIZER_ROOT/result/ORBITAL_RESULTS.txt
checkpoint=$OPTIMIZER_ROOT/result/BEST_ORBITAL_CHECKPOINT.txt
original=$REPO_ROOT/SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt
producer_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1
q3=$producer_root/c-diamond-g24-fd8-nfreq6-q3-product-pca-711af860c-kpar8-retry1/OUT.C_DIAMOND_G24_FD8_NFREQ6_Q3_PRODUCT_PCA/STERNHEIMER_BASIS_OPT_V1
comparator=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/compare_periodic_candidates.py
analyzer=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/analyze_periodic_candidate_spectrum.py
exporter=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/export_periodic_orbitals.py
promoter=$REPO_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/prepare_relaxed_dzp_candidate.py

for path in "$optimization" "$candidate" "$checkpoint" "$original" "$q3/manifest.dat" "$q3/status.dat" "$comparator" "$analyzer" "$exporter" "$promoter"; do
  test -s "$path"
done
grep -qx 'status=success' "$OPTIMIZER_ROOT/provenance.txt"
grep -qx 'status success' "$q3/status.dat"
grep -qx 'all_converged yes' "$q3/status.dat"
grep -qx 'selected_iq 43' "$q3/manifest.dat"
grep -qx 'kernel full_coulomb' "$q3/manifest.dat"
cmp "$candidate" "$checkpoint"

export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages:$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p "$gate_root"
{
  echo status=running
  echo purpose=relaxed_dzp_independent_q3_spectrum_and_export_gate
  echo siab_commit=$EXPECTED_SIAB_COMMIT
  echo optimizer_root=$OPTIMIZER_ROOT
  echo selected_iq=43
  sha256sum "$0" "$optimization" "$candidate" "$checkpoint" "$original" "$q3/manifest.dat" "$comparator" "$analyzer" "$exporter" "$promoter"
} > "$gate_root/provenance.txt"

/usr/bin/time -v -o "$gate_root/compare.time" \
  "$python_exe" -u "$comparator" \
    --dataset "$q3" \
    --candidate original-tzdp:"$original":3,3,2,0,0 \
    --candidate relaxed-dzp:"$candidate":3,3,2,0,0 \
    --occupied-capture-floor 0.9998982409775239 \
    --output "$gate_root/COMPARISON_RESULT.json" \
    > "$gate_root/compare.log" 2>&1

"$python_exe" "$analyzer" \
  --dataset "$q3" --coefficients "$original" --nu 3,3,2,0,0 \
  --label original-tzdp --output "$gate_root/ORIGINAL_SPECTRUM.json"
"$python_exe" "$analyzer" \
  --dataset "$q3" --coefficients "$candidate" --nu 3,3,2,0,0 \
  --label relaxed-dzp --output "$gate_root/CANDIDATE_SPECTRUM.json"

"$python_exe" "$exporter" \
  --input "$candidate" \
  --output "$gate_root/C_gga_10au_100Ry_3s3p2d_relaxed_response.orb" \
  --element C \
  --nu 3,3,2,0,0 \
  --radial-rows 31 \
  --ecut-ry 100 \
  --rcut-bohr 10 \
  --dr-bohr 0.01 \
  --smoothing-sigma-bohr 0.1 \
  > "$gate_root/EXPORT_METADATA.json"

"$python_exe" "$promoter" \
  --optimizer-result "$optimization" \
  --comparison "$gate_root/COMPARISON_RESULT.json" \
  --original-spectrum "$gate_root/ORIGINAL_SPECTRUM.json" \
  --candidate-spectrum "$gate_root/CANDIDATE_SPECTRUM.json" \
  --orbital "$gate_root/C_gga_10au_100Ry_3s3p2d_relaxed_response.orb" \
  --output-directory "$CANDIDATE_ROOT" \
  > "$gate_root/promote.log"

test -s "$CANDIDATE_ROOT/CANDIDATE.json"
grep -q '"pre_pbe_gate": "pass"' "$CANDIDATE_ROOT/CANDIDATE.json"
{
  echo status=success
  sha256sum "$gate_root/COMPARISON_RESULT.json" "$gate_root/ORIGINAL_SPECTRUM.json" "$gate_root/CANDIDATE_SPECTRUM.json" "$gate_root/EXPORT_METADATA.json" "$CANDIDATE_ROOT/CANDIDATE.json"
} >> "$gate_root/provenance.txt"
printf 'success\n' > "$gate_root/STATUS"
echo "C_RELAXED_DZP_PROMOTION_OK candidate_root=$CANDIDATE_ROOT"
