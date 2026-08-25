#!/bin/bash
#SBATCH --job-name=c24_selected_qavg
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-selected-qavg-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-selected-qavg-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_SELECTED_QAVG_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
sos_body=$campaign/runs/product-pca-20260825/ordinary-sos-selected-grid-full-bz-reader-fractional-nfreq6-d4810f73
delta_body=$campaign/runs/product-pca-20260825/matched-delta-selected-grid-symmetry-reader-nfreq6-d4810f73
headwing=$(readlink -f "$campaign/runs/product-pca-20260825/headwing-selected-fixed-prefix-ad29464fd/latest")
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$campaign/runs/product-pca-20260825/matched-selected-full-qavg-d4810f73

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$librpa"
test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -x "$python"
grep -qx 'status=success' "$sos_body/provenance.txt"
grep -qx 'status=success' "$delta_body/provenance.txt"
grep -qx 'scope body_only_no_analytic_headwing' "$sos_body/RESULT_SUMMARY.txt"
grep -qx 'scope body_only_no_analytic_headwing' "$delta_body/RESULT_SUMMARY.txt"
grep -qx 'status=success' "$headwing/provenance.txt"
test -s "$headwing/pyatb_librpa_df/velocity_matrix"
mkdir -p "$work"

prepare_case() {
  local label=$1 source_root=$2
  local case_root=$work/$label
  local reader=$case_root/reader_v1
  mkdir -p "$reader"
  while IFS= read -r -d '' path; do
    ln -s "$path" "$reader/$(basename "$path")"
  done < <(find -L "$source_root/reader_v1" -maxdepth 1 -type f -size +0c -print0 | sort -z)
  ln -s "$headwing/pyatb_librpa_df" "$reader/pyatb_librpa_df"
  sed -E \
    -e "s|^[[:space:]]*input_dir[[:space:]]*=.*|input_dir = $reader/|" \
    -e 's|^[[:space:]]*replace_w_head[[:space:]]*=.*|replace_w_head = true|' \
    -e 's|^[[:space:]]*sqrt_coulomb_threshold[[:space:]]*=.*|sqrt_coulomb_threshold = 1e-5|' \
    "$source_root/librpa.in" > "$case_root/librpa.in"
  cat >> "$case_root/librpa.in" <<'EOF'
option_dielect_func = 3
rpa_headwing_mode = qavg
rpa_headwing_body_start = 1
EOF
}

prepare_case sos "$sos_body"
prepare_case delta "$delta_body"
find -L "$work/sos/reader_v1" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work/SOS_INPUT_SHA256.txt"
find -L "$work/delta/reader_v1" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work/DELTA_INPUT_SHA256.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

for label in sos delta; do
  cd "$work/$label"
  /usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
  grep -q 'libRPA finished successfully' librpa.out
  grep -q 'Reading head/wing input from' librpa.out
  grep -q 'replace_w_head = true' librpa.out
  grep -q 'rpa_headwing_mode = qavg' librpa.out
done

sos_full=$(awk '/^\| Total EcRPA:/ {value=$NF} END {print value}' "$work/sos/librpa.out")
delta_full=$(awk '/Total Sternheimer EcRPA:/ {value=$(NF-1)} END {print value}' "$work/delta/librpa.out")
sos_body_energy=$(awk '$1=="ecrpa_ha" {print $2}' "$sos_body/RESULT_SUMMARY.txt")
delta_body_energy=$(awk '$1=="delta_st_ecrpa_ha" {print $2}' "$delta_body/RESULT_SUMMARY.txt")
test -n "$sos_full"
test -n "$delta_full"
test -n "$sos_body_energy"
test -n "$delta_body_energy"

"$python" - "$sos_full" "$delta_full" "$sos_body_energy" "$delta_body_energy" > "$work/RESULT_SUMMARY.txt" <<'PY'
import math
import sys

sos, delta, sos_body, delta_body = map(float, sys.argv[1:])
assert all(math.isfinite(value) for value in (sos, delta, sos_body, delta_body))
difference = sos - delta
kcal_per_c = difference * 627.5094740631 / 2.0
threshold = 0.1
print("status success")
print("scope full_qavg_shared_selected_headwing")
print(f"sos_ecrpa_ha {sos:.12f}")
print(f"delta_st_ecrpa_ha {delta:.12f}")
print(f"sos_headwing_correction_ha {sos - sos_body:.12e}")
print(f"delta_st_headwing_correction_ha {delta - delta_body:.12e}")
print(f"sos_minus_delta_st_ha {difference:.12e}")
print(f"sos_minus_delta_st_kcal_per_c {kcal_per_c:.9f}")
print(f"acceptance_threshold_kcal_per_c {threshold:.6f}")
print("basis_full_qavg_gate " + ("pass" if abs(kcal_per_c) <= threshold else "fail"))
print("headwing_convergence_gate pending")
PY

{
  echo status=success
  echo purpose=periodic_c_selected_candidate_matched_full_qavg
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo headwing_source=$headwing
  echo headwing_convergence_gate=pending
  echo body_sos_source=$sos_body
  echo body_delta_source=$delta_body
  echo shared_headwing=yes
  sha256sum "$0" "$librpa" "$work/sos/librpa.in" "$work/delta/librpa.in" \
    "$work/SOS_INPUT_SHA256.txt" "$work/DELTA_INPUT_SHA256.txt" "$work/RESULT_SUMMARY.txt"
} > "$work/provenance.txt"
cat "$work/RESULT_SUMMARY.txt"
echo "C24_SELECTED_QAVG_OK work=$work"
