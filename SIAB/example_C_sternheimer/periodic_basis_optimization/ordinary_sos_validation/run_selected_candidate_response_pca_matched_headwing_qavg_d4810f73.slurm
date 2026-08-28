#!/bin/bash
#SBATCH --job-name=c24_rsp_qavg
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-rsp-qavg-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-rsp-qavg-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_RESPONSE_PCA_QAVG_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
sos_body=$campaign/runs/product-pca-20260825/ordinary-sos-response-aware-full-bz-nfreq6-d4810f73
sos_summary=$sos_body/POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt
sos_provenance=$sos_body/POSTPROCESS_5c5570cf_RETRY1_PROVENANCE.txt
delta_body=$campaign/runs/product-pca-20260825/matched-delta-response-aware-fixed-prefix-reader-nfreq6-d4810f73
headwing=$(readlink -f "$campaign/runs/product-pca-20260825/headwing-selected-fixed-prefix-ad29464fd/latest")
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
frequency=$sos_body/SELECTED_SOS_FREQUENCY_GRID.dat
frequency_manifest=$sos_body/SELECTED_SOS_FREQUENCY_GRID.json
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$campaign/runs/product-pca-20260825/matched-response-aware-full-qavg-d4810f73

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$librpa" && test -x "$python"
test -s "$build_provenance" && test -s "$frequency" && test -s "$frequency_manifest"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
grep -qx 'status success' "$sos_summary"
grep -qx 'scope body_only_no_analytic_headwing' "$sos_summary"
grep -qx 'auxiliary_basis response_aware_fixed_prefix_product_pca' "$sos_summary"
grep -qx 'auxiliary_dimension 522' "$sos_summary"
grep -qx 'status=success' "$sos_provenance"
grep -qx 'status=success' "$delta_body/provenance.txt"
grep -qx 'scope body_only_no_analytic_headwing' "$delta_body/RESULT_SUMMARY.txt"
grep -qx 'frequency_grid_source response_aware_selected_sos' "$delta_body/RESULT_SUMMARY.txt"
grep -qx 'status=success' "$headwing/provenance.txt"
grep -qx 'headwing_basis=112' "$headwing/provenance.txt"
grep -qx 'headwing_kpoints=64' "$headwing/provenance.txt"
test -s "$headwing/pyatb_librpa_df/velocity_matrix"

frequency_sha=$(sha256sum "$frequency" | awk '{print $1}')
"$python" - "$frequency_manifest" "$frequency_sha" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert manifest["status"] == "success"
assert manifest["nfreq"] == 6
assert manifest["frequency_grid_sha256"] == sys.argv[2]
PY

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 OMP_PLACES=cores OMP_PROC_BIND=close I_MPI_FABRICS=shm:ofi

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
sos_body_energy=$(awk '$1=="ecrpa_ha" {print $2}' "$sos_summary")
delta_body_energy=$(awk '$1=="delta_st_ecrpa_ha" {print $2}' "$delta_body/RESULT_SUMMARY.txt")
test -n "$sos_full" && test -n "$delta_full"
test -n "$sos_body_energy" && test -n "$delta_body_energy"

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
print("auxiliary_basis response_aware_fixed_prefix_product_pca")
print("frequency_grid_source response_aware_selected_sos")
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
  echo purpose=periodic_c_response_aware_matched_full_qavg
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo frequency_grid_sha256=$frequency_sha
  echo headwing_source=$headwing
  echo headwing_convergence_gate=pending
  echo body_sos_source=$sos_body
  echo body_delta_source=$delta_body
  echo auxiliary_basis=response_aware_fixed_prefix_product_pca
  echo shared_headwing=yes
  sha256sum "$0" "$librpa" "$frequency" "$frequency_manifest" \
    "$work/sos/librpa.in" "$work/delta/librpa.in" \
    "$work/SOS_INPUT_SHA256.txt" "$work/DELTA_INPUT_SHA256.txt" "$work/RESULT_SUMMARY.txt"
} > "$work/provenance.txt"
cat "$work/RESULT_SUMMARY.txt"
echo "C24_RESPONSE_PCA_QAVG_OK work=$work"
