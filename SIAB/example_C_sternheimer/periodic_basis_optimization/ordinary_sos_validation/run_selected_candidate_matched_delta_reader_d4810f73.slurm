#!/bin/bash
#SBATCH --job-name=c24_selected_delta_rpa
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-selected-delta-rpa-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-selected-delta-rpa-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_SELECTED_MATCHED_DELTA_RPA_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
validation=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
merge_script=$validation/merge_sternheimer_symmetry_manifests.py
selection_root=$campaign/runs/product-pca-20260825/selected-fixed-prefix-checkpoint-500
selection=$selection_root/SELECTED_CANDIDATE.json
response_root=$campaign/runs/product-pca-20260825/matched-delta-selected-fixed-prefix-nfreq6-ad29464fd
grid_root=$campaign/runs/product-pca-20260825/grid-coulomb-selected-fixed-prefix-full-bz-ad29464fd
sos_root=$campaign/runs/product-pca-20260825/ordinary-sos-selected-grid-full-bz-reader-fractional-nfreq6-d4810f73
frequency=$sos_root/SELECTED_SOS_FREQUENCY_GRID.dat
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$campaign/runs/product-pca-20260825/matched-delta-selected-grid-symmetry-reader-nfreq6-d4810f73
reader=$work/reader_v1

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -s "$selection"
test -s "$frequency"
test -s "$merge_script"
test -x "$python"
test -x "$librpa"
test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
grep -qx "driver=$librpa" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "${#expected_librpa_sha}" = 64
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
grep -qx 'status=success' "$sos_root/provenance.txt"
grep -qx 'frequency_grid_source=selected_candidate_sos' "$sos_root/provenance.txt"

q1=$response_root/q1
test -d "$q1"
grep -qx 'status=success' "$q1/RUN_PROVENANCE.txt"
grep -qx "selected_sos_job=${SELECTED_SOS_JOB_ID:?missing selected SOS job id}" "$q1/RUN_PROVENANCE.txt"
grep -qx "grid_coulomb_array_job=${GRID_COULOMB_ARRAY_JOB_ID:?missing grid Coulomb job id}" "$q1/RUN_PROVENANCE.txt"
report=$(find "$q1" -path '*/OUT.*/STERNHEIMER_CHI0.dat' -type f -print -quit)
test -s "$report"
mapfile -t representatives < <(awk '$1=="sternheimer_canonical_q_indices" {for(i=2;i<=NF;i++) print $i}' "$report")
test "${#representatives[@]}" -gt 1

read -r q1_state q1_exit < <(sacct -n -X -j "${Q1_RESPONSE_JOB_ID:?missing q1 response job id}" --format=State,ExitCode -P | awk -F'|' 'NF>=2 && $1!="" {print $1, $2; exit}')
read -r array_state array_exit < <(sacct -n -X -j "${RESPONSE_ARRAY_JOB_ID:?missing response array job id}" --format=State,ExitCode -P | awk -F'|' '$1!~/\./ && NF>=2 && $1!="" {print $1, $2; exit}')
test "$q1_state" = COMPLETED
test "$q1_exit" = 0:0
test "$array_state" = COMPLETED
test "$array_exit" = 0:0

run_dirs=()
channels=
frequency_sha=$(sha256sum "$frequency" | awk '{print $1}')
for iq in "${representatives[@]}"; do
  qdir=$response_root/q${iq}
  test -d "$qdir"
  grep -qx 'status=success' "$qdir/RUN_PROVENANCE.txt"
  grep -qx "full_q_index=$iq" "$qdir/RUN_PROVENANCE.txt"
  grep -qx "frequency_grid_sha256=$frequency_sha" "$qdir/RUN_PROVENANCE.txt"
  current=$(awk -F= '$1=="abfs_channels" {print $2}' "$qdir/RUN_PROVENANCE.txt")
  test -n "$current"
  if test -z "$channels"; then channels=$current; else test "$channels" = "$current"; fi
  run_dirs+=("$qdir")
done

mkdir -p "$reader"
"$python" "$merge_script" --output-dir "$reader" "${run_dirs[@]}" > "$work/merge.out"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out stru_out; do
  test -s "$q1/$name"
  ln -s "$q1/$name" "$reader/$name"
done
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(find "$q1" -maxdepth 1 -type f -name 'KS_eigenvector_*.dat' -size +0c -print0 | sort -z)
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(find "$q1" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c -print0 | sort -z)

for ordinal in $(seq 1 64); do
  grid_q=$grid_root/q${ordinal}
  matrix=$grid_q/v1_coulomb_grid_iq_${ordinal}_rank0.dat
  test -s "$matrix"
  test -s "$grid_q/GRID_MATRIX_HEADER.json"
  grep -qx 'status=success' "$grid_q/provenance.txt"
  grep -qx "full_q_ordinal=$ordinal" "$grid_q/provenance.txt"
  grep -qx "full_q_index=$ordinal" "$grid_q/provenance.txt"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done

grid_coulomb_file_count=$(find -L "$reader" -maxdepth 1 -type f -name 'v1_coulomb_grid_iq_*_rank*.dat' -size +0c | wc -l)
cs_file_count=$(find -L "$reader" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)
test "$grid_coulomb_file_count" -eq 64
test "$cs_file_count" -eq 1
"$python" - "$grid_root" "$channels" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
channels = int(sys.argv[2])
dimensions = {
    json.loads((root / f"q{ordinal}" / "GRID_MATRIX_HEADER.json").read_text(encoding="ascii"))["naux"]
    for ordinal in range(1, 65)
}
assert dimensions == {channels}
PY
cmp -s "$q1/basis_aux_out" "$grid_root/q1/basis_aux_out"
cmp -s "$q1/basis_out" "$grid_root/q1/basis_out"
cmp -s "$q1/basis_wfc_out" "$grid_root/q1/basis_wfc_out"

cat > "$work/librpa.in" <<EOF
task = sternheimer_rpa
input_dir = $reader/
version_coul_reader = 1
version_lri_reader = 1
prefix_lri_coeff = v1_Cs_data_
nfreq = 6
sqrt_coulomb_threshold = 1e-5
prefix_coul_full = v1_coulomb_grid_iq_
fn_sternheimer_qpoints = v1_sternheimer_qpoints.dat
fn_sternheimer_partial_manifest = v1_sternheimer_partial_manifest.dat
fn_sternheimer_symmetry_routes = v1_sternheimer_symmetry_routes.dat
fn_sternheimer_qstar_routes = v1_sternheimer_qstar_routes.dat
use_rpa_gamma = true
replace_w_head = false
parallel_routing = libri
use_symmetry_rpa = true
use_kpara_scf_eigvec = true
EOF

find -L "$reader" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work/INPUT_SHA256.txt"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_selected_candidate_matched_delta_st_grid_kernel_rpa
  echo q1_response_job=$Q1_RESPONSE_JOB_ID
  echo response_array_job=$RESPONSE_ARRAY_JOB_ID
  echo selected_sos_job=$SELECTED_SOS_JOB_ID
  echo grid_coulomb_array_job=$GRID_COULOMB_ARRAY_JOB_ID
  echo abacus_commit=ad29464fdc9125114cc052840415647a1b5a6ef1
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo frequency_grid_source=selected_candidate_sos
  echo frequency_grid_sha256=$frequency_sha
  echo auxiliary_basis_source=product_pca
  echo auxiliary_dimension=$channels
  echo pca_threshold=1e-4
  echo nfreq=6
  echo kmesh=4,4,4
  echo q_representative_count=${#representatives[@]}
  echo canonical_q_indices=$(IFS=,; echo "${representatives[*]}")
  echo full_q_count=64
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo analytic_headwing=no_body_screen
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$selection" "$frequency" "$merge_script" "$work/librpa.in" \
    "$work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
grep -q 'libRPA finished successfully' librpa.out
grep -q 'Total Sternheimer EcRPA:' librpa.out

sternheimer_ecrpa=$(awk '/Total Sternheimer EcRPA:/ {value=$(NF-1)} END {print value}' librpa.out)
sos_ecrpa=$(awk '$1=="ecrpa_ha" {print $2}' "$sos_root/RESULT_SUMMARY.txt")
test -n "$sternheimer_ecrpa"
test -n "$sos_ecrpa"
"$python" - "$sternheimer_ecrpa" "$sos_ecrpa" > RESULT_SUMMARY.txt <<'PY'
import math
import sys

delta_st = float(sys.argv[1])
sos = float(sys.argv[2])
assert math.isfinite(delta_st) and math.isfinite(sos)
difference = sos - delta_st
kcal_per_cell = difference * 627.5094740631
kcal_per_c = kcal_per_cell / 2.0
threshold = 0.1
print("status success")
print("scope body_only_no_analytic_headwing")
print("coulomb_kernel exact_rhs_full_periodic_poisson")
print("frequency_grid_source selected_candidate_sos")
print(f"delta_st_ecrpa_ha {delta_st:.12f}")
print(f"sos_ecrpa_ha {sos:.12f}")
print(f"sos_minus_delta_st_ha {difference:.12e}")
print(f"sos_minus_delta_st_kcal_per_cell {kcal_per_cell:.9f}")
print(f"sos_minus_delta_st_kcal_per_c {kcal_per_c:.9f}")
print(f"acceptance_threshold_kcal_per_c {threshold:.6f}")
print("basis_body_gate " + ("pass" if abs(kcal_per_c) <= threshold else "fail"))
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt
} >> provenance.txt

echo "C24_SELECTED_MATCHED_DELTA_RPA_OK work=$work"
