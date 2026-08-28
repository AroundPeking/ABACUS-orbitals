#!/bin/bash
#SBATCH --job-name=c24_rsp_delta_q1
#SBATCH --partition=p1
#SBATCH --nodes=48
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=1
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-rsp-delta-q1-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-rsp-delta-q1-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C24_RESPONSE_PCA_MATCHED_DELTA_Q1_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1
selection_root=$campaign/runs/product-pca-20260825/selected-fixed-prefix-checkpoint-500
selection=$selection_root/SELECTED_CANDIDATE.json
orbital=$selection_root/C_gga_10au_100Ry_selected_product_pca.orb
sos_root=$campaign/runs/product-pca-20260825/ordinary-sos-response-aware-full-bz-nfreq6-d4810f73
frequency_source=$sos_root/SELECTED_SOS_FREQUENCY_GRID.dat
frequency_manifest=$sos_root/SELECTED_SOS_FREQUENCY_GRID.json
sos_summary=$sos_root/POSTPROCESS_5c5570cf_RETRY1_RESULT_SUMMARY.txt
frequency_name=SELECTED_SOS_FREQUENCY_GRID.dat
grid_q1=$campaign/runs/product-pca-20260825/response-aware-product-pca-q1-55d25e3c9
grid_rest=$campaign/runs/product-pca-20260825/response-aware-product-pca-full-bz-55d25e3c9
abacus=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534/abacus_3p
abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
fixed_nu=2,2,1,0,0
expected_naux=522
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
iq=${SLURM_ARRAY_TASK_ID:?missing q index}
root=$campaign/runs/product-pca-20260825/matched-delta-response-aware-fixed-prefix-nfreq6-55d25e3c9
work=$root/q${iq}
suffix=C_DIAMOND_RESPONSE_PCA_MATCHED_DELTA_Q${iq}

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 48
test "$SLURM_NTASKS_PER_NODE" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test "$iq" -ge 1
test "$iq" -le 64
test ! -e "$work"
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test -x "$python"
test -s "$selection" && test -s "$orbital"
test -s "$frequency_source" && test -s "$frequency_manifest" && test -s "$sos_summary"
grep -qx 'status success' "$sos_summary"
grep -qx 'scope body_only_no_analytic_headwing' "$sos_summary"
if test "$iq" = 1; then grid_q=$grid_q1; else grid_q=$grid_rest/q${iq}; fi
grid_matrix=$grid_q/v1_coulomb_grid_iq_${iq}_rank0.dat
test -s "$grid_matrix" && test -s "$grid_q/GRID_MATRIX_HEADER.json"

read -r selected_g expected_ao_count orbital_sha frequency_sha < <("$python" - "$selection" "$orbital" "$frequency_source" "$frequency_manifest" <<'PY'
import hashlib
import json
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
orbital = pathlib.Path(sys.argv[2])
frequency = pathlib.Path(sys.argv[3])
manifest = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="ascii"))
assert selection["label"] == "joint-three-g"
assert selection["nu"] == [3, 3, 2, 1, 3]
assert selection["ao_count_cell"] == 112
assert selection["minimum_occupied_capture"] >= selection["occupied_capture_floor"]
assert manifest["status"] == "success" and manifest["nfreq"] == 6
orbital_sha = hashlib.sha256(orbital.read_bytes()).hexdigest()
frequency_sha = hashlib.sha256(frequency.read_bytes()).hexdigest()
assert orbital_sha == selection["exported_orbital_sha256"]
assert frequency_sha == manifest["frequency_grid_sha256"]
print(3, 112, orbital_sha, frequency_sha)
PY
)

"$python" - "$grid_q/GRID_MATRIX_HEADER.json" <<'PY'
import json
import pathlib
import sys
header = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert header["status"] == "success"
assert header["naux"] == 522
PY

for asset in INPUT STRU KPT C_ONCV_PBE-1.0.upf; do test -s "$source/$asset"; done
mkdir -p "$work"
cp "$source/INPUT" "$source/STRU" "$source/KPT" "$source/C_ONCV_PBE-1.0.upf" \
  "$orbital" "$frequency_source" "$work/"
cd "$work"

set_input_key() {
  local key=$1 value=$2
  if awk -v key="$key" '$1 == key {found=1} END {exit !found}' INPUT; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]+.*|${key} ${value}|" INPUT
  else
    printf '%-36s %s\n' "$key" "$value" >> INPUT
  fi
}

sed -i -E 's|^C_gga_10au_100Ry_3s3p2d\.orb$|C_gga_10au_100Ry_selected_product_pca.orb|' STRU
grep -qx 'C_gga_10au_100Ry_selected_product_pca.orb' STRU
set_input_key suffix "$suffix"
set_input_key symmetry 1
set_input_key nbands "$expected_ao_count"
set_input_key kpar 8
set_input_key rpa 1
set_input_key out_librpa_reader_version 1
set_input_key out_sternheimer_librpa 1
set_input_key out_sternheimer_basis_opt 0
set_input_key sternheimer_q_index "$iq"
set_input_key sternheimer_nfreq 6
set_input_key sternheimer_frequency_grid_file "$frequency_name"
set_input_key sternheimer_frequency_mpi 1
set_input_key sternheimer_channel_mpi 0
set_input_key sternheimer_mpi_layout frequency_grouped
set_input_key sternheimer_delta 1
set_input_key sternheimer_fd_order 8
set_input_key exx_pca_threshold 1e-4
set_input_key rpa_pca_fixed_nu "$fixed_nu"
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 2
set_input_key nx 24
set_input_key ny 24
set_input_key nz 24

export LD_LIBRARY_PATH=/data/home/df_iopcas_ghj/app/elpa-build/lib:/data/app/libxc/6.2.2-intel/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE OMP_STACKSIZE=2G
export MKL_DYNAMIC=FALSE I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp I_MPI_FABRICS=shm:ofi
export ABACUS_STERNHEIMER_CHANNEL_THREADS=$OMP_NUM_THREADS
unset ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL ABACUS_STERNHEIMER_FD_ST_MAX_ITER

{
  echo status=running
  echo purpose=periodic_c_response_aware_matched_delta_st_q1_gate
  echo execution_server=df
  echo local_compute=no
  echo abacus_branch=codex/response-aware-product-pca
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo version_verdict=feature_branch_exception_response_aware_fixed_prefix
  echo selected_orbital_sha256=$orbital_sha
  echo nbands=$expected_ao_count
  echo fixed_nu=$fixed_nu
  echo expected_naux=$expected_naux
  echo frequency_grid_source=response_aware_selected_sos
  echo frequency_grid_sha256=$frequency_sha
  echo full_q_index=$iq
  echo nfreq=6
  echo pca_threshold=1e-4
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo grid=24,24,24
  echo kmesh=4,4,4
  echo mpi_layout=frequency_grouped
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$selection" "$frequency_manifest" "$sos_summary" INPUT STRU KPT C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_selected_product_pca.orb "$frequency_name"
} > RUN_PROVENANCE.txt

/usr/bin/time -v -o abacus.time mpirun -np "$SLURM_NTASKS" -ppn "$SLURM_NTASKS_PER_NODE" "$abacus" > abacus.out 2>&1

report=OUT.${suffix}/STERNHEIMER_CHI0.dat
test -s "$report"
grep -qx 'status success' "$report"
grep -qx 'format v1_partial' "$report"
grep -qx "sternheimer_q_index $iq" "$report"
grep -qx 'nfreq 6' "$report"
grep -qx "frequency_grid_file $frequency_name" "$report"
grep -qx 'abfs_source product_pca' "$report"
grep -qx 'perturbation_coulomb_kernel full_periodic_poisson' "$report"
grep -qx 'sternheimer_mode delta' "$report"
grep -qx 'sternheimer_fd_order 8' "$report"
grep -qx 'sternheimer_nested_k_frequency_mpi yes' "$report"
grep -qx 'sternheimer_kpoint_groups 8' "$report"
grep -qx 'sternheimer_frequency_groups 6' "$report"
grep -qx 'mpi_ranks 48' "$report"
grep -qx 'all_converged yes' "$report"
test -s v1_sternheimer_partial_manifest_iq_${iq}.dat
test -s v1_sternheimer_symmetry_routes_iq_${iq}.dat
test -s v1_sternheimer_qstar_routes_iq_${iq}.dat
test -s v1_sternheimer_full_kpoints.dat
channels=$(awk '$1=="abfs_channels" {print $2}' "$report")
test -n "$channels"
"$python" - "$grid_q/GRID_MATRIX_HEADER.json" "$channels" "$report" <<'PY'
import json
import math
import pathlib
import sys
header = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
expected_naux = 522
channels = int(sys.argv[2])
report = pathlib.Path(sys.argv[3]).read_text(encoding="ascii").splitlines()
assert header["naux"] == expected_naux == channels
residual = next(float(line.split()[1]) for line in report if line.split()[:1] == ["max_solver_relative_residual"])
assert math.isfinite(residual) and residual <= 1.01e-6
PY
cmp -s basis_aux_out "$grid_q/basis_aux_out"
cmp -s basis_out "$grid_q/basis_out"
cmp -s basis_wfc_out "$grid_q/basis_wfc_out"

{
  echo status=success
  echo abfs_channels=$channels
  echo grid_coulomb_file=$grid_matrix
  echo canonical_q_indices=$(awk '$1=="sternheimer_canonical_q_indices" {for(i=2;i<=NF;i++) printf "%s%s", (i==2?"":","), $i; print ""}' "$report")
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$report" "$grid_matrix" basis_aux_out basis_out basis_wfc_out v1_sternheimer_partial_manifest_iq_${iq}.dat v1_sternheimer_symmetry_routes_iq_${iq}.dat v1_sternheimer_qstar_routes_iq_${iq}.dat v1_sternheimer_full_kpoints.dat
} >> RUN_PROVENANCE.txt

echo "C24_RESPONSE_PCA_MATCHED_DELTA_Q1_OK work=$work"
