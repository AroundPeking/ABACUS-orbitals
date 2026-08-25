#!/bin/bash
#SBATCH --job-name=c24_selected_delta_q
#SBATCH --partition=p1
#SBATCH --nodes=48
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=1
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-selected-delta-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-selected-delta-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C24_SELECTED_MATCHED_DELTA_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1
selection_root=$campaign/runs/product-pca-20260825/selected-fixed-prefix-checkpoint-500
selection=$selection_root/SELECTED_CANDIDATE.json
orbital=$selection_root/C_gga_10au_100Ry_selected_product_pca.orb
sos_root=$campaign/runs/product-pca-20260825/ordinary-sos-selected-grid-full-bz-reader-fractional-nfreq6-d4810f73
frequency_source=$sos_root/SELECTED_SOS_FREQUENCY_GRID.dat
frequency_manifest=$sos_root/SELECTED_SOS_FREQUENCY_GRID.json
frequency_name=SELECTED_SOS_FREQUENCY_GRID.dat
grid_root=$campaign/runs/product-pca-20260825/grid-coulomb-selected-fixed-prefix-full-bz-ad29464fd
abacus=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-grid-coulomb-ad29464fd-20260825/artifacts/job3121650/abacus_3p
abacus_commit=ad29464fdc9125114cc052840415647a1b5a6ef1
expected_abacus_sha=18633e8465492e4b4fae20c9e4136ac30890192b98e1c03029bbc38b11e696c0
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
iq=${SLURM_ARRAY_TASK_ID:?missing canonical full-q index}
root=$campaign/runs/product-pca-20260825/matched-delta-selected-fixed-prefix-nfreq6-ad29464fd
work=$root/q${iq}
suffix=C_DIAMOND_SELECTED_MATCHED_DELTA_Q${iq}

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
test -s "$selection"
test -s "$orbital"
test -s "$frequency_source"
test -s "$frequency_manifest"
grep -qx 'status=success' "$sos_root/provenance.txt"
grep -qx 'frequency_grid_source=selected_candidate_sos' "$sos_root/provenance.txt"

read -r selected_g expected_ao_count orbital_sha frequency_sha < <("$python" - "$selection" "$orbital" "$frequency_source" "$frequency_manifest" <<'PY'
import hashlib
import json
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
orbital = pathlib.Path(sys.argv[2])
frequency = pathlib.Path(sys.argv[3])
frequency_manifest = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="ascii"))
expected = {
    "joint-two-g": ([3, 3, 2, 1, 2], 94, 2),
    "joint-three-g": ([3, 3, 2, 1, 3], 112, 3),
}
label = selection["label"]
assert label in expected
nu, ao_count, selected_g = expected[label]
assert selection["nu"] == nu
assert selection["ao_count_cell"] == ao_count
assert selection["minimum_occupied_capture"] >= selection["occupied_capture_floor"]
orbital_sha = hashlib.sha256(orbital.read_bytes()).hexdigest()
frequency_sha = hashlib.sha256(frequency.read_bytes()).hexdigest()
assert orbital_sha == selection["exported_orbital_sha256"]
assert frequency_manifest["status"] == "success"
assert frequency_manifest["nfreq"] == 6
assert frequency_sha == frequency_manifest["frequency_grid_sha256"]
print(selected_g, ao_count, orbital_sha, frequency_sha)
PY
)

grep -qx 'Energy Cutoff(Ry)           100.0' "$orbital"
grep -qx 'Radius Cutoff(a.u.)         10.0' "$orbital"
grep -qx 'Lmax                        4' "$orbital"
grep -qx "Number of Gorbital-->       $selected_g" "$orbital"
for asset in INPUT STRU KPT C_ONCV_PBE-1.0.upf; do
  test -s "$source/$asset"
done
grid_q=$grid_root/q${iq}
grid_matrix=$grid_q/v1_coulomb_grid_iq_${iq}_rank0.dat
test -s "$grid_matrix"
test -s "$grid_q/GRID_MATRIX_HEADER.json"
grep -qx 'status=success' "$grid_q/provenance.txt"
grep -qx "full_q_index=$iq" "$grid_q/provenance.txt"

mkdir -p "$work"
cp "$source/INPUT" "$source/STRU" "$source/KPT" \
  "$source/C_ONCV_PBE-1.0.upf" "$orbital" "$frequency_source" "$work/"
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
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 2
set_input_key nx 24
set_input_key ny 24
set_input_key nz 24

export LD_LIBRARY_PATH=/data/home/df_iopcas_ghj/app/elpa-build/lib:/data/app/libxc/6.2.2-intel/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export OMP_STACKSIZE=2G
export MKL_DYNAMIC=FALSE
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export I_MPI_FABRICS=shm:ofi
export ABACUS_STERNHEIMER_CHANNEL_THREADS=$OMP_NUM_THREADS
unset ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL
unset ABACUS_STERNHEIMER_FD_ST_MAX_ITER

{
  echo status=running
  echo purpose=periodic_c_selected_candidate_matched_frequency_delta_st
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo selected_g=$selected_g
  echo nbands=$expected_ao_count
  echo selected_orbital_sha256=$orbital_sha
  echo frequency_grid_source=selected_candidate_sos
  echo frequency_grid_sha256=$frequency_sha
  echo selected_sos_job=${SELECTED_SOS_JOB_ID:?missing selected SOS LibRPA job id}
  echo grid_coulomb_array_job=${GRID_COULOMB_ARRAY_JOB_ID:?missing grid Coulomb array job id}
  echo symmetry=1
  echo full_q_index=$iq
  echo nfreq=6
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo grid=24,24,24
  echo kmesh=4,4,4
  echo mpi_layout=frequency_grouped
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$selection" "$frequency_manifest" INPUT STRU KPT \
    C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_selected_product_pca.orb "$frequency_name"
} > RUN_PROVENANCE.txt

/usr/bin/time -v -o abacus.time \
  mpirun -np "$SLURM_NTASKS" -ppn "$SLURM_NTASKS_PER_NODE" \
  "$abacus" > abacus.out 2>&1

report=OUT.${suffix}/STERNHEIMER_CHI0.dat
test -s "$report"
grep -qx 'status success' "$report"
grep -qx 'format v1_partial' "$report"
grep -qx "sternheimer_q_index $iq" "$report"
grep -qx 'grid 24 24 24 size 13824 dV 0.005693817519038907' "$report"
grep -qx 'nfreq 6' "$report"
grep -qx 'frequency_grid_source file' "$report"
grep -qx "frequency_grid_file $frequency_name" "$report"
grep -qx 'pca_threshold 0.0001' "$report"
grep -qx 'abfs_source product_pca' "$report"
grep -qx 'perturbation_coulomb_kernel full_periodic_poisson' "$report"
grep -qx 'periodic_kmesh 4 4 4' "$report"
grep -qx 'sternheimer_mode delta' "$report"
grep -qx 'sternheimer_fd_order 8' "$report"
grep -qx 'sternheimer_nested_k_frequency_mpi yes' "$report"
grep -qx 'sternheimer_kpoint_groups 8' "$report"
grep -qx 'sternheimer_frequency_groups 6' "$report"
grep -qx 'mpi_ranks 48' "$report"
grep -qx 'all_converged yes' "$report"
awk -v iq="$iq" '$1=="sternheimer_canonical_q_indices" {for(i=2;i<=NF;i++) if($i==iq) found=1} END {exit !found}' "$report"
test -s "v1_sternheimer_partial_manifest_iq_${iq}.dat"
test -s "v1_sternheimer_symmetry_routes_iq_${iq}.dat"
test -s "v1_sternheimer_qstar_routes_iq_${iq}.dat"
test -s v1_sternheimer_full_kpoints.dat
channels=$(awk '$1=="abfs_channels" {print $2}' "$report")
test -n "$channels"
"$python" - "$grid_q/GRID_MATRIX_HEADER.json" "$channels" "$report" <<'PY'
import json
import math
import pathlib
import sys

header = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
channels = int(sys.argv[2])
report = pathlib.Path(sys.argv[3]).read_text(encoding="ascii").splitlines()
assert header["status"] == "success"
assert header["naux"] == channels
residual = None
for line in report:
    fields = line.split()
    if fields and fields[0] == "max_solver_relative_residual":
        residual = float(fields[1])
assert residual is not None and math.isfinite(residual) and residual <= 1.01e-6
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
  sha256sum "$report" "$grid_matrix" basis_aux_out basis_out basis_wfc_out \
    v1_sternheimer_partial_manifest_iq_${iq}.dat \
    v1_sternheimer_symmetry_routes_iq_${iq}.dat \
    v1_sternheimer_qstar_routes_iq_${iq}.dat v1_sternheimer_full_kpoints.dat
} >> RUN_PROVENANCE.txt

echo "C24_SELECTED_MATCHED_DELTA_OK q=$iq work=$work"
