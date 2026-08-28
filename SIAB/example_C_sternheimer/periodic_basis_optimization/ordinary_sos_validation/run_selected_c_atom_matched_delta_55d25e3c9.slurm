#!/bin/bash
#SBATCH --job-name=c_atom_selected_delta
#SBATCH --partition=p1
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-selected-delta-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-selected-delta-%j.err

set -euo pipefail
trap 'status=$?; echo "C_SELECTED_ATOM_DELTA_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
base=$campaign/runs/product-pca-20260825
atom_root=$base/selected-c-atom-response-aware-product-pca-55d25e3c9
producer=$atom_root/producer-retry1
sos=$atom_root/sos-nfreq6-d4810f73
selection_root=$base/selected-fixed-prefix-checkpoint-500
selection=$selection_root/SELECTED_CANDIDATE.json
orbital=$selection_root/C_gga_10au_100Ry_selected_product_pca.orb
frequency_source=$sos/ATOM_SOS_FREQUENCY_GRID.dat
frequency_manifest=$sos/ATOM_SOS_FREQUENCY_GRID.json
frequency_name=ATOM_SOS_FREQUENCY_GRID.dat
abacus=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534/abacus_3p
abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
expected_orbital_sha=3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1
expected_naux=261
fixed_nu=2,2,1,0,0
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$atom_root/matched-delta-nfreq6-55d25e3c9
suffix=C_ATOM_SELECTED_MATCHED_DELTA

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 16
test "${SLURM_NTASKS_PER_NODE:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -x "$abacus" && test -x "$python"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test "$(sha256sum "$orbital" | awk '{print $1}')" = "$expected_orbital_sha"
test -s "$frequency_source" && test -s "$frequency_manifest"
test -s "$sos/RESULT_SUMMARY.txt" && test -s "$producer/ATOM_REFERENCE.txt"
grep -qx 'status success' "$sos/RESULT_SUMMARY.txt"
grep -qx 'side atom' "$sos/RESULT_SUMMARY.txt"
grep -qx 'method sos' "$sos/RESULT_SUMMARY.txt"
frequency_sha=$(sha256sum "$frequency_source" | awk '{print $1}')
grep -qx "frequency_grid_sha256 $frequency_sha" "$sos/RESULT_SUMMARY.txt"
"$python" - "$frequency_manifest" "$frequency_sha" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert payload["status"] == "success" and payload["nfreq"] == 6
assert payload["grid_sha256"] == sys.argv[2]
PY

mkdir -p "$work"
cp "$producer/INPUT" "$producer/STRU" "$producer/KPT" "$producer/C_ONCV_PBE-1.0.upf" "$orbital" "$frequency_source" "$work/"
cd "$work"

set_input_key() {
  local key=$1 value=$2
  if awk -v key="$key" '$1 == key {found=1} END {exit !found}' INPUT; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]+.*|${key} ${value}|" INPUT
  else
    printf '%-36s %s\n' "$key" "$value" >> INPUT
  fi
}

set_input_key suffix "$suffix"
set_input_key calculation scf
set_input_key nelec 4
set_input_key nspin 2
set_input_key nupdown 2
set_input_key nbands 56
set_input_key symmetry 0
set_input_key gamma_only 1
set_input_key kpar 1
set_input_key smearing_method fixed
set_input_key ocp 1
set_input_key ocp_set "3*1 53*0 1*1 55*0"
set_input_key init_wfc atomic
set_input_key init_chg atomic
set_input_key rpa 1
set_input_key out_librpa_reader_version 1
set_input_key out_sternheimer_librpa 1
set_input_key out_sternheimer_basis_opt 0
set_input_key sternheimer_q_index 1
set_input_key sternheimer_nfreq 6
set_input_key sternheimer_frequency_grid_file "$frequency_name"
set_input_key sternheimer_frequency_mpi 1
set_input_key sternheimer_channel_mpi 1
set_input_key sternheimer_mpi_layout global_equation
set_input_key sternheimer_delta 1
set_input_key sternheimer_fd_order 8
set_input_key exx_pca_threshold 1e-4
set_input_key rpa_pca_fixed_nu "$fixed_nu"
set_input_key exx_singularity_correction massidda
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 1
set_input_key nx 135
set_input_key ny 135
set_input_key nz 135

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
unset ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL ABACUS_STERNHEIMER_FD_ST_MAX_ITER

{
  echo status=running
  echo purpose=selected_c_atom_matched_delta_st_body
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo selected_orbital_sha256=$expected_orbital_sha
  echo spin_state=fixed_triplet_nup3_ndown1
  echo nbands_per_spin=56
  echo fixed_nu=$fixed_nu
  echo expected_naux=$expected_naux
  echo frequency_grid_source=atom_sos_exact
  echo frequency_grid_sha256=$frequency_sha
  echo nfreq=6
  echo pca_threshold=1e-4
  echo grid=135,135,135
  echo fd_order=8
  echo mpi_layout=global_equation
  echo coulomb_kernel=full_periodic_poisson
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$selection" "$frequency_manifest" "$sos/RESULT_SUMMARY.txt" INPUT STRU KPT C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_selected_product_pca.orb "$frequency_name"
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np "$SLURM_NTASKS" -ppn "$SLURM_NTASKS_PER_NODE" "$abacus" > abacus.out 2>&1

report=OUT.${suffix}/STERNHEIMER_CHI0.dat
test -s "$report"
grep -qx 'status success' "$report"
grep -qx 'format v1_partial' "$report"
grep -qx 'sternheimer_q_index 1' "$report"
grep -qx 'nfreq 6' "$report"
grep -qx "frequency_grid_file $frequency_name" "$report"
grep -qx 'abfs_source product_pca' "$report"
grep -qx 'perturbation_coulomb_kernel full_periodic_poisson' "$report"
grep -qx 'sternheimer_mode delta' "$report"
grep -qx 'sternheimer_fd_order 8' "$report"
grep -qx 'all_converged yes' "$report"
test -s v1_sternheimer_partial_manifest_iq_1.dat
test -s v1_sternheimer_symmetry_routes_iq_1.dat
test -s v1_sternheimer_qstar_routes_iq_1.dat
test -s v1_sternheimer_full_kpoints.dat
channels=$(awk '$1=="abfs_channels" {print $2}' "$report")
test "$channels" = "$expected_naux"
"$python" - "$report" <<'PY'
import math
import pathlib
import sys
values = {}
for line in pathlib.Path(sys.argv[1]).read_text(encoding="ascii").splitlines():
    fields = line.split()
    if len(fields) == 2:
        values[fields[0]] = fields[1]
residual = float(values["max_solver_relative_residual"])
assert math.isfinite(residual) and residual <= 1.01e-6, residual
assert values["all_converged"] == "yes"
PY
cmp -s basis_aux_out "$producer/basis_aux_out"
cmp -s basis_out "$producer/basis_out"
cmp -s basis_wfc_out "$producer/basis_wfc_out"
cmp -s v1_coulomb_grid_iq_1_rank0.dat "$producer/v1_coulomb_grid_iq_1_rank0.dat"

reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' abacus.out)
producer_reference=$(awk '$1=="reference_ha" {print $2}' "$producer/ATOM_REFERENCE.txt")
"$python" - "$reference_ha" "$producer_reference" <<'PY'
import math
import sys
left, right = map(float, sys.argv[1:])
assert math.isfinite(left) and math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-10), (left, right)
PY

{
  echo status=success
  echo reference_ha=$reference_ha
  echo abfs_channels=$channels
  echo frequency_grid_sha256=$frequency_sha
  echo all_converged=yes
  echo max_relative_residual=$(awk '$1=="max_solver_relative_residual" {print $2}' "$report")
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$report" basis_aux_out basis_out basis_wfc_out v1_coulomb_grid_iq_1_rank0.dat v1_sternheimer_partial_manifest_iq_1.dat v1_sternheimer_symmetry_routes_iq_1.dat v1_sternheimer_qstar_routes_iq_1.dat v1_sternheimer_full_kpoints.dat
} >> provenance.txt

echo "C_SELECTED_ATOM_DELTA_OK work=$work reference_ha=$reference_ha"
