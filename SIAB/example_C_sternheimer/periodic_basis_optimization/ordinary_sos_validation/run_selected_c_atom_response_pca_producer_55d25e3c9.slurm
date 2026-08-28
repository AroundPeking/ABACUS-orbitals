#!/bin/bash
#SBATCH --job-name=c_atom_selected_pca
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-selected-pca-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-selected-pca-%j.err

set -euo pipefail
trap 'status=$?; echo "C_SELECTED_ATOM_PRODUCER_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
base=$campaign/runs/product-pca-20260825
source=/data/home/df_iopcas_ghj/gw/260823/c-atom-siab-reference-fixedocc-p1-8cf890e8
selection_root=$base/selected-fixed-prefix-checkpoint-500
selection=$selection_root/SELECTED_CANDIDATE.json
orbital=$selection_root/C_gga_10au_100Ry_selected_product_pca.orb
build_root=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534
abacus=$build_root/abacus_3p
abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
expected_orbital_sha=3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1
expected_naux=261
fixed_nu=2,2,1,0,0
root=$base/selected-c-atom-response-aware-product-pca-55d25e3c9
work=$root/producer
suffix=C_ATOM_SELECTED_RESPONSE_PCA

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test -s "$selection" && test -s "$orbital"
test "$(sha256sum "$orbital" | awk '{print $1}')" = "$expected_orbital_sha"
python3 - "$selection" "$expected_orbital_sha" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert payload["label"] == "joint-three-g"
assert payload["nu"] == [3, 3, 2, 1, 3]
assert payload["ao_count_cell"] == 112
assert payload["exported_orbital_sha256"] == sys.argv[2]
assert payload["minimum_occupied_capture"] >= payload["occupied_capture_floor"]
PY
for asset in INPUT STRU KPT C_ONCV_PBE-1.0.upf; do
  test -s "$source/$asset"
done

mkdir -p "$work"
cp "$source/INPUT" "$source/STRU" "$source/KPT" "$source/C_ONCV_PBE-1.0.upf" "$orbital" "$work/"
cd "$work"

set_input_key() {
  local key=$1 value=$2
  if awk -v key="$key" '$1 == key {found=1} END {exit !found}' INPUT; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]+.*|${key} ${value}|" INPUT
  else
    printf '%-36s %s\n' "$key" "$value" >> INPUT
  fi
}

remove_input_key() {
  local key=$1 temporary
  temporary=$(mktemp)
  awk -v key="$key" '$1 != key' INPUT > "$temporary"
  mv "$temporary" INPUT
}

remove_stru_abfs_section() {
  python3 - STRU <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
sections = {
    "ATOMIC_SPECIES", "NUMERICAL_ORBITAL", "ABFS_ORBITAL",
    "LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS",
}
lines = path.read_text(encoding="ascii").splitlines()
output = []
skip = False
for line in lines:
    stripped = line.strip()
    if stripped == "ABFS_ORBITAL":
        skip = True
        continue
    if skip and stripped in sections:
        skip = False
    if not skip:
        output.append(line)
path.write_text("\n".join(output).rstrip() + "\n", encoding="ascii")
PY
}

sed -i -E 's|^C_gga_10au_100Ry_3s3p2d\.orb$|C_gga_10au_100Ry_selected_product_pca.orb|' STRU
remove_stru_abfs_section
grep -qx 'C_gga_10au_100Ry_selected_product_pca.orb' STRU
! grep -q '^ABFS_ORBITAL$' STRU
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
set_input_key out_sternheimer_librpa 0
set_input_key out_sternheimer_basis_opt 1
remove_input_key out_sternheimer_siab
set_input_key sternheimer_q_index 1
set_input_key sternheimer_nfreq 6
set_input_key sternheimer_frequency_mpi 0
set_input_key sternheimer_channel_mpi 0
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
export I_MPI_FABRICS=shm:ofi
export ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1

{
  echo status=running
  echo purpose=selected_c_atom_zero_order_and_response_aware_product_pca
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo selected_orbital_sha256=$expected_orbital_sha
  echo spin_state=fixed_triplet_nup3_ndown1
  echo nbands_per_spin=56
  echo fixed_nu=$fixed_nu
  echo expected_naux=$expected_naux
  echo pca_threshold=1e-4
  echo grid=135,135,135
  echo fd_order=8
  echo cell_angstrom=20,20,20
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$selection" INPUT STRU KPT C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_selected_product_pca.orb
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np 1 "$abacus" > abacus.out 2>&1

status_file=OUT.${suffix}/STERNHEIMER_SIAB_STATUS.dat
matrix=v1_coulomb_grid_iq_1_rank0.dat
eig_occ=OUT.${suffix}/eig_occ.txt
test -s "$status_file" && test -s "$matrix" && test -s "$eig_occ"
grep -qx 'status abfs_diag_only' "$status_file"
grep -qx 'grid_coulomb_v1_kernel exact_rhs_full_periodic_poisson' "$status_file"
grep -Fq "Response-aware RPA product PCA fixed radial profiles: $fixed_nu" abacus.out

python3 - "$matrix" "$expected_naux" > GRID_MATRIX_HEADER.json <<'PY'
import json
import pathlib
import struct
import sys
path = pathlib.Path(sys.argv[1])
expected_naux = int(sys.argv[2])
with path.open("rb") as handle:
    marker, iq, naux, value_flag, natoms, nblocks = struct.unpack("<6i", handle.read(24))
    atom_dimensions = list(struct.unpack(f"<{natoms}i", handle.read(4 * natoms)))
header = {
    "status": "success", "marker": marker, "iq": iq, "naux": naux,
    "value_flag": value_flag, "natoms": natoms, "nblocks": nblocks,
    "atom_auxiliary_dimensions": atom_dimensions,
}
assert marker == -20129433
assert iq == 1 and value_flag == 1
assert header["natoms"] == 1
assert nblocks == 1
assert header["naux"] == expected_naux
assert atom_dimensions == [expected_naux]
print(json.dumps(header, indent=2, sort_keys=True))
PY

python3 - "$eig_occ" <<'PY'
import math
import pathlib
import re
import sys
states = {1: [], 2: []}
spin = None
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"spin=(\d+)", line)
    if match:
        spin = int(match.group(1))
        continue
    fields = line.split()
    if spin in states and len(fields) == 3 and fields[0].isdigit():
        states[spin].append(float(fields[2]))
assert all(len(values) == 56 for values in states.values()), {key: len(value) for key, value in states.items()}
assert states[1][:3] == [1.0, 1.0, 1.0] and all(abs(value) <= 1.0e-12 for value in states[1][3:])
assert states[2][0] == 1.0 and all(abs(value) <= 1.0e-12 for value in states[2][1:])
assert math.isclose(sum(states[1]), 3.0) and math.isclose(sum(states[2]), 1.0)
PY

reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' abacus.out)
test -n "$reference_ha"
python3 - "$reference_ha" <<'PY'
import math
import sys
assert math.isfinite(float(sys.argv[1]))
PY
cat > ATOM_REFERENCE.txt <<EOF
status success
side atom
method zero_order
selected_orbital_sha256 $expected_orbital_sha
naux $expected_naux
reference_ha $reference_ha
EOF

for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat KS_eigenvector_1.dat stru_out v1_Cs_data_0.txt; do
  test -s "$name"
done
{
  echo status=success
  echo reference_ha=$reference_ha
  echo naux=$expected_naux
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$matrix" "$status_file" GRID_MATRIX_HEADER.json ATOM_REFERENCE.txt band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat KS_eigenvector_1.dat stru_out v1_Cs_data_0.txt
} >> provenance.txt

echo "C_SELECTED_ATOM_PRODUCER_OK work=$work reference_ha=$reference_ha"
