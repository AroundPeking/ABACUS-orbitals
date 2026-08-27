#!/bin/bash
#SBATCH --job-name=c24_highl_metric
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-highl-metric-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-highl-metric-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_HIGH_L_METRIC_CONTROL_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source=$campaign/runs/product-pca-20260825/grid-coulomb-selected-fixed-prefix-full-bz-ad29464fd/q1
grid_matrix=$source/v1_coulomb_grid_iq_1_rank0.dat
abacus_root=/data/home/df_iopcas_ghj/app/abacus/abacus-master-ghj-dbb767-20260826
abacus=$abacus_root/build_ifx/abacus_3p
abacus_commit=dbb7671896910155bd9b4983ce72ca7caa451f19
expected_abacus_sha=d05b3088acd5ec170f62027c35de69c22205dba82ac6799eb4afe877c8943708
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$campaign/runs/product-pca-20260825/highl-ewald-metric-control-dbb767189-compat1
suffix=C_DIAMOND_SELECTED_HIGH_L_METRIC_CONTROL

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$abacus"
test -x "$python"
test -s "$grid_matrix"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
grep -Fq "#define COMMIT \"$abacus_commit " "$abacus_root/build_ifx/commit.h"
for asset in INPUT STRU KPT C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_selected_product_pca.orb; do
  test -s "$source/$asset"
done

mkdir -p "$work"
cp "$source/INPUT" "$source/STRU" "$source/KPT" \
  "$source/C_ONCV_PBE-1.0.upf" \
  "$source/C_gga_10au_100Ry_selected_product_pca.orb" "$work/"
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
  local key=$1
  sed -i -E "/^[[:space:]]*${key}([[:space:]]+|$)/d" INPUT
}

set_input_key suffix "$suffix"
set_input_key symmetry -1
set_input_key nbands 112
set_input_key kpar 1
set_input_key rpa 1
set_input_key out_librpa_reader_version 1
set_input_key out_sternheimer_librpa 0
remove_input_key out_sternheimer_basis_opt
set_input_key exx_pca_threshold 1e-4
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
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_selected_candidate_high_l_ewald_metric_control
  echo changed_variable=abacus_high_l_ewald_implementation
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo reference_grid_coulomb=$grid_matrix
  echo orbital_basis=selected_fixed_prefix_joint_three_g
  echo nbands=112
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo grid=24,24,24
  echo kmesh=4,4,4
  echo symmetry=-1
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$grid_matrix" INPUT STRU KPT \
    C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_selected_product_pca.orb
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np 1 "$abacus" > abacus.out 2>&1

native_matrix=v1_coulomb_full_iq_1_rank0.dat
cs_file=v1_Cs_data_0.txt
test -s "OUT.${suffix}/running_scf.log"
grep -q '!FINAL_ETOT_IS' "OUT.${suffix}/running_scf.log"
test -s "$native_matrix"
test -s "$cs_file"
for asset in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out; do
  test -s "$asset"
done

export MKL_NUM_THREADS=8
"$python" - "$native_matrix" "$grid_matrix" "$cs_file" > HIGH_L_METRIC_CONTROL.json <<'PY'
import json
import math
import os
import struct
import sys

import numpy as np


COULOMB_MARKER = -20129433
CS_MARKER = -10267453


def atom_pair(pair_index, natoms):
    local = pair_index
    for iatom in range(natoms):
        count = natoms - iatom
        if local < count:
            return iatom, iatom + local
        local -= count
    raise ValueError("invalid atom-pair index")


def read_coulomb(path):
    with open(path, "rb") as handle:
        marker, iq, naux, value_flag, natoms, nblocks = struct.unpack("=6i", handle.read(24))
        if marker != COULOMB_MARKER or value_flag != 1:
            raise ValueError("invalid Coulomb v1 header")
        atom_naux = struct.unpack(f"={natoms}i", handle.read(4 * natoms))
        table = [struct.unpack("=iq", handle.read(12)) for _ in range(nblocks)]
        offsets = np.cumsum((0, *atom_naux))
        matrix = np.zeros((naux, naux), dtype=np.complex128)
        for pair_index, offset in table:
            iatom, jatom = atom_pair(pair_index, natoms)
            shape = (atom_naux[iatom], atom_naux[jatom])
            handle.seek(offset)
            block = np.frombuffer(
                handle.read(16 * shape[0] * shape[1]), dtype=np.complex128
            ).reshape(shape)
            i0, i1 = offsets[iatom], offsets[iatom + 1]
            j0, j1 = offsets[jatom], offsets[jatom + 1]
            matrix[i0:i1, j0:j1] = block
            if iatom != jatom:
                matrix[j0:j1, i0:i1] = block.conj().T
    return iq, atom_naux, matrix


def cs_header(path):
    with open(path, "rb") as handle:
        marker, natoms, ncells = struct.unpack("=3i", handle.read(12))
        nblocks, reserved = struct.unpack("=2q", handle.read(16))
        if marker != CS_MARKER or nblocks <= 0 or reserved < nblocks:
            raise ValueError("invalid Cs v1 header")
        max_abs = []
        for index in range(reserved):
            raw = handle.read(36)
            value, _ = struct.unpack("=dq", raw[20:])
            if index < nblocks:
                max_abs.append(value)
    return {
        "natoms": natoms,
        "ncells": ncells,
        "nblocks": nblocks,
        "reserved_blocks": reserved,
        "size_bytes": os.path.getsize(path),
        "max_abs_min": min(max_abs),
        "max_abs_max": max(max_abs),
    }


native_iq, native_dims, native_raw = read_coulomb(sys.argv[1])
grid_iq, grid_dims, grid_raw = read_coulomb(sys.argv[2])
if native_iq != grid_iq or native_dims != grid_dims:
    raise ValueError("native and grid Coulomb metadata differ")

native_hermiticity = np.linalg.norm(native_raw - native_raw.conj().T) / max(
    np.linalg.norm(native_raw), 1.0
)
grid_hermiticity = np.linalg.norm(grid_raw - grid_raw.conj().T) / max(
    np.linalg.norm(grid_raw), 1.0
)
native = 0.5 * (native_raw + native_raw.conj().T)
grid = 0.5 * (grid_raw + grid_raw.conj().T)
native_eigenvalues = np.linalg.eigvalsh(native)
grid_eigenvalues = np.linalg.eigvalsh(grid)
native_norm = np.linalg.norm(native)
grid_norm = np.linalg.norm(grid)
relative_difference = np.linalg.norm(native - grid) / max(grid_norm, 1.0)

result = {
    "status": "success",
    "definition": "one-variable high-l Ewald implementation control",
    "iq": native_iq,
    "atom_auxiliary_dimensions": list(native_dims),
    "native_hermiticity_relative": float(native_hermiticity),
    "grid_hermiticity_relative": float(grid_hermiticity),
    "native_frobenius": float(native_norm),
    "grid_frobenius": float(grid_norm),
    "native_minus_grid_relative_to_grid": float(relative_difference),
    "native_eigenvalue_min": float(native_eigenvalues[0]),
    "native_eigenvalue_max": float(native_eigenvalues[-1]),
    "grid_eigenvalue_min": float(grid_eigenvalues[0]),
    "grid_eigenvalue_max": float(grid_eigenvalues[-1]),
    "native_negative_eigenvalue_count_below_minus_1e_8": int(
        np.count_nonzero(native_eigenvalues < -1.0e-8)
    ),
    "cs": cs_header(sys.argv[3]),
}
if not all(
    math.isfinite(value)
    for key, value in result.items()
    if isinstance(value, float)
):
    raise ValueError("non-finite high-l metric diagnostic")
print(json.dumps(result, indent=2, sort_keys=True))
PY

grep -q '"status": "success"' HIGH_L_METRIC_CONTROL.json
{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$native_matrix" "$cs_file" HIGH_L_METRIC_CONTROL.json band_out basis_aux_out \
    basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out
} >> provenance.txt

echo "C24_HIGH_L_METRIC_CONTROL_OK work=$work"
