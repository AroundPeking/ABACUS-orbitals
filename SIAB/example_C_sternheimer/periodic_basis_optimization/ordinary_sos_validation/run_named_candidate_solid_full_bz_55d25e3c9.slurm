#!/bin/bash
#SBATCH --job-name=c2g_solid_q
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --array=1-64%8
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c2g-solid-q-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c2g-solid-q-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C_NAMED_SOLID_Q_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
manifest=$candidate_root/CANDIDATE.json
orbital_name=C_gga_10au_100Ry_joint_two_g.orb
orbital=$candidate_root/$orbital_name
source=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1
build_root=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534
abacus=$build_root/abacus_3p
abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
ordinal=${SLURM_ARRAY_TASK_ID:?missing full-BZ q ordinal}
work=$run_root/solid-grid/q${ordinal}
suffix=C_DIAMOND_NAMED_CANDIDATE_Q${ordinal}

test "${SLURM_JOB_PARTITION:?}" = p1 && test "${SLURM_NTASKS:?}" = 1 && test "${SLURM_CPUS_PER_TASK:?}" = 40
test "$ordinal" -ge 1 && test "$ordinal" -le 64
test ! -e "$work"
test -x "$abacus" && test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test -s "$manifest" && test -s "$orbital"
mapfile -t candidate < <(python3 - "$manifest" "$orbital" <<'PY'
import hashlib, json, pathlib, sys
payload = json.load(open(sys.argv[1], encoding="ascii"))
orbital = pathlib.Path(sys.argv[2])
assert payload["status"] == "success" and payload["pre_sos_gate"] == "pass"
assert payload["label"] == "joint-two-g" and payload["nu"] == [3, 3, 2, 1, 2]
assert payload["ao_count_cell"] == 94 and payload["ao_count_atom"] == 47
assert payload["fixed_nu"] == [2, 2, 1, 0, 0]
assert hashlib.sha256(orbital.read_bytes()).hexdigest() == payload["exported_orbital_sha256"]
print(payload["ao_count_cell"])
print(",".join(map(str, payload["fixed_nu"])))
print(payload["exported_orbital_sha256"])
PY
)
ao_count_cell=${candidate[0]}
fixed_nu=${candidate[1]}
orbital_sha=${candidate[2]}
for asset in INPUT STRU KPT C_ONCV_PBE-1.0.upf; do test -s "$source/$asset"; done

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
sed -i -E "s|^C_gga_10au_100Ry_3s3p2d\.orb$|$orbital_name|" STRU
grep -qx "$orbital_name" STRU
set_input_key suffix "$suffix"
set_input_key symmetry -1
set_input_key nbands "$ao_count_cell"
set_input_key kpar 1
set_input_key rpa 1
set_input_key out_librpa_reader_version 1
set_input_key out_sternheimer_librpa 0
set_input_key out_sternheimer_basis_opt 1
set_input_key sternheimer_q_index "$ordinal"
set_input_key sternheimer_frequency_mpi 0
set_input_key sternheimer_channel_mpi 0
set_input_key exx_pca_threshold 1e-4
set_input_key rpa_pca_fixed_nu "$fixed_nu"
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
export ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1
{
  echo status=running
  echo purpose=named_candidate_c_solid_all_band_sos_exact_grid_full_bz
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo selected_orbital_sha256=$orbital_sha
  echo ao_count_cell=$ao_count_cell
  echo fixed_nu=$fixed_nu
  echo full_q_ordinal=$ordinal
  echo full_q_index=$ordinal
  echo pca_threshold=1e-4
  echo grid=24,24,24
  echo kmesh=4,4,4
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$manifest" INPUT STRU KPT C_ONCV_PBE-1.0.upf "$orbital_name"
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np 1 "$abacus" > abacus.out 2>&1
status_file=OUT.${suffix}/STERNHEIMER_SIAB_STATUS.dat
matrix=v1_coulomb_grid_iq_${ordinal}_rank0.dat
test -s "$status_file" && test -s "$matrix" && test -s bz_sampling_out
grep -Fq "Response-aware RPA product PCA fixed radial profiles: $fixed_nu" abacus.out
grep -qx 'status abfs_diag_only' "$status_file"
grep -qx "sternheimer_q_index $ordinal" "$status_file"
grep -qx "grid_coulomb_v1_file $matrix" "$status_file"
grep -qx "grid_coulomb_v1_librpa_iq $ordinal" "$status_file"
grep -qx "grid_coulomb_v1_full_q_index $ordinal" "$status_file"
grep -qx 'grid_coulomb_v1_kernel exact_rhs_full_periodic_poisson' "$status_file"
python3 - "$matrix" "$ordinal" bz_sampling_out > GRID_MATRIX_HEADER.json <<'PY'
import json, math, struct, sys
with open(sys.argv[1], "rb") as handle:
    marker, iq, naux, value_flag, natoms, nblocks, atom0, atom1 = struct.unpack("<8i", handle.read(32))
assert marker == -20129433 and iq == int(sys.argv[2]) and value_flag == 1
assert natoms == 2 and nblocks == 3 and atom0 == atom1 and atom0 + atom1 == naux and naux > 0
lines = [line.split() for line in open(sys.argv[3], encoding="ascii") if line.strip()]
assert [int(value) for value in lines[0]] == [4, 4, 4]
assert [int(value) for value in lines[1]] == [64, 64]
assert len(lines[2:]) == 64 and math.isclose(sum(float(row[1]) for row in lines[2:]), 1.0, abs_tol=1e-12)
print(json.dumps({"status": "success", "iq": iq, "naux": naux, "atom_auxiliary_dimensions": [atom0, atom1]}, indent=2, sort_keys=True))
PY
{
  echo status=success
  echo grid_coulomb_file=$matrix
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$matrix" "$status_file" GRID_MATRIX_HEADER.json band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out
} >> provenance.txt
echo "C_NAMED_SOLID_Q_OK ordinal=$ordinal work=$work"
