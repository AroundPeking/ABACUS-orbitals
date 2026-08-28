#!/bin/bash
#SBATCH --job-name=c2g_atom_restart
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c2g-atom-restart-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c2g-atom-restart-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NAMED_ATOM_RESTART_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
manifest=$candidate_root/CANDIDATE.json
producer=$run_root/atom-producer
initial_snapshot=$run_root/atom-producer-initial-nonconverged-snapshot
recovery=$run_root/atom-producer-restart-recovery-55d25e3c9
suffix=C_ATOM_NAMED_CANDIDATE_PCA_RESTART
initial_out=$producer/OUT.C_ATOM_NAMED_CANDIDATE_PCA
out=$recovery/OUT.$suffix
orbital_name=C_gga_10au_100Ry_joint_two_g.orb
abacus=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534/abacus_3p
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test -d "$producer" && test ! -L "$producer"
test ! -e "$initial_snapshot" && test ! -e "$recovery"
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test -s "$manifest"
grep -qx 'status=running' "$producer/provenance.txt"
grep -q '!!SCF IS NOT CONVERGED!!' "$initial_out/running_scf.log"
! grep -q '#SCF IS CONVERGED#' "$initial_out/running_scf.log"

mkdir -p "$out" "$recovery/restart_input_snapshot"
cp "$producer/INPUT" "$producer/STRU" "$producer/KPT" \
  "$producer/C_ONCV_PBE-1.0.upf" "$producer/$orbital_name" "$recovery/"
for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do
  test -s "$initial_out/$name"
  cp "$initial_out/$name" "$out/$name"
  cp "$initial_out/$name" "$recovery/restart_input_snapshot/$name"
done
cd "$recovery"

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
set_input_key nbands 47
set_input_key smearing_method fixed
set_input_key ocp 1
set_input_key ocp_set "3*1 44*0 1*1 46*0"
set_input_key init_wfc file
set_input_key init_chg file
set_input_key mixing_type broyden
set_input_key mixing_beta 0.3
set_input_key mixing_beta_mag 0.3
set_input_key scf_thr 1e-10
set_input_key scf_nmax 300
set_input_key rpa 1

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
  echo purpose=named_candidate_c_atom_restart_from_nonconverged_density
  echo physical_change=none
  echo numerical_change=reset_broyden_history_and_load_existing_density_and_wavefunctions
  echo source_producer=$producer
  echo selected_orbital_sha256=$(sha256sum "$orbital_name" | awk '{print $1}')
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$manifest" INPUT STRU KPT C_ONCV_PBE-1.0.upf \
    "$orbital_name" restart_input_snapshot/*
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np 1 "$abacus" > abacus.out 2>&1

log=$out/running_scf.log
test -s "$log"
grep -q '#SCF IS CONVERGED#' "$log"
! grep -q '!!SCF IS NOT CONVERGED!!' "$log"
for spin in 1 2; do
  grep -Eq "Read NAO wave functions from .*wfs${spin}_nao.txt" abacus.out
  grep -Eq "Read in electron density: .*chgs${spin}\.cube|Read electron density from file: .*chgs${spin}\.cube" "$log"
done
grep -Fq 'Response-aware RPA product PCA fixed radial profiles: 2,2,1,0,0' abacus.out

matrix=v1_coulomb_full_iq_1_rank0.dat
eig_occ=$out/eig_occ.txt
test -s "$matrix" && test -s "$eig_occ"
python3 - "$matrix" > GRID_MATRIX_HEADER.json <<'PY'
import json
import struct
import sys
with open(sys.argv[1], "rb") as handle:
    marker, iq, naux, value_flag, natoms, nblocks = struct.unpack("<6i", handle.read(24))
    dimensions = list(struct.unpack(f"<{natoms}i", handle.read(4 * natoms)))
assert marker == -20129433 and iq == 1 and value_flag == 1
assert natoms == 1 and nblocks == 1 and naux > 0 and dimensions == [naux]
print(json.dumps({"status": "success", "iq": iq, "naux": naux, "natoms": natoms, "nblocks": nblocks, "atom_auxiliary_dimensions": dimensions}, indent=2, sort_keys=True))
PY
naux=$(python3 -c 'import json; print(json.load(open("GRID_MATRIX_HEADER.json"))["naux"])')

python3 - "$eig_occ" <<'PY'
import math
import pathlib
import re
import sys
states, spin = {1: [], 2: []}, None
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"spin=(\d+)", line)
    if match:
        spin = int(match.group(1))
        continue
    fields = line.split()
    if spin in states and len(fields) == 3 and fields[0].isdigit():
        states[spin].append(float(fields[2]))
assert all(len(values) == 47 for values in states.values())
assert states[1][:3] == [1.0, 1.0, 1.0] and all(abs(value) <= 1e-12 for value in states[1][3:])
assert states[2][0] == 1.0 and all(abs(value) <= 1e-12 for value in states[2][1:])
assert math.isclose(sum(states[1]), 3.0) and math.isclose(sum(states[2]), 1.0)
PY

reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' abacus.out)
python3 - "$reference_ha" <<'PY'
import math
import sys
assert math.isfinite(float(sys.argv[1]))
PY
orbital_sha=$(sha256sum "$orbital_name" | awk '{print $1}')
cat > ATOM_REFERENCE.txt <<EOF
status success
side atom
method zero_order_restart
selected_orbital_sha256 $orbital_sha
naux $naux
reference_ha $reference_ha
EOF
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out v1_Cs_data_0.txt; do
  test -s "$name"
done
{
  echo status=success
  echo reference_ha=$reference_ha
  echo naux=$naux
  echo restart_loaded=yes
  echo occupations=spin_1_3_spin_2_1
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$matrix" GRID_MATRIX_HEADER.json ATOM_REFERENCE.txt band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out v1_Cs_data_0.txt
} >> provenance.txt

# Preserve the failed first attempt and expose only the validated restart at the
# canonical producer path already referenced by the unique pending SOS job.
ln -s "$recovery" "$run_root/.atom-producer-recovered-link-$SLURM_JOB_ID"
mv "$producer" "$initial_snapshot"
mv "$run_root/.atom-producer-recovered-link-$SLURM_JOB_ID" "$producer"
printf 'status success\nsource %s\naccepted %s\n' "$initial_snapshot" "$recovery" > "$run_root/ATOM_PRODUCER_RECOVERY.txt"

echo "C_NAMED_ATOM_RESTART_OK work=$recovery reference_ha=$reference_ha naux=$naux"
