#!/bin/bash
#SBATCH --job-name=c_atom_restart_gate
#SBATCH --partition=p1
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-restart-gate-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-restart-gate-%j.err

set -euo pipefail
trap 'status=$?; echo "C_SELECTED_ATOM_RESTART_GATE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
base=$campaign/runs/product-pca-20260825
atom_root=$base/selected-c-atom-response-aware-product-pca-55d25e3c9
producer=$atom_root/producer-accepted
source_producer=$atom_root/producer-final
restart_source=$source_producer/OUT.C_ATOM_SELECTED_RESPONSE_PCA
orbital_name=C_gga_10au_100Ry_selected_product_pca.orb
abacus=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534/abacus_3p
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
expected_orbital_sha=3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1
work=$atom_root/restart-zero-order-mpi16-55d25e3c9
suffix=C_ATOM_SELECTED_RESTART_MPI16
out=OUT.$suffix

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 16
test "${SLURM_NTASKS_PER_NODE:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test -s "$producer/PRODUCER_ACCEPTANCE.txt"
test -s "$producer/ATOM_REFERENCE.txt"
grep -qx 'status success' "$producer/ATOM_REFERENCE.txt"
grep -qx 'status success' "$producer/PRODUCER_ACCEPTANCE.txt"
test "$(readlink -f "$producer/INPUT")" = "$source_producer/INPUT"
test "$(sha256sum "$source_producer/$orbital_name" | awk '{print $1}')" = "$expected_orbital_sha"
test -d "$restart_source"

mkdir -p "$work/$out" "$work/restart_input_snapshot"
cp "$source_producer/INPUT" "$source_producer/STRU" "$source_producer/KPT" \
  "$source_producer/C_ONCV_PBE-1.0.upf" "$source_producer/$orbital_name" "$work/"
for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do
  test -s "$restart_source/$name"
  cp "$restart_source/$name" "$work/$out/$name"
  cp "$restart_source/$name" "$work/restart_input_snapshot/$name"
done
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
set_input_key init_wfc file
set_input_key init_chg file
set_input_key scf_thr 1e-10
set_input_key scf_nmax 300
set_input_key out_chg 1
set_input_key out_wfc_lcao 1
set_input_key out_app_flag 1
set_input_key rpa 0
set_input_key out_sternheimer_librpa 0
set_input_key out_sternheimer_basis_opt 0

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

{
  echo status=running
  echo purpose=selected_c_atom_zero_order_restart_mpi16_gate
  echo source_zero_order=$source_producer
  echo restart_source=$restart_source
  echo selected_orbital_sha256=$expected_orbital_sha
  echo mpi_ranks=$SLURM_NTASKS
  echo mpi_ranks_per_node=$SLURM_NTASKS_PER_NODE
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" INPUT STRU KPT C_ONCV_PBE-1.0.upf "$orbital_name" restart_input_snapshot/*
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np "$SLURM_NTASKS" -ppn "$SLURM_NTASKS_PER_NODE" "$abacus" > abacus.out 2>&1

log=$out/running_scf.log
test -s "$log"
grep -q '#SCF IS CONVERGED#' "$log"
if grep -q '!!SCF IS NOT CONVERGED!!' "$log"; then
  echo 'restart gate reached the SCF iteration limit' >&2
  exit 1
fi
for spin in 1 2; do
  grep -Eq "Read NAO wave functions from .*wfs${spin}_nao.txt" abacus.out
  grep -Eq "Read in electron density: .*chgs${spin}\.cube|Read electron density from file: .*chgs${spin}\.cube" "$log"
done

python3 - "$source_producer/OUT.C_ATOM_SELECTED_RESPONSE_PCA/running_scf.log" "$log" "$out/eig_occ.txt" <<'PY'
import math
import pathlib
import re
import sys

def final_energy_ev(path):
    values = []
    for line in pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV", line)
        if match:
            values.append(float(match.group(1)))
    assert len(values) == 1, (path, values)
    return values[0]

reference_ev = final_energy_ev(sys.argv[1])
restart_ev = final_energy_ev(sys.argv[2])
assert math.isfinite(reference_ev) and math.isfinite(restart_ev)
assert abs(reference_ev - restart_ev) <= 1.0e-8, (reference_ev, restart_ev)

states = {1: [], 2: []}
spin = None
for line in pathlib.Path(sys.argv[3]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"spin=(\d+)", line)
    if match:
        spin = int(match.group(1))
        continue
    fields = line.split()
    if spin in states and len(fields) == 3 and fields[0].isdigit():
        states[spin].append(float(fields[2]))
assert all(len(values) == 56 for values in states.values()), {key: len(value) for key, value in states.items()}
assert states[1][:3] == [1.0, 1.0, 1.0]
assert all(abs(value) <= 1.0e-12 for value in states[1][3:])
assert states[2][0] == 1.0
assert all(abs(value) <= 1.0e-12 for value in states[2][1:])
assert math.isclose(sum(states[1]), 3.0)
assert math.isclose(sum(states[2]), 1.0)

pathlib.Path("RESTART_GATE_RESULT.txt").write_text(
    "status success\n"
    f"reference_pbe_ev {reference_ev:.15f}\n"
    f"restart_pbe_ev {restart_ev:.15f}\n"
    f"energy_difference_ev {restart_ev - reference_ev:.15e}\n"
    "occupations spin_1=3 spin_2=1\n"
    "scf_converged yes\n"
    "restart_loaded yes\n",
    encoding="ascii",
)
PY

test -s RESTART_GATE_RESULT.txt
grep -qx 'status success' RESTART_GATE_RESULT.txt
grep -qx 'scf_converged yes' RESTART_GATE_RESULT.txt
grep -qx 'restart_loaded yes' RESTART_GATE_RESULT.txt
{
  cat RESTART_GATE_RESULT.txt
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum RESTART_GATE_RESULT.txt "$out/eig_occ.txt" restart_input_snapshot/*
} >> provenance.txt

echo "C_SELECTED_ATOM_RESTART_GATE_OK work=$work"
