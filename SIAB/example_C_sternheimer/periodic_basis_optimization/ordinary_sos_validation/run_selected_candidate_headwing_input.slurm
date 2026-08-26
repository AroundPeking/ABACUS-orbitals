#!/bin/bash
#SBATCH --job-name=c24_selected_headwing
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-selected-headwing-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-selected-headwing-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_SELECTED_HEADWING_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
validation=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
get_diel=$validation/get_diel_selected_c_headwing.py
output_librpa=$validation/pyatb_librpa_writer_selected_c.py
source=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1
selection_root=$campaign/runs/product-pca-20260825/selected-fixed-prefix-checkpoint-500
selection=$selection_root/SELECTED_CANDIDATE.json
orbital=$selection_root/C_gga_10au_100Ry_selected_product_pca.orb
abacus=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-grid-coulomb-ad29464fd-20260825/artifacts/job3121650/abacus_3p
expected_abacus_sha=18633e8465492e4b4fae20c9e4136ac30890192b98e1c03029bbc38b11e696c0
abacus_commit=ad29464fdc9125114cc052840415647a1b5a6ef1
pyatb_python=/data/home/df_iopcas_ghj/app/oml-pinned/pyatb-env/bin/python
pyatb_source=/data/home/df_iopcas_ghj/app/oml-pinned/pyatb-enable_head_wing-9fb9028c
pyatb_commit=9fb9028c59b1dbaf9cf66965280961fc2225d9eb
root=$campaign/runs/product-pca-20260825/headwing-selected-fixed-prefix-ad29464fd
work=$root/job${SLURM_JOB_ID}
suffix=ABACUS

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test -x "$pyatb_python"
test -d "$pyatb_source"
test "$(cat "$pyatb_source/.git/HEAD")" = "$pyatb_commit"
test -s "$get_diel"
test -s "$output_librpa"
test -s "$selection"
test -s "$orbital"

read -r selected_g expected_ao_count < <(python3 - "$selection" "$orbital" <<'PY'
import hashlib
import json
import pathlib
import sys

selection = json.load(open(sys.argv[1], encoding="ascii"))
orbital = pathlib.Path(sys.argv[2])
expected = {
    "joint-two-g": ([3, 3, 2, 1, 2], 94, 2),
    "joint-three-g": ([3, 3, 2, 1, 3], 112, 3),
}
nu, ao_count, selected_g = expected[selection["label"]]
assert selection["nu"] == nu
assert selection["ao_count_cell"] == ao_count
assert selection["minimum_occupied_capture"] >= selection["occupied_capture_floor"]
assert hashlib.sha256(orbital.read_bytes()).hexdigest() == selection["exported_orbital_sha256"]
print(selected_g, ao_count)
PY
)

for asset in INPUT STRU KPT C_ONCV_PBE-1.0.upf; do
  test -s "$source/$asset"
done
mkdir -p "$work" "$root"
cp "$source/INPUT" "$source/STRU" "$source/KPT" \
  "$source/C_ONCV_PBE-1.0.upf" "$orbital" "$work/"
cp "$get_diel" "$work/get_diel.py"
cp "$output_librpa" "$work/output_librpa.py"
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
ln -s KPT KPT_scf
set_input_key suffix "$suffix"
set_input_key calculation scf
set_input_key symmetry 1
set_input_key nbands "$expected_ao_count"
set_input_key kpar 1
set_input_key rpa 0
set_input_key out_sternheimer_librpa 0
set_input_key out_sternheimer_basis_opt 0
set_input_key out_mat_hs2 1
set_input_key out_mat_r 1
set_input_key out_chg 0
set_input_key out_wfc_lcao 0
set_input_key nx 24
set_input_key ny 24
set_input_key nz 24

export LD_LIBRARY_PATH=/data/home/df_iopcas_ghj/app/elpa-build/lib:/data/app/libxc/6.2.2-intel/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export MKL_DYNAMIC=FALSE
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_selected_candidate_all_band_headwing_input
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo pyatb_commit=$pyatb_commit
  echo selected_g=$selected_g
  echo headwing_basis=$expected_ao_count
  echo nbands=$expected_ao_count
  echo kmesh=4,4,4
  echo grid=24,24,24
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$selection" "$orbital" "$get_diel" "$output_librpa" INPUT STRU KPT
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np 1 "$abacus" > abacus.out 2>&1
grep -q 'TOTAL  Time' abacus.out
for matrix in OUT.ABACUS/hrs1_nao.csr OUT.ABACUS/srs1_nao.csr OUT.ABACUS/rr.csr; do
  test -s "$matrix"
done

/usr/bin/time -v -o pyatb.time "$pyatb_python" get_diel.py > pyatb.out 2>&1
for name in velocity_matrix band_out k_path_info KS_eigenvector_0.dat; do
  test -s "pyatb_librpa_df/$name"
done
read -r nbasis nstates nspin nk < pyatb_librpa_df/k_path_info
test "$nbasis" -eq "$expected_ao_count"
test "$nstates" -eq "$expected_ao_count"
test "$nspin" -eq 1
test "$nk" -eq 64

{
  echo status=success
  echo headwing_basis=$expected_ao_count
  echo headwing_kpoints=$nk
  echo headwing_convergence_gate=pending
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/abacus_wall_clock=/p' abacus.time
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/pyatb_wall_clock=/p' pyatb.time
  sha256sum pyatb_librpa_df/velocity_matrix pyatb_librpa_df/band_out \
    pyatb_librpa_df/k_path_info pyatb_librpa_df/KS_eigenvector_0.dat
} >> provenance.txt
ln -sfn "$work" "$root/latest"
echo "C24_SELECTED_HEADWING_OK work=$work headwing_basis=$expected_ao_count"
