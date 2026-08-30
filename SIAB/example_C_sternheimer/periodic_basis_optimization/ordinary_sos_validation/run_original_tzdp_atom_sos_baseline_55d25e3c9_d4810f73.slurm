#!/bin/bash
#SBATCH --job-name=c_tzdp_atom_sos
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-tzdp-atom-sos-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-tzdp-atom-sos-%j.err

set -euo pipefail
trap 'status=$?; echo "C_TZDP_ATOM_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

run_root=${RUN_ROOT:?missing immutable run root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
source=/data/home/df_iopcas_ghj/gw/260823/c-atom-siab-reference-fixedocc-p1-8cf890e8
build_root=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534
abacus=$build_root/abacus_3p
abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
orbital_name=C_gga_10au_100Ry_3s3p2d.orb
expected_orbital_sha=7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
extractor=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/extract_librpa_frequency_grid.py
suffix=C_ATOM_ORIGINAL_TZDP_PCA

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$run_root"
test -x "$abacus" && test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test -x "$librpa" && test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -x "$python" && test -s "$extractor"
for asset in INPUT STRU KPT C_ONCV_PBE-1.0.upf "$orbital_name"; do test -s "$source/$asset"; done
test "$(sha256sum "$source/$orbital_name" | awk '{print $1}')" = "$expected_orbital_sha"

mkdir -p "$run_root"
cp "$source/INPUT" "$source/STRU" "$source/KPT" "$source/C_ONCV_PBE-1.0.upf" "$source/$orbital_name" "$run_root/"
cd "$run_root"

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
  "$python" - STRU <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sections = {
    "ATOMIC_SPECIES", "NUMERICAL_ORBITAL", "ABFS_ORBITAL",
    "LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS",
}
output = []
skip = False
for line in path.read_text(encoding="ascii").splitlines():
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

remove_stru_abfs_section
grep -qx "$orbital_name" STRU
! grep -q '^ABFS_ORBITAL$' STRU
set_input_key suffix "$suffix"
set_input_key calculation scf
set_input_key nelec 4
set_input_key nspin 2
set_input_key nupdown 2
set_input_key nbands 22
set_input_key symmetry 0
set_input_key gamma_only 1
set_input_key kpar 1
set_input_key smearing_method fixed
set_input_key ocp 1
set_input_key ocp_set "3*1 19*0 1*1 21*0"
set_input_key init_wfc atomic
set_input_key init_chg atomic
set_input_key rpa 1
set_input_key out_librpa_reader_version 1
set_input_key out_sternheimer_librpa 0
set_input_key out_sternheimer_basis_opt 0
remove_input_key out_sternheimer_siab
set_input_key sternheimer_q_index 1
set_input_key sternheimer_nfreq 6
remove_input_key sternheimer_frequency_grid_file
set_input_key sternheimer_frequency_mpi 0
set_input_key sternheimer_channel_mpi 0
set_input_key sternheimer_mpi_layout frequency_grouped
set_input_key sternheimer_delta 1
set_input_key sternheimer_fd_order 8
set_input_key exx_pca_threshold 1e-4
set_input_key rpa_pca_fixed_nu 2,2,1,0,0
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
{
  echo status=running
  echo purpose=original_tzdp_c_atom_all_band_sos_baseline
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo orbital_basis=C_TZDP_10au_3s3p2d
  echo orbital_sha256=$expected_orbital_sha
  echo pca_threshold=1e-4
  echo fixed_nu=2,2,1,0,0
  echo grid=135,135,135
  echo fd_order=8
  echo spin_state=fixed_triplet_nup3_ndown1
  echo n_bands_chi0=-1
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$abacus" "$librpa" "$build_provenance" "$extractor" INPUT STRU KPT C_ONCV_PBE-1.0.upf "$orbital_name"
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np 1 "$abacus" > abacus.out 2>&1

out=OUT.${suffix}
matrix=v1_coulomb_full_iq_1_rank0.dat
eig_occ=$out/eig_occ.txt
test -s "$matrix" && test -s "$eig_occ" && test -s "$out/running_scf.log"
grep -q '#SCF IS CONVERGED#' "$out/running_scf.log"
! grep -q '!!SCF IS NOT CONVERGED!!' "$out/running_scf.log"
grep -Fq 'Response-aware RPA product PCA fixed radial profiles: 2,2,1,0,0' abacus.out

"$python" - "$matrix" > GRID_MATRIX_HEADER.json <<'PY'
import json
import struct
import sys

with open(sys.argv[1], "rb") as handle:
    marker, iq, naux, value_flag, natoms, nblocks = struct.unpack("<6i", handle.read(24))
    dimensions = list(struct.unpack(f"<{natoms}i", handle.read(4 * natoms)))
assert marker == -20129433 and iq == 1 and value_flag == 1
assert natoms == 1 and nblocks == 1 and naux > 0 and dimensions == [naux]
print(json.dumps({"status": "success", "iq": iq, "naux": naux}, indent=2, sort_keys=True))
PY
naux=$("$python" -c 'import json; print(json.load(open("GRID_MATRIX_HEADER.json"))["naux"])')

"$python" - "$eig_occ" <<'PY'
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
assert all(len(values) == 22 for values in states.values())
assert states[1][:3] == [1.0, 1.0, 1.0] and all(abs(value) <= 1e-12 for value in states[1][3:])
assert states[2][0] == 1.0 and all(abs(value) <= 1e-12 for value in states[2][1:])
assert math.isclose(sum(states[1]), 3.0) and math.isclose(sum(states[2]), 1.0)
PY

reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' abacus.out)
"$python" - "$reference_ha" <<'PY'
import math
import sys
assert math.isfinite(float(sys.argv[1]))
PY

reader=$run_root/reader_v1
mkdir "$reader"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out v1_Cs_data_0.txt "$matrix"; do
  test -s "$run_root/$name"
  ln -s "$run_root/$name" "$reader/$name"
done
cat > librpa.in <<EOF
input_dir = $reader/
task = rpa
nfreq = 6
tfgrids_type = minimax
parallel_routing = libri
version_coul_reader = 1
version_lri_reader = 1
prefix_coul_full = v1_coulomb_full_iq_
prefix_lri_coeff = v1_Cs_data_
use_scalapack_ecrpa = t
use_shrink_abfs = f
use_shrink_chi = f
n_bands_chi0 = -1
vq_threshold = 0
sqrt_coulomb_threshold = 1e-5
cs_threshold = 0
use_fullcoul_exx = t
use_symmetry_exx = f
use_symmetry_rpa = f
use_symmetry_gw = f
use_rpa_gamma = true
replace_w_head = false
debug = f
EOF
find -L "$reader" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > INPUT_SHA256.txt

export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=close
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
grep -q 'libRPA finished successfully' librpa.out
grep -q '^| Total EcRPA:' librpa.out
"$python" "$extractor" --source librpa.out --output ATOM_SOS_FREQUENCY_GRID.dat --manifest ATOM_SOS_FREQUENCY_GRID.json --expected-nfreq 6 > frequency_extract.out
ecrpa_ha=$(awk '/^\| Total EcRPA:/ {value=$NF} END {print value}' librpa.out)
frequency_sha=$(sha256sum ATOM_SOS_FREQUENCY_GRID.dat | awk '{print $1}')
"$python" - "$reference_ha" "$ecrpa_ha" <<'PY'
import math
import sys
assert all(math.isfinite(float(value)) for value in sys.argv[1:])
PY

cat > RESULT_SUMMARY.txt <<EOF
status success
side atom
method sos
scope body_only_no_analytic_headwing
coulomb_kernel full_periodic_poisson
orbital_basis C_TZDP_10au_3s3p2d
selected_orbital_sha256 $expected_orbital_sha
frequency_grid_sha256 $frequency_sha
naux $naux
reference_ha $reference_ha
ecrpa_ha $ecrpa_ha
EOF
{
  echo status=success
  echo reference_ha=$reference_ha
  echo ecrpa_ha=$ecrpa_ha
  echo naux=$naux
  echo frequency_grid_sha256=$frequency_sha
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/abacus_wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/abacus_max_rss_kb=/p' abacus.time
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/librpa_wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/librpa_max_rss_kb=/p' librpa.time
  sha256sum "$matrix" GRID_MATRIX_HEADER.json RESULT_SUMMARY.txt INPUT_SHA256.txt ATOM_SOS_FREQUENCY_GRID.dat ATOM_SOS_FREQUENCY_GRID.json librpa.in librpa.out
} >> provenance.txt
printf 'success\n' > STATUS
echo "C_TZDP_ATOM_SOS_OK work=$run_root reference_ha=$reference_ha ecrpa_ha=$ecrpa_ha naux=$naux"
