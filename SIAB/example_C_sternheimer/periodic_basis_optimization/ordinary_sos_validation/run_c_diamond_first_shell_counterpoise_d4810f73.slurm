#!/bin/bash
#SBATCH --job-name=c_bsse_nn1
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-bsse-nn1-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-bsse-nn1-%j.err

set -euo pipefail
trap 'status=$?; echo "C_DIAMOND_FIRST_SHELL_CP_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
raw_root=${RAW_BINDING_ROOT:?missing completed raw atom-solid binding root}
run_root=${RUN_ROOT:?missing immutable counterpoise run root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
abacus=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534/abacus_3p
abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
counterpoise=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/c_diamond_counterpoise.py
extractor=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/extract_librpa_frequency_grid.py
manifest_reader=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/read_periodic_candidate_manifest.py
source_atom=$raw_root/atom
raw_binding=$raw_root/binding/RESULT.json
suffix=C_ATOM_DIAMOND_FIRST_SHELL_GHOST_PCA
work=$run_root/ghost

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$run_root"
test -s "$raw_binding" && grep -qx success "$raw_root/binding/STATUS"
grep -qx 'status=success' "$candidate_root/provenance.txt"
test -x "$abacus" && test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
test -x "$librpa" && test -s "$build_provenance" && test -s "$counterpoise" && test -s "$extractor" && test -s "$manifest_reader"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
mapfile -t candidate < <("$python" "$manifest_reader" "$candidate_root")
manifest=${candidate[0]}
orbital_name=${candidate[1]}
orbital_sha=${candidate[2]}
ao_per_atom=${candidate[3]}
orbital_layout=${candidate[4]}
orbital=$candidate_root/$orbital_name
for asset in INPUT KPT C_ONCV_PBE-1.0.upf; do test -s "$source_atom/$asset"; done
"$python" - "$raw_binding" "$orbital_sha" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload["status"] == "success"
assert payload["selected_orbital_sha256"] == sys.argv[2]
PY

mkdir -p "$run_root"
"$python" "$counterpoise" prepare \
  --output-root "$work" \
  --orbital-filename "$orbital_name" \
  --pseudopotential-filename C_ONCV_PBE-1.0.upf \
  --lattice-constant-angstrom 3.6 \
  --box-angstrom 20.0 \
  --shell-count 1
cp "$source_atom/INPUT" "$source_atom/KPT" "$source_atom/C_ONCV_PBE-1.0.upf" "$orbital" "$work/"
cd "$work"

ghost_count=$("$python" -c 'import json; print(json.load(open("CLUSTER.json"))["ghost_count"])')
shell_count=$("$python" -c 'import json; print(json.load(open("CLUSTER.json"))["shell_count"])')
total_sites=$((ghost_count + 1))
nbands=$((ao_per_atom * total_sites))

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

set_input_key suffix "$suffix"
set_input_key calculation scf
set_input_key ntype 2
set_input_key nelec 4
set_input_key nspin 2
set_input_key nupdown 2
set_input_key nbands "$nbands"
set_input_key symmetry 0
set_input_key gamma_only 1
set_input_key kpar 1
set_input_key smearing_method fixed
set_input_key ocp 1
set_input_key ocp_set "3*1 $((nbands - 3))*0 1*1 $((nbands - 1))*0"
set_input_key init_wfc atomic
set_input_key init_chg atomic
set_input_key out_chg 1
set_input_key out_wfc_lcao 1
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
remove_input_key rpa_pca_fixed_nu
set_input_key exx_singularity_correction massidda
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 1
set_input_key nx 135
set_input_key ny 135
set_input_key nz 135
! grep -q '^ABFS_ORBITAL$' STRU
test "$(grep -xc "$orbital_name" STRU)" = 2

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
  echo purpose=diamond_c_first_neighbor_shell_counterpoise_ordinary_allband_sos
  echo job_id=${SLURM_JOB_ID:?}
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo orbital_layout=$orbital_layout
  echo selected_orbital_sha256=$orbital_sha
  echo shell_count=$shell_count
  echo ghost_count=$ghost_count
  echo total_sites=$total_sites
  echo nbands=$nbands
  echo pca_threshold=1e-4
  echo auxiliary_basis_rule=product_pca_threshold_only
  echo grid=135,135,135
  echo spin_state=fixed_triplet_nup3_ndown1
  echo n_bands_chi0=-1
  echo frequency_grid_rule=independent_nfreq6_minimax
  sha256sum "$0" "$abacus" "$librpa" "$build_provenance" "$counterpoise" "$extractor" "$manifest_reader" "$manifest" "$raw_binding" INPUT STRU KPT CLUSTER.json C_ONCV_PBE-1.0.upf "$orbital_name"
} > provenance.txt

/usr/bin/time -v -o abacus.time mpirun -np 1 "$abacus" > abacus.out 2>&1

out=OUT.${suffix}
matrix=v1_coulomb_full_iq_1_rank0.dat
eig_occ=$out/eig_occ.txt
test -s "$matrix" && test -s "$eig_occ" && test -s "$out/running_scf.log"
grep -q '#SCF IS CONVERGED#' "$out/running_scf.log"
! grep -q '!!SCF IS NOT CONVERGED!!' "$out/running_scf.log"
"$python" - "$matrix" "$eig_occ" "$nbands" "$total_sites" <<'PY'
import json, pathlib, re, struct, sys
with open(sys.argv[1], "rb") as handle:
    marker, iq, naux, value_flag, natoms, nblocks = struct.unpack("<6i", handle.read(24))
assert marker == -20129433 and iq == 1 and value_flag == 1 and naux > 0
assert natoms == int(sys.argv[4]) and nblocks >= natoms
states, spin = {1: [], 2: []}, None
for line in pathlib.Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"spin=(\d+)", line)
    if match:
        spin = int(match.group(1))
        continue
    fields = line.split()
    if spin in states and len(fields) == 3 and fields[0].isdigit():
        states[spin].append(float(fields[2]))
nbands = int(sys.argv[3])
assert all(len(values) == nbands for values in states.values())
assert states[1][:3] == [1.0, 1.0, 1.0] and all(abs(value) <= 1e-12 for value in states[1][3:])
assert states[2][0] == 1.0 and all(abs(value) <= 1e-12 for value in states[2][1:])
print(json.dumps({"status": "success", "naux": naux, "natoms": natoms, "nblocks": nblocks}, indent=2, sort_keys=True))
PY
"$python" - "$matrix" > GRID_MATRIX_HEADER.json <<'PY'
import json, struct, sys
with open(sys.argv[1], "rb") as handle:
    marker, iq, naux, value_flag, natoms, nblocks = struct.unpack("<6i", handle.read(24))
assert marker == -20129433 and iq == 1 and value_flag == 1 and naux > 0
print(json.dumps({"status": "success", "iq": iq, "naux": naux, "natoms": natoms, "nblocks": nblocks}, indent=2, sort_keys=True))
PY
naux=$("$python" -c 'import json; print(json.load(open("GRID_MATRIX_HEADER.json"))["naux"])')
reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' abacus.out)
"$python" - "$reference_ha" <<'PY'
import math, sys
assert math.isfinite(float(sys.argv[1]))
PY

reader=$work/reader_v1
mkdir "$reader"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out v1_Cs_data_0.txt "$matrix"; do
  test -s "$work/$name"
  ln -s "$work/$name" "$reader/$name"
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
"$python" "$extractor" --source librpa.out --output GHOST_SOS_FREQUENCY_GRID.dat --manifest GHOST_SOS_FREQUENCY_GRID.json --expected-nfreq 6 > frequency_extract.out
ecrpa_ha=$(awk '/^\| Total EcRPA:/ {value=$NF} END {print value}' librpa.out)
frequency_sha=$(sha256sum GHOST_SOS_FREQUENCY_GRID.dat | awk '{print $1}')
"$python" - "$reference_ha" "$ecrpa_ha" <<'PY'
import math, sys
assert all(math.isfinite(float(value)) for value in sys.argv[1:])
PY
"$python" - "$reference_ha" "$ecrpa_ha" "$orbital_sha" "$shell_count" "$ghost_count" "$naux" "$frequency_sha" > GHOST_SUMMARY.json <<'PY'
import json, sys
print(json.dumps({
    "status": "success",
    "quantity": "diamond_c_first_shell_ghost_atom_ordinary_sos",
    "reference_ha": float(sys.argv[1]),
    "ecrpa_ha": float(sys.argv[2]),
    "selected_orbital_sha256": sys.argv[3],
    "shell_count": int(sys.argv[4]),
    "ghost_count": int(sys.argv[5]),
    "naux": int(sys.argv[6]),
    "frequency_grid_sha256": sys.argv[7],
    "n_bands_chi0": -1,
    "pca_threshold": 1e-4,
    "coulomb_kernel": "full_periodic_poisson",
}, indent=2, sort_keys=True))
PY
"$python" "$counterpoise" collect \
  --raw-binding "$raw_binding" \
  --ghost-summary GHOST_SUMMARY.json \
  --output-root "$run_root/counterpoise"
grep -qx success "$run_root/counterpoise/STATUS"
grep -q '"counterpoise_acceptance": "requires_shell_convergence"' "$run_root/counterpoise/RESULT.json"

{
  echo status=success
  echo reference_ha=$reference_ha
  echo ecrpa_ha=$ecrpa_ha
  echo naux=$naux
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/abacus_wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/abacus_max_rss_kb=/p' abacus.time
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/librpa_wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/librpa_max_rss_kb=/p' librpa.time
  sha256sum "$matrix" GRID_MATRIX_HEADER.json GHOST_SUMMARY.json INPUT_SHA256.txt GHOST_SOS_FREQUENCY_GRID.dat GHOST_SOS_FREQUENCY_GRID.json librpa.in librpa.out "$run_root/counterpoise/RESULT.json"
} >> provenance.txt
printf 'success\n' > STATUS
echo "C_DIAMOND_FIRST_SHELL_CP_OK root=$run_root reference_ha=$reference_ha ecrpa_ha=$ecrpa_ha naux=$naux"

