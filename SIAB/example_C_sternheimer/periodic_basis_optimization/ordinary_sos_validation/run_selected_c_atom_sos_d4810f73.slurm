#!/bin/bash
#SBATCH --job-name=c_atom_selected_sos
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-selected-sos-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-selected-sos-%j.err

set -euo pipefail
trap 'status=$?; echo "C_SELECTED_ATOM_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
base=$campaign/runs/product-pca-20260825
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
validation=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
frequency_extractor=$validation/extract_librpa_frequency_grid.py
producer=$base/selected-c-atom-response-aware-product-pca-55d25e3c9/producer
selection=$base/selected-fixed-prefix-checkpoint-500/SELECTED_CANDIDATE.json
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
expected_naux=261
expected_orbital_sha=3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$base/selected-c-atom-response-aware-product-pca-55d25e3c9/sos-nfreq6-d4810f73
reader=$work/reader_v1

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -x "$librpa" && test -x "$python"
test -s "$build_provenance" && test -s "$frequency_extractor" && test -s "$selection"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "${#expected_librpa_sha}" = 64
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -d "$producer"
grep -qx 'status=success' "$producer/provenance.txt"
grep -qx "naux=$expected_naux" "$producer/provenance.txt"
grep -qx "selected_orbital_sha256=$expected_orbital_sha" "$producer/provenance.txt"
test -s "$producer/ATOM_REFERENCE.txt" && test -s "$producer/GRID_MATRIX_HEADER.json"

mkdir -p "$reader"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat KS_eigenvector_1.dat stru_out v1_Cs_data_0.txt v1_coulomb_grid_iq_1_rank0.dat; do
  test -s "$producer/$name"
  ln -s "$producer/$name" "$reader/$name"
done
"$python" - "$producer/GRID_MATRIX_HEADER.json" "$expected_naux" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert payload["status"] == "success"
assert payload["natoms"] == 1 and payload["nblocks"] == 1
assert payload["naux"] == int(sys.argv[2])
PY

cat > "$work/librpa.in" <<EOF
input_dir = $reader/
task = rpa
nfreq = 6
tfgrids_type = minimax
parallel_routing = libri
version_coul_reader = 1
version_lri_reader = 1
prefix_coul_full = v1_coulomb_grid_iq_
prefix_lri_coeff = v1_Cs_data_
use_scalapack_ecrpa = t
use_shrink_abfs = f
use_shrink_chi = f
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

find -L "$reader" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work/INPUT_SHA256.txt"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=selected_c_atom_all_band_sos_body
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo selected_orbital_sha256=$expected_orbital_sha
  echo auxiliary_dimension=$expected_naux
  echo pca_threshold=1e-4
  echo nfreq=6
  echo spin_state=fixed_triplet_nup3_ndown1
  echo qpoint_count=1
  echo coulomb_kernel=full_periodic_poisson
  echo analytic_headwing=no_atom_supercell
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$selection" "$frequency_extractor" "$producer/ATOM_REFERENCE.txt" "$producer/GRID_MATRIX_HEADER.json" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
grep -q 'libRPA finished successfully' librpa.out
grep -q '^| Total EcRPA:' librpa.out
"$python" "$frequency_extractor" \
  --source librpa.out \
  --output ATOM_SOS_FREQUENCY_GRID.dat \
  --manifest ATOM_SOS_FREQUENCY_GRID.json \
  --expected-nfreq 6 > frequency_extract.out

reference_ha=$(awk '$1=="reference_ha" {print $2}' "$producer/ATOM_REFERENCE.txt")
ecrpa_ha=$(awk '/^\| Total EcRPA:/ {value=$NF} END {print value}' librpa.out)
frequency_sha=$(sha256sum ATOM_SOS_FREQUENCY_GRID.dat | awk '{print $1}')
"$python" - "$reference_ha" "$ecrpa_ha" "$frequency_sha" <<'PY' > RESULT_SUMMARY.txt
import math
import sys
reference = float(sys.argv[1])
ecrpa = float(sys.argv[2])
assert math.isfinite(reference) and math.isfinite(ecrpa)
assert len(sys.argv[3]) == 64
print("status success")
print("side atom")
print("method sos")
print("scope body_only_no_analytic_headwing")
print("coulomb_kernel full_periodic_poisson")
print("selected_orbital_sha256 3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1")
print(f"frequency_grid_sha256 {sys.argv[3]}")
print("frequency_grid_source atom_sos_minimax")
print("naux 261")
print(f"reference_ha {reference:.15f}")
print(f"ecrpa_ha {ecrpa:.15f}")
PY

{
  echo status=success
  echo frequency_grid_source=atom_sos_exact
  echo frequency_grid_sha256=$frequency_sha
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt ATOM_SOS_FREQUENCY_GRID.dat ATOM_SOS_FREQUENCY_GRID.json frequency_extract.out
} >> provenance.txt

echo "C_SELECTED_ATOM_SOS_OK work=$work ecrpa_ha=$ecrpa_ha"
