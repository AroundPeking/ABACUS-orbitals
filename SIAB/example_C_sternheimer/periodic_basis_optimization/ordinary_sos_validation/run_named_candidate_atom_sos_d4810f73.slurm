#!/bin/bash
#SBATCH --job-name=c2g_atom_sos
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c2g-atom-sos-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c2g-atom-sos-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NAMED_ATOM_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
manifest=$candidate_root/CANDIDATE.json
producer=$run_root/atom-producer
work=$run_root/atom-sos
reader=$work/reader_v1
extractor=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/extract_librpa_frequency_grid.py
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt

test "${SLURM_JOB_PARTITION:?}" = p1 && test "${SLURM_NTASKS:?}" = 1 && test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -s "$manifest" && test -s "$producer/provenance.txt" && test -s "$producer/GRID_MATRIX_HEADER.json"
grep -qx 'status=success' "$producer/provenance.txt"
test -x "$librpa" && test -x "$python" && test -s "$extractor" && test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
mapfile -t values < <(python3 - "$manifest" "$producer/GRID_MATRIX_HEADER.json" <<'PY'
import json, sys
candidate = json.load(open(sys.argv[1], encoding="ascii"))
header = json.load(open(sys.argv[2], encoding="ascii"))
assert candidate["status"] == "success" and header["status"] == "success"
print(candidate["exported_orbital_sha256"])
print(header["naux"])
PY
)
orbital_sha=${values[0]}
naux=${values[1]}

mkdir -p "$reader"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out v1_Cs_data_0.txt v1_coulomb_full_iq_1_rank0.dat; do
  test -s "$producer/$name"
  ln -s "$producer/$name" "$reader/$name"
done
cat > "$work/librpa.in" <<EOF
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
  echo purpose=named_candidate_c_atom_all_band_sos
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo selected_orbital_sha256=$orbital_sha
  echo naux=$naux
  echo nfreq=6
  echo n_bands_chi0=-1
  echo pca_threshold=1e-4
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  echo mkl_threads=$MKL_NUM_THREADS
  sha256sum "$0" "$manifest" "$extractor" "$producer/ATOM_REFERENCE.txt" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
grep -q 'libRPA finished successfully' librpa.out
grep -q '^| Total EcRPA:' librpa.out
"$python" "$extractor" --source librpa.out --output ATOM_SOS_FREQUENCY_GRID.dat --manifest ATOM_SOS_FREQUENCY_GRID.json --expected-nfreq 6 > frequency_extract.out
reference_ha=$(awk '$1=="reference_ha" {print $2}' "$producer/ATOM_REFERENCE.txt")
ecrpa_ha=$(awk '/^\| Total EcRPA:/ {value=$NF} END {print value}' librpa.out)
frequency_sha=$(sha256sum ATOM_SOS_FREQUENCY_GRID.dat | awk '{print $1}')
python3 - "$reference_ha" "$ecrpa_ha" <<'PY'
import math, sys
assert all(math.isfinite(float(value)) for value in sys.argv[1:])
PY
cat > RESULT_SUMMARY.txt <<EOF
status success
side atom
method sos
scope body_only_no_analytic_headwing
coulomb_kernel full_periodic_poisson
selected_orbital_sha256 $orbital_sha
frequency_grid_sha256 $frequency_sha
naux $naux
reference_ha $reference_ha
ecrpa_ha $ecrpa_ha
EOF
{
  echo status=success
  echo frequency_grid_sha256=$frequency_sha
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt ATOM_SOS_FREQUENCY_GRID.dat ATOM_SOS_FREQUENCY_GRID.json frequency_extract.out
} >> provenance.txt
echo "C_NAMED_ATOM_SOS_OK work=$work ecrpa_ha=$ecrpa_ha"
