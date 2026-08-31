#!/bin/bash
#SBATCH --job-name=c_thr_qsos
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-threshold-qstar-sos-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-threshold-qstar-sos-%j.err

set -euo pipefail
trap 'status=$?; echo "C_THRESHOLD_QSTAR_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
source_commit=${SIAB_SOURCE_COMMIT:?missing exact SIAB source commit}
grid_root=$run_root/solid-grid
work=$run_root/solid-sos
reader=$work/reader_v1
zero_writer=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/sparse_qstar_coulomb.py
qstar_gate=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/sparse_qstar_sos_gate.py
extractor=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/extract_librpa_frequency_grid.py
manifest_reader=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/read_periodic_candidate_manifest.py
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
q_indices=(1 2 3 6 7 8 11 28)

[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test "${source_commit:0:8}" = "${source_root##*-}"
case "${SLURM_JOB_PARTITION:?}" in
  p1|48cp2) ;;
  *) echo "unsupported df partition: $SLURM_JOB_PARTITION" >&2; exit 1 ;;
esac
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
grep -qx 'status=success' "$candidate_root/provenance.txt"
test -s "$zero_writer" && test -s "$qstar_gate" && test -s "$extractor" && test -s "$manifest_reader"
test -x "$librpa" && test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
mapfile -t candidate < <("$python" "$manifest_reader" "$candidate_root")
manifest=${candidate[0]}
orbital_sha=${candidate[2]}

mkdir -p "$reader" "$work/zero-manifests"
q1=$grid_root/q1
grep -qx 'status=success' "$q1/provenance.txt"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out; do
  test -s "$q1/$name"
  ln -s "$q1/$name" "$reader/$name"
done
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(find "$q1" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c -print0 | sort -z)
test "$(find -L "$reader" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)" -eq 1

for q_index in "${q_indices[@]}"; do
  qdir=$grid_root/q${q_index}
  grep -qx 'status=success' "$qdir/provenance.txt"
  grep -qx "full_q_index=$q_index" "$qdir/provenance.txt"
  matrix=$qdir/v1_coulomb_grid_iq_${q_index}_rank0.dat
  test -s "$matrix" && test -s "$qdir/GRID_MATRIX_HEADER.json"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done
naux=$("$python" - "$grid_root" "${q_indices[@]}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
indices = [int(value) for value in sys.argv[2:]]
dimensions = {
    json.loads((root / f"q{index}" / "GRID_MATRIX_HEADER.json").read_text(encoding="ascii"))["naux"]
    for index in indices
}
assert len(dimensions) == 1
value = dimensions.pop()
assert value > 0 and value % 2 == 0
print(value)
PY
)
template=$q1/v1_coulomb_grid_iq_1_rank0.dat
for q_index in $(seq 1 64); do
  if printf '%s\n' "${q_indices[@]}" | grep -qx "$q_index"; then
    continue
  fi
  "$python" "$zero_writer" \
    --source "$template" \
    --output "$reader/v1_coulomb_grid_iq_${q_index}_rank0.dat" \
    --iq "$q_index" \
    --manifest "$work/zero-manifests/q${q_index}.json" > /dev/null
done
test "$(find -L "$reader" -maxdepth 1 -type f -name 'v1_coulomb_grid_iq_*_rank0.dat' -size +0c | wc -l)" -eq 64

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
n_bands_chi0 = -1
vq_threshold = 0
sqrt_coulomb_threshold = 1e-5
cs_threshold = 0
use_fullcoul_exx = t
use_symmetry_exx = f
use_symmetry_rpa = f
use_symmetry_gw = f
replace_w_head = f
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
  echo purpose=threshold_candidate_c_solid_sparse_eight_qstar_all_band_sos
  echo qstar_array_job=${QSTAR_ARRAY_JOB_ID:?missing q-star array job id}
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo selected_orbital_sha256=$orbital_sha
  echo naux=$naux
  echo pca_threshold=1e-4
  echo auxiliary_basis_rule=product_pca_threshold_only
  echo nfreq=6
  echo n_bands_chi0=-1
  echo kmesh=4,4,4
  echo qstar_indices=1,2,3,6,7,8,11,28
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  sha256sum "$0" "$manifest_reader" "$manifest" "$zero_writer" "$qstar_gate" "$extractor" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
grep -q 'libRPA finished successfully' librpa.out
grep -q '^| Total EcRPA:' librpa.out
"$python" "$extractor" --source librpa.out --output SOLID_SOS_FREQUENCY_GRID.dat --manifest SOLID_SOS_FREQUENCY_GRID.json --expected-nfreq 6 > frequency_extract.out
"$python" "$qstar_gate" \
  --sparse-output "allband=$work/librpa.out" \
  --output-root "$work/qstar-gate" \
  --binding-tolerance-kcal-mol-per-c 0.1 \
  --provenance "source_commit=$source_commit" \
  --provenance "qstar_array_job=$QSTAR_ARRAY_JOB_ID" > qstar_gate.out
grep -qx 'success' qstar-gate/STATUS
ecrpa_ha=$("$python" - qstar-gate/RESULT.json <<'PY'
import json, math, sys
payload = json.load(open(sys.argv[1], encoding="ascii"))
assert payload["status"] == "success" and payload["sparse_qstar_gate"] == "pass"
assert len(payload["rows"]) == 1
value = float(payload["rows"][0]["qstar_reconstruction_ha"])
assert math.isfinite(value)
print(f"{value:.17g}")
PY
)
reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' "$q1/abacus.out")
frequency_sha=$(sha256sum SOLID_SOS_FREQUENCY_GRID.dat | awk '{print $1}')
"$python" - "$reference_ha" "$ecrpa_ha" <<'PY'
import math, sys
assert all(math.isfinite(float(value)) for value in sys.argv[1:])
PY
cat > RESULT_SUMMARY.txt <<EOF
status success
side solid
method sos
scope body_only_no_analytic_headwing_qstar_reconstructed
coulomb_kernel full_periodic_poisson
selected_orbital_sha256 $orbital_sha
frequency_grid_sha256 $frequency_sha
naux $naux
reference_ha $reference_ha
ecrpa_ha $ecrpa_ha
EOF
{
  echo status=success
  echo reference_ha=$reference_ha
  echo ecrpa_qstar_reconstructed_ha=$ecrpa_ha
  echo naux=$naux
  echo frequency_grid_sha256=$frequency_sha
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt SOLID_SOS_FREQUENCY_GRID.dat SOLID_SOS_FREQUENCY_GRID.json qstar-gate/RESULT.json
} >> provenance.txt
printf 'success\n' > STATUS
echo "C_THRESHOLD_QSTAR_SOS_OK root=$work ecrpa_ha=$ecrpa_ha"
