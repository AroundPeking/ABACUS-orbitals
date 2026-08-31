#!/bin/bash
#SBATCH --job-name=c_f_twoq_tail
#SBATCH --partition=p1,48cp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-f-twoq-tail-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-f-twoq-tail-%j.err

set -euo pipefail
trap 'status=$?; echo "C_F_TWO_Q_TAIL_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
source_commit=${SIAB_SOURCE_COMMIT:?missing exact SIAB source commit}
grid_root=$run_root/solid-grid
work=$run_root/two-q-tail
reader=$work/reader_v1
q_indices=(2 6)
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
scripts=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
zero_writer=$scripts/sparse_qstar_coulomb.py
parser=$scripts/build_qstar_frequency_decomposition.py
manifest_reader=$scripts/read_periodic_candidate_manifest.py
librpa=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-freqdiag-87cf66a7/build-df-intel/chi0_main.exe
librpa_commit=87cf66a768afaa45714a5aed5431b53ac7ea0eb3
parent_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
expected_librpa_sha=de334e398cccd3343da3003bd3c654474faabc467e72ba321ff3d215e4e300e4

[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test "${source_commit:0:8}" = "${source_root##*-}"
case "${SLURM_JOB_PARTITION:?}" in
  p1|48cp2) ;;
  *) echo "unsupported df partition: $SLURM_JOB_PARTITION" >&2; exit 1 ;;
esac
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -n "${QSTAR_ARRAY_JOB_ID:?missing q2/q6 array job id}"
grep -qx 'status=success' "$candidate_root/provenance.txt"
test -s "$zero_writer" && test -s "$parser" && test -s "$manifest_reader"
test -x "$librpa"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
mapfile -t candidate < <("$python" "$manifest_reader" "$candidate_root")
manifest=${candidate[0]}
orbital_sha=${candidate[2]}

mkdir -p "$reader" "$work/zero-manifests"
q2=$grid_root/q2
q6=$grid_root/q6
for qdir in "$q2" "$q6"; do
  grep -qx 'status=success' "$qdir/provenance.txt"
  grep -qx 'success' "$qdir/STATUS"
done
grep -qx 'full_q_index=2' "$q2/provenance.txt"
grep -qx 'full_q_index=6' "$q6/provenance.txt"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out; do
  test -s "$q2/$name"
  ln -s "$q2/$name" "$reader/$name"
done
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(find "$q2" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c -print0 | sort -z)
test "$(find -L "$reader" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)" -eq 1

for q_index in "${q_indices[@]}"; do
  matrix=$grid_root/q${q_index}/v1_coulomb_grid_iq_${q_index}_rank0.dat
  test -s "$matrix"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done
template=$q2/v1_coulomb_grid_iq_2_rank0.dat
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
debug = t
EOF
find -L "$reader" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work/INPUT_SHA256.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
export I_MPI_FABRICS=shm:ofi
{
  echo status=running
  echo purpose=controlled_f_q2_q6_high_frequency_tail_gate
  echo qstar_array_job=$QSTAR_ARRAY_JOB_ID
  echo source_commit=$source_commit
  echo librpa_commit=$librpa_commit
  echo parent_physics_commit=$parent_commit
  echo librpa_sha256=$expected_librpa_sha
  echo selected_orbital_sha256=$orbital_sha
  echo qstar_indices=2,6
  echo n_bands_chi0=-1
  echo nfreq=6
  echo nonzero_record_count=12
  sha256sum "$0" "$manifest_reader" "$manifest" "$zero_writer" "$parser" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2> librpa.err
grep -q 'libRPA finished successfully' librpa.out
test "$(grep -c '^RPA normal split ifreq=' librpa.out)" -eq 384
test "$(grep -c '^RPA freqdiag ifreq=' librpa.out)" -eq 384
grep '^RPA normal split ifreq=' librpa.out > normal_split.dat
grep '^RPA freqdiag ifreq=' librpa.out > freqdiag.dat
actual_ecrpa=$(awk '/Total EcRPA:/{value=$NF} END{print value}' librpa.out)
"$python" "$parser" \
  --normal-split normal_split.dat \
  --freqdiag freqdiag.dat \
  --name controlled_f_two_q \
  --qstar-indices 2,6 \
  --output-tsv decomposition.tsv \
  --output-json FREQUENCY_RESULT.json \
  --expected-sparse-ecrpa "$actual_ecrpa" \
  > parser.out

gate=$(
  "$python" - FREQUENCY_RESULT.json GATE_RESULT.json <<'PY'
import json, pathlib, sys
result = json.load(open(sys.argv[1], encoding="ascii"))
thresholds = {
    "q2": {
        "highest_frequency_cancellation_ratio": 0.010207674744,
        "high_frequency_tail_fraction": 0.021127738990,
    },
    "q6": {
        "highest_frequency_cancellation_ratio": 0.010130257316,
        "high_frequency_tail_fraction": 0.021125217918,
    },
}
comparisons = {}
passed = True
for q, limits in thresholds.items():
    comparisons[q] = {}
    for metric, limit in limits.items():
        value = float(result["q_metrics"][q][metric])
        item_passed = value <= limit
        passed = passed and item_passed
        comparisons[q][metric] = {
            "value": value,
            "limit": limit,
            "ratio_to_limit": value / limit,
            "gate": "pass" if item_passed else "fail",
        }
payload = {
    "status": "success",
    "scope": "empirical q2/q6 high-frequency-tail stability gate",
    "physics_interpretation": "operational screen; not proof of a dielectric eigenmode mechanism",
    "tail_gate": "pass" if passed else "fail",
    "comparisons": comparisons,
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="ascii",
)
print(payload["tail_gate"])
PY
)

{
  echo status=success
  echo actual_ecrpa=$actual_ecrpa
  echo tail_gate=$gate
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out normal_split.dat freqdiag.dat decomposition.tsv FREQUENCY_RESULT.json GATE_RESULT.json INPUT_SHA256.txt
} >> provenance.txt
if test "$gate" = pass; then
  printf 'success\n' > STATUS
  echo "C_F_TWO_Q_TAIL_PASS root=$work"
else
  printf 'rejected\n' > STATUS
  echo "C_F_TWO_Q_TAIL_REJECTED root=$work"
fi
