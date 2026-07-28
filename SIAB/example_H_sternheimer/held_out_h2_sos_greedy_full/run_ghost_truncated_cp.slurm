#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=siab_ghost_nb160
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --output=ghost-nb160.%j.out
#SBATCH --error=ghost-nb160.%j.err

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

script_dir=${SLURM_SUBMIT_DIR:?}
campaign_root=${SIAB_TRUNCATED_CP_ROOT:?}
template_dir=$script_dir/cases/H_ghost
case_dir=$campaign_root/ghost_nbands160_case
output_dir=OUT.H_GHOST_SIAB_GREEDY_FULL_13S11P10D5F4G_BOX20A_NB160_FIXED_ABS_RMESH5
orbital=H_gga_8au_100Ry_13s11p10d5f4g.orb
pseudo=H_ONCV_PBE-1.0.upf
fixed_abs=H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs
abacus=/public/home/ghj/app/src/abacus-sternheimer-siab-task4-80a606f57a26/build-task4-o3-debug-info/abacus-80a606f57a26
librpa=/public/home/ghj/app/src/LibRPA-sternheimer-rpa-20260707/build-sternheimer-fast/chi0_main.exe
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python
expected_nbands=160
expected_spins=2
expected_electrons=1

check_hash() {
  local expected=$1
  local path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

test -s "$script_dir/SOURCE_COMMIT"
test -s "$script_dir/SOURCE_SHA256SUMS"
(cd "$script_dir" && sha256sum -c SOURCE_SHA256SUMS)
test -d "$template_dir"
test ! -e "$campaign_root"
mkdir -p "$case_dir"
install -m 0644 "$template_dir/INPUT" "$case_dir/INPUT"
cp "$template_dir/STRU" "$template_dir/KPT" "$template_dir/librpa.in" \
  "$case_dir/"
cp "$script_dir/$orbital" "$script_dir/$pseudo" "$script_dir/$fixed_abs" \
  "$case_dir/"

check_hash d518c5997f667249554ab19af1a36d12f17439ffb7f0ecaf5977c1e829be7b00 \
  "$case_dir/$orbital"
check_hash 3f5deaaf57ff09d87da2c2c186eb1654d174fe385c9edc265036b3c8869286ef \
  "$case_dir/$pseudo"
check_hash d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f \
  "$case_dir/$fixed_abs"
check_hash 2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d \
  "$abacus"
check_hash defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c \
  "$librpa"

"$python" "$script_dir/analyze_truncated_cp.py" prepare-ghost-input \
  --source "$template_dir/INPUT" \
  --output "$case_dir/INPUT" \
  --suffix "$output_dir"

"$python" - "$case_dir/librpa.in" <<'PY'
import pathlib
import sys

values = {}
for raw in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    key, value = (field.strip() for field in line.split("=", 1))
    if key in values:
        raise SystemExit(f"duplicate librpa.in key: {key}")
    values[key] = value
expected = {
    "task": "rpa",
    "prefix_coul_full": "v1_coulomb_full_iq_",
    "version_coul_reader": "1",
    "version_lri_reader": "1",
    "nfreq": "16",
}
for key, value in expected.items():
    if values.get(key) != value:
        raise SystemExit(f"librpa.in {key} expected {value}, got {values.get(key)}")
if "n_bands_chi0" in values:
    raise SystemExit("source librpa.in must use all producer bands")
PY

cd "$case_dir"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE

echo "date_start=$(date +%F_%T)"
echo "case=H_ghost_nbands160"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "omp_threads=$OMP_NUM_THREADS"
echo "source_commit=$(cat "$script_dir/SOURCE_COMMIT")"
echo "abacus_source_commit=80a606f57a2610bc2532468661b687b01f58074c"
echo "librpa_reference_commit=d70b55714515c4c6446a87fa96fd1fd62c7f6fde"
echo "version_exception=historical_exact_sos_binary_for_apples_to_apples"
sha256sum INPUT STRU KPT librpa.in "$orbital" "$pseudo" "$fixed_abs"

/usr/bin/time -v mpirun -np 1 -ppn 1 "$abacus" \
  > "abacus.${SLURM_JOB_ID}.out" 2>&1

grep -q '!FINAL_ETOT_IS' "$output_dir/running_scf.log"
grep -q 'rpa_lcao_exx(Ha):' "abacus.${SLURM_JOB_ID}.out"
test -s band_out
test -s basis_wfc_out
test -s basis_aux_out
test -s KS_eigenvector_0.dat
test -s v1_Cs_data_0.txt
test -s v1_coulomb_full_iq_1_rank0.dat

awk -v expected_bands="$expected_nbands" \
    -v expected_spins="$expected_spins" \
    -v expected_electrons="$expected_electrons" '
  NR == 2 && $1 != expected_spins { exit 2 }
  NR == 3 && $1 != expected_bands { exit 3 }
  NR >= 7 && NF >= 4 {
    rows += 1
    occ = $2 + 0.0
    occ_sum += occ
    if (occ < -1.0e-8 || occ > 2.0 + 1.0e-8) { exit 4 }
    nearest = int(occ + 0.5)
    diff = occ - nearest
    if (diff < 0) { diff = -diff }
    if (diff > 1.0e-8) { exit 5 }
  }
  END {
    if (rows != expected_bands * expected_spins) { exit 6 }
    diff = occ_sum - expected_electrons
    if (diff < 0) { diff = -diff }
    if (diff > 1.0e-8) { exit 7 }
  }
' band_out

for lane in 120 160; do
  lane_dir=$case_dir/librpa_nbands$lane
  mkdir "$lane_dir"
  "$python" - "$case_dir/librpa.in" "$lane_dir/librpa.in" "$case_dir" "$lane" <<'PY'
import pathlib
import sys

source_path, output_path, input_dir, lane = sys.argv[1:]
result = []
for raw in pathlib.Path(source_path).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line.startswith("input_dir"):
        result.append(f"input_dir = {input_dir}")
    elif line.startswith("output_dir"):
        result.append("output_dir = ./dump")
    else:
        result.append(raw)
if lane == "120":
    result.append("n_bands_chi0 = 120")
elif lane != "160":
    raise SystemExit(f"unsupported lane {lane}")
pathlib.Path(output_path).write_text("\n".join(result) + "\n", encoding="utf-8")
PY
  (
    cd "$lane_dir"
    /usr/bin/time -v mpirun -np 1 -ppn 1 "$librpa" \
      > "librpa.nbands${lane}.${SLURM_JOB_ID}.out" 2>&1
  )
  grep -q 'libRPA finished successfully' \
    "$lane_dir/librpa.nbands${lane}.${SLURM_JOB_ID}.out"
  grep -q 'Total EcRPA:' \
    "$lane_dir/librpa.nbands${lane}.${SLURM_JOB_ID}.out"
done

sha256sum band_out basis_wfc_out basis_aux_out KS_eigenvector_0.dat \
  v1_Cs_data_0.txt v1_coulomb_full_iq_1_rank0.dat \
  librpa_nbands120/librpa.in librpa_nbands160/librpa.in \
  > PRODUCTION_OUTPUTS.sha256
echo "date_end=$(date +%F_%T)"
