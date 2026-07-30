#!/bin/bash
#SBATCH -J h2_guarded_dzp_cp
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH -t 1-00:00:00
#SBATCH --array=0-2
#SBATCH -o guarded_sos_cp.%A_%a.out
#SBATCH -e guarded_sos_cp.%A_%a.err

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

source_root=/work1/ghj/sternheimer_abacus_tests/siab_fixed_dzp_tzdp_sos_source_v2_20260730
campaign_root=/work1/ghj/sternheimer_abacus_tests/siab_low_frequency_guard_sos_campaign_f17a794d_20260730
asset_root=/work1/ghj/sternheimer_abacus_tests/h_atom_st_rpa_pca1e-4_rmesh5_20260704/st_atom_nspin1_gauss_abfs
candidate_root=/work1/ghj/sternheimer_abacus_tests/siab_low_frequency_guard_active_f17a794d_20260730/source/SIAB/example_H_sternheimer
template_root=$source_root/SIAB/example_H_sternheimer/held_out_h2_sos_greedy_full/cases
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

lanes=(guarded_fixed_dzp_joint guarded_fixed_dzp_joint guarded_fixed_dzp_joint)
case_names=(H2 H H_ghost)
# Each H has 3 s functions and 2 complete p shells: 3 + 2 * 3 = 9 AO.
nbands=(18 9 18)
expected_spins=(1 2 2)
expected_electrons=(2 1 1)
expected_orbital_entries=(1 1 2)

task=${SLURM_ARRAY_TASK_ID:?}
lane=${lanes[$task]}
case_name=${case_names[$task]}
case_bands=${nbands[$task]}
case_spins=${expected_spins[$task]}
case_electrons=${expected_electrons[$task]}
case_orbital_entries=${expected_orbital_entries[$task]}
template_dir=$template_root/$case_name
run_root=$campaign_root/$lane/$case_name

case $lane in
  guarded_fixed_dzp_joint)
    orbital_name=H_gga_8au_100Ry_guarded_fixed_dzp_joint_3s2p.orb
    orbital_source=$candidate_root/ORBITAL_1U.dat
    orbital_sha256=81c3f21a817d30d9d6802529650c4f177e27a62b39bea5d9d9de8f6c425d5330
    ;;
  *) echo "invalid lane: $lane" >&2; exit 2 ;;
esac

pseudo_name=H_ONCV_PBE-1.0.upf
fixed_abs=H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs
abacus=/public/home/ghj/app/src/abacus-sternheimer-siab-task4-80a606f57a26/build-task4-o3-debug-info/abacus-80a606f57a26
librpa=/public/home/ghj/app/src/LibRPA-sternheimer-rpa-20260707/build-sternheimer-fast/chi0_main.exe
abacus_source_commit=80a606f57a2610bc2532468661b687b01f58074c
librpa_reference_commit=d70b55714515c4c6446a87fa96fd1fd62c7f6fde

check_hash() {
  local expected=$1
  local path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

test -s "$source_root/SOURCE_COMMIT"
test "$(cat "$candidate_root/SOURCE_COMMIT")" = \
  f17a794d134fb2c4bcd7ec7634d0eda0ff5e2cae
test -d "$template_dir"
test ! -e "$run_root"
check_hash "$orbital_sha256" "$orbital_source"
check_hash 3f5deaaf57ff09d87da2c2c186eb1654d174fe385c9edc265036b3c8869286ef \
  "$asset_root/$pseudo_name"
check_hash d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f \
  "$asset_root/$fixed_abs"
check_hash 2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d \
  "$abacus"
check_hash defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c \
  "$librpa"

mkdir -p "$run_root"
install -m 0644 "$template_dir/INPUT" "$template_dir/STRU" \
  "$template_dir/KPT" "$template_dir/librpa.in" "$run_root/"
cp "$orbital_source" "$run_root/$orbital_name"
cp "$asset_root/$pseudo_name" "$asset_root/$fixed_abs" "$run_root/"

suffix="LOW_FREQUENCY_GUARD_${lane}_${case_name}_BOX20A_FIXED_ABS_RMESH5"
"$python" - "$run_root/INPUT" "$run_root/STRU" "$suffix" "$case_bands" \
  "$orbital_name" "$fixed_abs" "$case_orbital_entries" <<'PY'
from pathlib import Path
import sys

input_path = Path(sys.argv[1])
stru_path = Path(sys.argv[2])
suffix = sys.argv[3]
nbands = sys.argv[4]
orbital = sys.argv[5]
fixed_abs = sys.argv[6]
expected_orbital_count = int(sys.argv[7])

lines = input_path.read_text().splitlines()
replacements = {"suffix": suffix, "nbands": nbands}
counts = {key: 0 for key in replacements}
output = []
for line in lines:
    fields = line.split()
    if fields and fields[0] in replacements:
        key = fields[0]
        output.append(f"{key:<24}{replacements[key]}")
        counts[key] += 1
    else:
        output.append(line)
if counts != {"suffix": 1, "nbands": 1}:
    raise SystemExit(f"INPUT rewrite count mismatch: {counts}")
input_path.write_text("\n".join(output) + "\n")

stru = stru_path.read_text()
old = "H_gga_8au_100Ry_13s11p10d5f4g.orb"
if stru.count(old) != expected_orbital_count:
    raise SystemExit("STRU orbital rewrite count mismatch")
stru = stru.replace(old, orbital)
if stru.count(fixed_abs) != expected_orbital_count:
    raise SystemExit("STRU fixed auxiliary basis count mismatch")
stru_path.write_text(stru)
PY

cd "$run_root"
output_dir=OUT.$suffix
test ! -e "$output_dir"
test ! -e band_out
test ! -e basis_wfc_out
test ! -e basis_aux_out
test ! -e sos_full_nfreq16_dump

"$python" - INPUT librpa.in "$case_bands" "$case_spins" "$case_electrons" <<'PY'
from pathlib import Path
import sys

values = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    fields = line.split()
    if len(fields) >= 2 and fields[0] != "INPUT_PARAMETERS":
        values[fields[0]] = " ".join(fields[1:])
expected = {
    "nbands": sys.argv[3],
    "nelec": sys.argv[5],
    "ecutwfc": "100",
    "rpa": "1",
    "out_librpa_reader_version": "1",
    "exx_pca_threshold": "10",
    "exx_singularity_correction": "massidda",
    "rpa_ccp_rmesh_times": "5",
}
if values.get("nspin", "1") != sys.argv[4]:
    raise SystemExit("INPUT nspin mismatch")
for key, value in expected.items():
    if values.get(key) != value:
        raise SystemExit(f"INPUT {key} expected {value}, got {values.get(key)}")
librpa = Path(sys.argv[2]).read_text()
for token in (
    "task = rpa",
    "nfreq = 16",
    "prefix_coul_full = v1_coulomb_full_iq_",
    "vq_threshold = 0",
    "sqrt_coulomb_threshold = 0",
):
    if token not in librpa:
        raise SystemExit(f"librpa.in missing {token}")
PY

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE

echo "date_start=$(date +%F_%T)"
echo "lane=$lane"
echo "case=$case_name"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "omp_threads=$OMP_NUM_THREADS"
echo "source_commit=$(cat "$source_root/SOURCE_COMMIT")"
echo "optimizer_source_commit=$(cat "$candidate_root/SOURCE_COMMIT")"
echo "orbital_sha256=$orbital_sha256"
echo "abacus_source_commit=$abacus_source_commit"
echo "librpa_reference_commit=$librpa_reference_commit"
echo "version_exception=historical_exact_sos_binary"
sha256sum INPUT STRU KPT librpa.in "$orbital_name" "$pseudo_name" "$fixed_abs"

/usr/bin/time -v mpirun -np 1 -ppn 1 "$abacus" \
  > "abacus.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

grep -q '!FINAL_ETOT_IS' "$output_dir/running_scf.log"
grep -q 'rpa_lcao_exx(Ha):' \
  "abacus.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
test -s band_out
test -s basis_wfc_out
test -s basis_aux_out
test -s v1_coulomb_full_iq_1_rank0.dat
test -s v1_Cs_data_0.txt
test -s KS_eigenvector_0.dat

awk -v expected_bands="$case_bands" \
    -v expected_spins="$case_spins" \
    -v expected_electrons="$case_electrons" '
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

/usr/bin/time -v mpirun -np 1 -ppn 1 "$librpa" \
  > "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

grep -q 'libRPA finished successfully' \
  "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
grep -q 'Total EcRPA:' \
  "librpa.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
sha256sum band_out basis_wfc_out basis_aux_out \
  v1_coulomb_full_iq_1_rank0.dat v1_Cs_data_0.txt KS_eigenvector_0.dat \
  > PRODUCTION_OUTPUTS.sha256
echo "date_end=$(date +%F_%T)"
