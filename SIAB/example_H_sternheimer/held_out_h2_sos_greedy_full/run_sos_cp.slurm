#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=siab_full_sos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=sos.%A_%a.out
#SBATCH --error=sos.%A_%a.err

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

case "${SLURM_ARRAY_TASK_ID:?}" in
  0)
    case_name=H2
    nbands=334
    expected_spins=1
    expected_electrons=2
    output_dir=OUT.H2_SIAB_GREEDY_FULL_13S11P10D5F4G_BOX20A_FIXED_ABS_RMESH5
    ;;
  1)
    case_name=H
    nbands=167
    expected_spins=2
    expected_electrons=1
    output_dir=OUT.H_SIAB_GREEDY_FULL_13S11P10D5F4G_BOX20A_FIXED_ABS_RMESH5
    ;;
  2)
    case_name=H_ghost
    nbands=334
    expected_spins=2
    expected_electrons=1
    output_dir=OUT.H_GHOST_SIAB_GREEDY_FULL_13S11P10D5F4G_BOX20A_FIXED_ABS_RMESH5
    ;;
  *)
    echo "invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

script_dir=${SLURM_SUBMIT_DIR:?}
case_dir=$script_dir/cases/$case_name
orbital=H_gga_8au_100Ry_13s11p10d5f4g.orb
pseudo=H_ONCV_PBE-1.0.upf
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

test -s "$script_dir/SOURCE_COMMIT"
test -d "$case_dir"
test ! -e "$case_dir/$output_dir"
test ! -e "$case_dir/band_out"
test ! -e "$case_dir/basis_wfc_out"
test ! -e "$case_dir/basis_aux_out"
test ! -e "$case_dir/sos_full_nfreq16_dump"
if [[ $case_name == H_ghost ]]; then
  grep -q '^H_empty ' "$case_dir/STRU"
fi

check_hash d518c5997f667249554ab19af1a36d12f17439ffb7f0ecaf5977c1e829be7b00 \
  "$script_dir/$orbital"
check_hash 3f5deaaf57ff09d87da2c2c186eb1654d174fe385c9edc265036b3c8869286ef \
  "$script_dir/$pseudo"
check_hash d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f \
  "$script_dir/$fixed_abs"
check_hash 2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d \
  "$abacus"
check_hash defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c \
  "$librpa"

cp "$script_dir/$orbital" "$case_dir/$orbital"
cp "$script_dir/$pseudo" "$case_dir/$pseudo"
cp "$script_dir/$fixed_abs" "$case_dir/$fixed_abs"
cd "$case_dir"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE

echo "date_start=$(date +%F_%T)"
echo "case=$case_name"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "omp_threads=$OMP_NUM_THREADS"
echo "source_commit=$(cat "$script_dir/SOURCE_COMMIT")"
echo "abacus_source_commit=$abacus_source_commit"
echo "librpa_reference_commit=$librpa_reference_commit"
echo "version_exception=historical_exact_sos_binary_for_apples_to_apples"
sha256sum INPUT STRU KPT librpa.in "$orbital" "$pseudo" "$fixed_abs"

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

awk -v expected_bands="$nbands" \
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
