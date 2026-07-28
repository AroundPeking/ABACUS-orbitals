#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=siab_ghost_mpi2
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --output=ghost-mpi2.%j.out
#SBATCH --error=ghost-mpi2.%j.err

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

script_dir=${SLURM_SUBMIT_DIR:?}
template_dir=$script_dir/cases/H_ghost
case_dir=$script_dir/ghost_mpi2_case
output_dir=OUT.H_GHOST_SIAB_GREEDY_FULL_13S11P10D5F4G_BOX20A_FIXED_ABS_RMESH5
orbital=H_gga_8au_100Ry_13s11p10d5f4g.orb
pseudo=H_ONCV_PBE-1.0.upf
fixed_abs=H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs
abacus=/public/home/ghj/app/src/abacus-sternheimer-siab-task4-80a606f57a26/build-task4-o3-debug-info/abacus-80a606f57a26
librpa=/public/home/ghj/app/src/LibRPA-sternheimer-rpa-20260707/build-sternheimer-fast/chi0_main.exe
abacus_source_commit=80a606f57a2610bc2532468661b687b01f58074c
librpa_reference_commit=d70b55714515c4c6446a87fa96fd1fd62c7f6fde
expected_reader_ranks=2

check_hash() {
  local expected=$1
  local path=$2
  test -s "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

test -s "$script_dir/SOURCE_COMMIT"
test -d "$template_dir"
test ! -e "$case_dir"
mkdir "$case_dir"
cp "$template_dir/INPUT" "$template_dir/STRU" "$template_dir/KPT" \
  "$template_dir/librpa.in" "$case_dir/"
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

cd "$case_dir"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
export I_MPI_PIN_DOMAIN=omp

echo "date_start=$(date +%F_%T)"
echo "case=H_ghost"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "mpi_ranks=${SLURM_NTASKS:?}"
echo "omp_threads=$OMP_NUM_THREADS"
echo "expected_reader_ranks=$expected_reader_ranks"
echo "source_commit=$(cat "$script_dir/SOURCE_COMMIT")"
echo "abacus_source_commit=$abacus_source_commit"
echo "librpa_reference_commit=$librpa_reference_commit"
echo "version_exception=historical_exact_sos_binary_for_apples_to_apples"
sha256sum INPUT STRU KPT librpa.in "$orbital" "$pseudo" "$fixed_abs"

/usr/bin/time -v mpirun -np "$SLURM_NTASKS" -ppn 1 "$abacus" \
  > "abacus.${SLURM_JOB_ID}.out" 2>&1

grep -q '!FINAL_ETOT_IS' "$output_dir/running_scf.log"
grep -q 'rpa_lcao_exx(Ha):' "abacus.${SLURM_JOB_ID}.out"
test -s band_out
test -s basis_wfc_out
test -s basis_aux_out
test -s KS_eigenvector_0.dat
test "$(find . -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' | wc -l)" \
  -eq "$expected_reader_ranks"
test "$(find . -maxdepth 1 -type f -name 'v1_coulomb_full_iq_1_rank*.dat' | wc -l)" \
  -ge 1

awk '
  NR == 2 && $1 != 2 { exit 2 }
  NR == 3 && $1 != 334 { exit 3 }
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
    if (rows != 668) { exit 6 }
    diff = occ_sum - 1.0
    if (diff < 0) { diff = -diff }
    if (diff > 1.0e-8) { exit 7 }
  }
' band_out

/usr/bin/time -v mpirun -np 1 -ppn 1 "$librpa" \
  > "librpa.${SLURM_JOB_ID}.out" 2>&1

grep -q 'libRPA finished successfully' "librpa.${SLURM_JOB_ID}.out"
grep -q 'Total EcRPA:' "librpa.${SLURM_JOB_ID}.out"
sha256sum band_out basis_wfc_out basis_aux_out KS_eigenvector_0.dat \
  v1_Cs_data_*.txt v1_coulomb_full_iq_1_rank*.dat \
  > PRODUCTION_OUTPUTS.sha256
echo "date_end=$(date +%F_%T)"
