#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=siab_h2_omp_ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

source_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_source_h2_v1_20260725
campaign_root=/work1/ghj/sternheimer_abacus_tests/siab_h2_target_openmp_ab_v1_20260725
asset_root=/work1/ghj/sternheimer_abacus_tests/h_atom_st_rpa_pca1e-4_rmesh5_20260704/st_atom_nspin1_gauss_abfs
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

abacus_work=/work1/ghj/sternheimer_abacus_tests/siab_openmp_red_v13_20260723/artifacts/f9c44f49e_job21356516/abacus_3p
abacus_source_commit=f9c44f49e0dedaeb6cb7cd0aa886cd924b280c9a
abacus_sha256=76e8ecfeba19a907507c24da5428d776bf34219f81319b574227f10772d96746
template_dir=$source_root/SIAB/example_H_sternheimer/greedy_response_selection/producer_h2
greedy=$source_root/SIAB/example_H_sternheimer/greedy_response_selection

thread_counts=(16 24)
index=${SLURM_ARRAY_TASK_ID:?}
threads=${thread_counts[$index]}
suffix=H2_SIAB_OMP${threads}_NFREQ1_E50
output_dir=$campaign_root/omp${threads}

test -x "$abacus_work"
test "$abacus_source_commit" = f9c44f49e0dedaeb6cb7cd0aa886cd924b280c9a
test "$abacus_sha256" = 76e8ecfeba19a907507c24da5428d776bf34219f81319b574227f10772d96746
for required in INPUT STRU KPT; do
  test -s "$template_dir/$required"
done
test ! -e "$output_dir"
mkdir -p "$output_dir"
cp "$template_dir/STRU" "$template_dir/KPT" "$output_dir/"
cp "$asset_root/H_ONCV_PBE-1.0.upf" "$output_dir/"
cp "$asset_root/H_gga_8au_100Ry_3s2p.orb" "$output_dir/"
cp "$asset_root/H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs" "$output_dir/"

# This benchmark retains the production H2 operator (ecutwfc 50 and
# global_full_coulomb whitening), but uses sternheimer_nfreq 1 at the first
# fixed minimax frequency.
sed \
  -e "s/^suffix .*/suffix                  $suffix/" \
  -e 's/^sternheimer_nfreq .*/sternheimer_nfreq       1/' \
  -e 's@^sternheimer_frequency_grid_file .*@sternheimer_frequency_grid_file fixed_frequency_grid_nfreq1_first_from_sos20A.dat@' \
  -e 's/^sternheimer_frequency_mpi .*/sternheimer_frequency_mpi 0/' \
  "$template_dir/INPUT" > "$output_dir/INPUT"
sed -n '1,2p' "$greedy/fixed_frequency_grid_nfreq16_from_sos20A.dat" \
  > "$output_dir/fixed_frequency_grid_nfreq1_first_from_sos20A.dat"
grep -Eq '^ecutwfc[[:space:]]+50$' "$output_dir/INPUT"
grep -Eq '^sternheimer_nfreq[[:space:]]+1$' "$output_dir/INPUT"

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export MKL_DYNAMIC=FALSE
export ABACUS_STERNHEIMER_FD_ST_ORBITAL_DIR="$output_dir/"
export ABACUS_STERNHEIMER_FD_ST_FREQ_RANK_SHIFT=1
export ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-8
export ABACUS_STERNHEIMER_FD_ST_SOLVER_MAX_ITER=300

cd "$output_dir"
echo "date_start=$(date +%F_%T)"
echo "nodes=${SLURM_JOB_NODELIST:?}"
echo "mpi_ranks=${SLURM_NTASKS:?}"
echo "omp_threads=$OMP_NUM_THREADS"
echo "abacus_source_commit=$abacus_source_commit"
actual_sha256=$(sha256sum "$abacus_work" | awk '{print $1}')
echo "abacus_sha256=$actual_sha256"
test "$actual_sha256" = "$abacus_sha256"
test "$(sha256sum H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs | awk '{print $1}')" = \
  d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f
sha256sum INPUT STRU KPT fixed_frequency_grid_nfreq1_first_from_sos20A.dat \
  H_ONCV_PBE-1.0.upf H_gga_8au_100Ry_3s2p.orb \
  H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs > input_sha256sums.txt

/usr/bin/time -v mpirun -np 1 -ppn 1 "$abacus_work" \
  > time_and_stderr.log 2>&1

result_dir=$output_dir/OUT.$suffix
test -s "$result_dir/sternheimer_matrix.dat"
test -s "$result_dir/STERNHEIMER_SIAB_STATUS.dat"
test -s "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat"
test -s "$output_dir/STERNHEIMER_SIAB_COULOMB_WHITENING.dat"
grep -q '^status success$' "$result_dir/STERNHEIMER_SIAB_STATUS.dat"
grep -q '^format siab_v1$' "$result_dir/STERNHEIMER_SIAB_STATUS.dat"
grep -q '^abfs_source explicit_abfs$' "$result_dir/STERNHEIMER_SIAB_STATUS.dat"
grep -q '^abfs_channels 428$' "$result_dir/STERNHEIMER_SIAB_STATUS.dat"

export PYTHONPATH=$source_root/SIAB/opt_orb_pytorch_dpsi
"$python" "$greedy/validate_targets.py" \
  --target "$result_dir/sternheimer_matrix.dat" \
  --channels "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat" \
  --status "$result_dir/STERNHEIMER_SIAB_STATUS.dat" \
  --expected-atoms 2 \
  --primitive-lmax 4 \
  --radial-primitives 25 \
  --expected-nfreq 1 \
  --expected-auxiliary-sha256 d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f \
  > target_validation.json

sha256sum "$result_dir/sternheimer_matrix.dat" \
  "$result_dir/STERNHEIMER_SIAB_STATUS.dat" \
  "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat" \
  "$output_dir/STERNHEIMER_SIAB_COULOMB_WHITENING.dat" \
  target_validation.json > output_sha256sums.txt
echo "comparison_command=$greedy/compare_sternheimer_targets.py"
echo "date_end=$(date +%F_%T)"
