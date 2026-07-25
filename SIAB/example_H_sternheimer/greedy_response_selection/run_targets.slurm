#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=siab_targets
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

source_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_source_h2_v1_20260725
campaign_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_h2_whitening_v1_20260725
asset_root=/work1/ghj/sternheimer_abacus_tests/h_atom_st_rpa_pca1e-4_rmesh5_20260704/st_atom_nspin1_gauss_abfs
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

abacus_work=/work1/ghj/sternheimer_abacus_tests/siab_openmp_red_v13_20260723/artifacts/f9c44f49e_job21356516/abacus_3p
abacus_source_commit=f9c44f49e0dedaeb6cb7cd0aa886cd924b280c9a
abacus_sha256=76e8ecfeba19a907507c24da5428d776bf34219f81319b574227f10772d96746

case_names=(producer_atom producer_h2 producer_h2_fragment_ghost)
suffixes=(
  H_SIAB_GREEDY_ATOM_WFULL_NF16_E50
  H2_SIAB_GREEDY_R074085_WFULL_NF16_E50
  H_GHOST1_SIAB_GREEDY_R074085_WFULL_NF16_E50
)
expected_atoms=(1 2 2)
expected_auxiliary_channels=(214 428 428)
expected_primitive_columns=(625 1250 1250)

index=${SLURM_ARRAY_TASK_ID:?}
case_name=${case_names[$index]}
suffix=${suffixes[$index]}
nat=${expected_atoms[$index]}
raw_naux=${expected_auxiliary_channels[$index]}
primitive_columns=${expected_primitive_columns[$index]}
template_dir=$source_root/SIAB/example_H_sternheimer/greedy_response_selection/$case_name
output_dir=$campaign_root/$case_name

test -x "$abacus_work"
test "$abacus_source_commit" = f9c44f49e0dedaeb6cb7cd0aa886cd924b280c9a
test "$abacus_sha256" = 76e8ecfeba19a907507c24da5428d776bf34219f81319b574227f10772d96746
for required in INPUT STRU KPT; do
  test -s "$template_dir/$required"
done
test ! -e "$output_dir"
mkdir -p "$output_dir"
cp "$template_dir/INPUT" "$template_dir/STRU" "$template_dir/KPT" "$output_dir/"
cp "$source_root/SIAB/example_H_sternheimer/greedy_response_selection/fixed_frequency_grid_nfreq16_from_sos20A.dat" "$output_dir/"
cp "$asset_root/H_ONCV_PBE-1.0.upf" "$output_dir/"
cp "$asset_root/H_gga_8au_100Ry_3s2p.orb" "$output_dir/"
cp "$asset_root/H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs" "$output_dir/"

threads=${ABACUS_STERNHEIMER_THREADS:-16}
case "$threads" in
  1|16|24|30) ;;
  *) echo "Unsupported ABACUS_STERNHEIMER_THREADS=$threads" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export MKL_DYNAMIC=FALSE
export ABACUS_STERNHEIMER_FD_ST_ORBITAL_DIR="$output_dir/"
export ABACUS_STERNHEIMER_FD_ST_FREQ_RANK_SHIFT=1
export ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-8
export ABACUS_STERNHEIMER_FD_ST_SOLVER_MAX_ITER=300

cd "$output_dir"
echo "date_start=$(date +%F_%T)"
echo "nodes=$SLURM_JOB_NODELIST"
echo "mpi_ranks=${SLURM_NTASKS:-16}"
echo "omp_threads=$OMP_NUM_THREADS"
echo "abacus_source_commit=$abacus_source_commit"
actual_sha256=$(sha256sum "$abacus_work" | awk '{print $1}')
echo "abacus_sha256=$actual_sha256"
test "$actual_sha256" = "$abacus_sha256"
test "$(sha256sum H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs | awk '{print $1}')" = \
  d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f
sha256sum INPUT STRU KPT fixed_frequency_grid_nfreq16_from_sos20A.dat \
  H_ONCV_PBE-1.0.upf H_gga_8au_100Ry_3s2p.orb \
  H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs

/usr/bin/time -v mpirun -np 16 -ppn 1 "$abacus_work" \
  > "abacus.${SLURM_JOB_ID}_${index}.out" 2>&1

result_dir=$output_dir/OUT.$suffix
test -s "$result_dir/sternheimer_matrix.dat"
test -s "$result_dir/STERNHEIMER_SIAB_STATUS.dat"
test -s "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat"
test -s "$output_dir/STERNHEIMER_SIAB_COULOMB_WHITENING.dat"
grep -q '^status success$' "$result_dir/STERNHEIMER_SIAB_STATUS.dat"
grep -q '^format siab_v1$' "$result_dir/STERNHEIMER_SIAB_STATUS.dat"
grep -q '^abfs_source explicit_abfs$' "$result_dir/STERNHEIMER_SIAB_STATUS.dat"
grep -q "^abfs_channels $raw_naux$" "$result_dir/STERNHEIMER_SIAB_STATUS.dat"

export PYTHONPATH=$source_root/SIAB/opt_orb_pytorch_dpsi
"$python" "$source_root/SIAB/example_H_sternheimer/greedy_response_selection/validate_targets.py" \
  --target "$result_dir/sternheimer_matrix.dat" \
  --channels "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat" \
  --status "$result_dir/STERNHEIMER_SIAB_STATUS.dat" \
  --expected-atoms "$nat" \
  --primitive-lmax 4 \
  --radial-primitives 25 \
  --expected-nfreq 16 \
  --expected-auxiliary-sha256 d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f \
  > "$output_dir/target_validation.json"

test "$(sha256sum "$abacus_work" | awk '{print $1}')" = "$abacus_sha256"
sha256sum "$result_dir/sternheimer_matrix.dat" \
  "$result_dir/STERNHEIMER_SIAB_STATUS.dat" \
  "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat" \
  "$output_dir/STERNHEIMER_SIAB_COULOMB_WHITENING.dat" \
  "$output_dir/target_validation.json"
echo "expected_primitive_columns=$primitive_columns"
echo "date_end=$(date +%F_%T)"
