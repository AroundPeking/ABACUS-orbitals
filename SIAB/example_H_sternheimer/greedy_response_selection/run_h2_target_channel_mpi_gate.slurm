#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=siab_h2_chmpi_gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH --time=1-00:00:00
#SBATCH --output=/work1/ghj/sternheimer_abacus_tests/siab_h2_channel_mpi_equivalence_e10_v1_20260726/logs/%x-%j.out

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

mode=${SIAB_CHANNEL_MPI_MODE:?}
# Accepted modes: serial|channel_mpi.
case "$mode" in
  serial)
    expected_mpi_ranks=1
    frequency_mpi=0
    channel_mpi=0
    ;;
  channel_mpi)
    expected_mpi_ranks=2
    frequency_mpi=1
    channel_mpi=1
    ;;
  *)
    echo "SIAB_CHANNEL_MPI_MODE must be serial or channel_mpi" >&2
    exit 2
    ;;
esac

source_root=/work1/ghj/sternheimer_abacus_tests/siab_greedy_targets_source_h2_channel_mpi_v1_20260726
campaign_root=/work1/ghj/sternheimer_abacus_tests/siab_h2_channel_mpi_equivalence_e10_v1_20260726
asset_root=/work1/ghj/sternheimer_abacus_tests/h_atom_st_rpa_pca1e-4_rmesh5_20260704/st_atom_nspin1_gauss_abfs
python=/work1/ghj/runtime/siab-py310-cpu-20260720/bin/python

abacus_work=/work1/ghj/sternheimer_abacus_tests/siab_channel_mpi_exact_c273b4ee7_20260726/artifacts/job21389808/abacus_3p
abacus_source_commit=c273b4ee7051138293d9988c3eb79bee36c0af10
abacus_sha256=ff38348fbad89fde4a985c13f97b59ffc94353c22c7098e19b373c1ef7e76fee
template_dir=$source_root/SIAB/example_H_sternheimer/greedy_response_selection/producer_h2
greedy=$source_root/SIAB/example_H_sternheimer/greedy_response_selection
output_dir=$campaign_root/$mode
suffix=H2_SIAB_CHANNEL_MPI_GATE_${mode^^}_NFREQ1_E10

test "${SLURM_NTASKS:?}" -eq "$expected_mpi_ranks"
test -x "$abacus_work"
test "$(sha256sum "$abacus_work" | awk '{print $1}')" = "$abacus_sha256"
for required in INPUT STRU KPT; do
  test -s "$template_dir/$required"
done
test ! -e "$output_dir"
mkdir -p "$output_dir"
cp "$template_dir/STRU" "$template_dir/KPT" "$output_dir/"
cp "$asset_root/H_ONCV_PBE-1.0.upf" "$output_dir/"
cp "$asset_root/H_gga_8au_100Ry_3s2p.orb" "$output_dir/"
cp "$asset_root/H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs" "$output_dir/"

sed \
  -e "s/^suffix .*/suffix                  $suffix/" \
  -e 's/^ecutwfc .*/ecutwfc 10/' \
  -e 's/^sternheimer_nfreq .*/sternheimer_nfreq 1/' \
  -e 's@^sternheimer_frequency_grid_file .*@sternheimer_frequency_grid_file fixed_frequency_grid_nfreq1_first_from_sos20A.dat@' \
  -e "s/^sternheimer_frequency_mpi .*/sternheimer_frequency_mpi $frequency_mpi/" \
  -e "s/^sternheimer_channel_mpi .*/sternheimer_channel_mpi $channel_mpi/" \
  "$template_dir/INPUT" > "$output_dir/INPUT"
sed -n '1,2p' "$greedy/fixed_frequency_grid_nfreq16_from_sos20A.dat" \
  > "$output_dir/fixed_frequency_grid_nfreq1_first_from_sos20A.dat"

grep -Eq '^ecutwfc[[:space:]]+10$' "$output_dir/INPUT"
grep -Eq '^sternheimer_nfreq[[:space:]]+1$' "$output_dir/INPUT"
grep -Eq "^sternheimer_frequency_mpi[[:space:]]+$frequency_mpi$" "$output_dir/INPUT"
grep -Eq "^sternheimer_channel_mpi[[:space:]]+$channel_mpi$" "$output_dir/INPUT"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
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
echo "mpi_ranks=$SLURM_NTASKS"
echo "omp_threads=$OMP_NUM_THREADS"
echo "abacus_source_commit=$abacus_source_commit"
echo "abacus_sha256=$abacus_sha256"
sha256sum INPUT STRU KPT fixed_frequency_grid_nfreq1_first_from_sos20A.dat \
  H_ONCV_PBE-1.0.upf H_gga_8au_100Ry_3s2p.orb \
  H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs > input_sha256sums.txt

/usr/bin/time -v mpirun -np "$SLURM_NTASKS" -ppn 1 "$abacus_work" \
  > time_and_stderr.log 2>&1

result_dir=$output_dir/OUT.$suffix
status_file=$result_dir/STERNHEIMER_SIAB_STATUS.dat
test -s "$result_dir/sternheimer_matrix.dat"
test -s "$status_file"
test -s "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat"
test -s "$output_dir/STERNHEIMER_SIAB_COULOMB_WHITENING.dat"
grep -q '^status success$' "$status_file"
grep -q '^format siab_v1$' "$status_file"
grep -q '^abfs_source explicit_abfs$' "$status_file"
grep -q '^abfs_channels 428$' "$status_file"
grep -q '^solved_equations 428$' "$status_file"
grep -q "^sternheimer_frequency_mpi $(test "$frequency_mpi" -eq 1 && echo yes || echo no)$" "$status_file"
grep -q "^sternheimer_channel_mpi $(test "$channel_mpi" -eq 1 && echo yes || echo no)$" "$status_file"
grep -q "^frequency_group_size $expected_mpi_ranks$" "$status_file"
grep -q "^mpi_ranks $expected_mpi_ranks$" "$status_file"

export PYTHONPATH=$source_root/SIAB/opt_orb_pytorch_dpsi
"$python" "$greedy/validate_targets.py" \
  --target "$result_dir/sternheimer_matrix.dat" \
  --channels "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat" \
  --status "$status_file" \
  --expected-atoms 2 \
  --primitive-lmax 4 \
  --radial-primitives 25 \
  --expected-nfreq 1 \
  --expected-auxiliary-sha256 d5d12b2eb09716803784418848c9cec9ea5633069b5c014e0f4399eeaa9b106f \
  > target_validation.json

sha256sum "$result_dir/sternheimer_matrix.dat" "$status_file" \
  "$output_dir/STERNHEIMER_ABFS_CHANNELS.dat" \
  "$output_dir/STERNHEIMER_SIAB_COULOMB_WHITENING.dat" \
  target_validation.json > output_sha256sums.txt
echo "date_end=$(date +%F_%T)"
