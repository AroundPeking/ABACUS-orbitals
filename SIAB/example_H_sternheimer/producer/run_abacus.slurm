#!/bin/bash
#SBATCH -J h_siab_st
#SBATCH -p normal
#SBATCH -N 16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110610M
#SBATCH -t 1-00:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-30}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,scatter
export ABACUS_STERNHEIMER_FD_ST_ORBITAL_DIR="$PWD/"
export ABACUS_STERNHEIMER_FD_ST_ORBITAL_FILES=H_gga_8au_100Ry_3s2p.orb
export ABACUS_STERNHEIMER_FD_ST_FREQ_RANK_SHIFT=1
export ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-8
export ABACUS_STERNHEIMER_FD_ST_SOLVER_MAX_ITER=300

abacus_work=/public/home/ghj/app/src/abacus-sternheimer-siab-task4-80a606f57a26/build-task4-o3-debug-info/abacus-80a606f57a26
abacus_sha256=2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d
output_dir=OUT.H_SIAB_ST_ECUT50_NFREQ16_PCA1EM6_TASK4_80A606F

echo "date_start=$(date +%F_%T)"
echo "pwd=$PWD"
echo "nodes=$SLURM_JOB_NODELIST"
echo "cpus=$SLURM_JOB_CPUS_PER_NODE"
echo "mpi_ranks=${SLURM_NTASKS:-16}"
echo "omp_threads=$OMP_NUM_THREADS"
echo "abacus=$abacus_work"
stat -c "abacus_stat=%y %s %i %a %n" "$abacus_work"
actual_sha256=$(sha256sum "$abacus_work" | awk '{print $1}')
echo "abacus_sha256=$actual_sha256"
test "$actual_sha256" = "$abacus_sha256"
sha256sum INPUT STRU KPT fixed_frequency_grid_nfreq16_from_sos20A.dat \
  H.upf H_gga_8au_100Ry_3s2p.orb

/usr/bin/time -p mpirun -np "${SLURM_NTASKS:-16}" -ppn 1 "$abacus_work" \
  > "abacus.${SLURM_JOB_ID}.out" 2>&1

test "$(sha256sum "$abacus_work" | awk '{print $1}')" = "$abacus_sha256"
test -s "$output_dir/sternheimer_matrix.dat"
grep -q "Sternheimer SIAB v1 output:" "$output_dir/running_scf.log"
echo "target_sha256=$(sha256sum "$output_dir/sternheimer_matrix.dat" | awk '{print $1}')"
echo "target_bytes=$(stat -c %s "$output_dir/sternheimer_matrix.dat")"
echo "date_end=$(date +%F_%T)"
