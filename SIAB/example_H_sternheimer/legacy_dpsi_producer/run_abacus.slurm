#!/bin/bash
#SBATCH -J h3_dpsi
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=30
#SBATCH --ntasks-per-node=30
#SBATCH --cpus-per-task=1
#SBATCH --mem=110610M
#SBATCH -t 1-00:00:00
#SBATCH --array=0-2
#SBATCH -o slurm.%A_%a.out
#SBATCH -e slurm.%A_%a.err

set -euo pipefail
source /public/home/ghj/app/src/env_60_245_intel2021.sh

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) case_name=H-STRU2-8-0.7 ;;
  1) case_name=H-STRU2-8-0.9 ;;
  2) case_name=H-STRU2-8-1.3 ;;
  *) echo "invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

script_dir=${SLURM_SUBMIT_DIR:?}
case_dir="$script_dir/cases/$case_name"
output_dir="OUT.$case_name"
abacus_work=/public/home/ghj/app/src/abacus-sternheimer-siab-task4-80a606f57a26/build-task4-o3-debug-info/abacus-80a606f57a26
abacus_sha256=2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d
h_upf_sha256=38add5ec2a81b907d7ae0891de1d8353c5d60f4b53b27ed854f05c2a955d7ddf

test -d "$case_dir"
test -s "$script_dir/H.upf"
test "$(sha256sum "$abacus_work" | awk '{print $1}')" = "$abacus_sha256"
test "$(sha256sum "$script_dir/H.upf" | awk '{print $1}')" = "$h_upf_sha256"
test ! -e "$case_dir/$output_dir"

cp "$script_dir/KPT" "$case_dir/KPT"
cp "$script_dir/H.upf" "$case_dir/H.upf"
cd "$case_dir"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

echo "date_start=$(date +%F_%T)"
echo "pwd=$PWD"
echo "nodes=$SLURM_JOB_NODELIST"
echo "mpi_ranks=${SLURM_NTASKS:-30}"
echo "omp_threads=$OMP_NUM_THREADS"
echo "abacus=$abacus_work"
sha256sum INPUT INPUTw STRU KPT H.upf

/usr/bin/time -p mpirun -np "${SLURM_NTASKS:-30}" -ppn 30 "$abacus_work" \
  > "abacus.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

test "$(sha256sum "$abacus_work" | awk '{print $1}')" = "$abacus_sha256"
test -s "$output_dir/orb_matrix.0.dat"
test -s "$output_dir/orb_matrix.1.dat"
grep -q '</OVERLAP_Q>' "$output_dir/orb_matrix.0.dat"
grep -q '</OVERLAP_Q>' "$output_dir/orb_matrix.1.dat"
grep -q '</OVERLAP_Sq>' "$output_dir/orb_matrix.0.dat"
grep -q '</OVERLAP_Sq>' "$output_dir/orb_matrix.1.dat"
grep -q '!FINAL_ETOT_IS' "$output_dir/running_scf.log"
sha256sum "$output_dir/orb_matrix.0.dat" "$output_dir/orb_matrix.1.dat"
echo "date_end=$(date +%F_%T)"
