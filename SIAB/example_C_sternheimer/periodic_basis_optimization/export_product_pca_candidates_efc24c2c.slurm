#!/bin/bash
#SBATCH --job-name=c24_pca_export
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --array=2,3
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-export-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-pca-export-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C24_PRODUCT_PCA_EXPORT_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load python/3.9.22

runtime=/data/home/df_iopcas_ghj/app/python/siab-torch19-py39
code=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
siab_commit=d5d799cfa2003fee79376d0578cb826a3c31030a
exporter=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/export_periodic_orbitals.py
ng=${SLURM_ARRAY_TASK_ID:?missing g multiplicity}

case "$ng" in
  2) nu=3,3,2,1,2 ;;
  3) nu=3,3,2,1,3 ;;
  *) echo "unsupported g multiplicity: $ng" >&2; exit 2 ;;
esac

input=$code/runs/product-pca-20260825/joint-${ng}g-nfreq6-q1-q2/result/ORBITAL_RESULTS.txt
output=$code/runs/product-pca-20260825/export-joint-${ng}g
orbital=$output/C_gga_10au_100Ry_3s3p2d1f${ng}g_product_pca.orb

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test "$(cat "$code/.git/HEAD")" = "$siab_commit"
test -s "$input"
test -s "$exporter"
test ! -e "$output"

export PYTHONPATH=$runtime/lib-dynload:$runtime/site-packages:$code
export LD_LIBRARY_PATH=$runtime/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p "$output"
/usr/bin/time -v -o "$output/export.time" \
  python3 "$exporter" \
    --input "$input" \
    --output "$orbital" \
    --element C \
    --nu "$nu" \
    --radial-rows 31 \
    --ecut-ry 100 \
    --rcut-bohr 10 \
    --dr-bohr 0.01 \
    --smoothing-sigma-bohr 0.1 \
    > "$output/EXPORT_METADATA.json"

test -s "$orbital"
grep -qx 'Energy Cutoff(Ry)           100.0' "$orbital"
grep -qx 'Radius Cutoff(a.u.)         10.0' "$orbital"
grep -qx 'Lmax                        4' "$orbital"
grep -qx 'Number of Sorbital-->       3' "$orbital"
grep -qx 'Number of Porbital-->       3' "$orbital"
grep -qx 'Number of Dorbital-->       2' "$orbital"
grep -qx 'Number of Forbital-->       1' "$orbital"
grep -qx "Number of Gorbital-->       $ng" "$orbital"

{
  echo status=success
  echo purpose=periodic_c_product_pca_joint_candidate_export
  echo siab_commit=$siab_commit
  echo optimizer_job=${OPTIMIZER_ARRAY_JOB_ID:?missing optimizer array job id}_${ng}
  echo heldout_gate_job=${HELDOUT_JOB_ID:?missing heldout job id}
  echo basis=3s3p2d1f${ng}g
  echo ecut_ry=100
  echo rcut_bohr=10
  echo dr_bohr=0.01
  echo smoothing_sigma_bohr=0.1
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' "$output/export.time"
  sha256sum "$0" "$input" "$exporter" "$output/EXPORT_METADATA.json" "$orbital"
} > "$output/provenance.txt"

echo "C24_PRODUCT_PCA_EXPORT_OK g=$ng orbital=$orbital"
