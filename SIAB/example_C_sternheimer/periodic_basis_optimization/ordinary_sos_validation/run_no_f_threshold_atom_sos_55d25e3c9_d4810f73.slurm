#!/bin/bash
#SBATCH --job-name=c_nof_atom
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-atom-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-atom-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NO_F_ATOM_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-ce2801f3
candidate_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-candidate-570a6d79
run_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-threshold-q12-control
base=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm
patched=$run_root/patched-atom.sh

test -s "$base"
test ! -e "$patched"
sed -E \
  -e 's/payload\["output_nu"\] == \[3, 3, 2, 1\]/payload["output_nu"] == [3, 3, 2]/' \
  -e 's/payload\["output_nao"\] == 29/payload["output_nao"] == 22/' \
  -e 's/set_input_key nbands 29/set_input_key nbands 22/' \
  -e 's/set_input_key ocp_set "3\*1 26\*0 1\*1 28\*0"/set_input_key ocp_set "3*1 19*0 1*1 21*0"/' \
  -e 's/len\(values\) == 29/len(values) == 22/' \
  -e 's/orbital_layout=3s3p2d1f/orbital_layout=3s3p2d/' \
  "$base" > "$patched"

grep -q 'payload\["output_nao"\] == 22' "$patched"
grep -q 'set_input_key nbands 22' "$patched"
grep -q 'set_input_key ocp_set "3\*1 19\*0 1\*1 21\*0"' "$patched"
grep -q 'len(values) == 22' "$patched"
env CANDIDATE_ROOT="$candidate_root" RUN_ROOT="$run_root" SIAB_SOURCE_ROOT="$source_root" bash "$patched"

echo "C_NO_F_ATOM_OK"
