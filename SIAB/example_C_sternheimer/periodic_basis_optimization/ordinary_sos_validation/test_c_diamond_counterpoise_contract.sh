#!/bin/bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
job=$root/run_c_diamond_first_shell_counterpoise_d4810f73.slurm
submitter=$root/submit_c_diamond_first_shell_counterpoise.sh

test -s "$job"
test -s "$submitter"
grep -q '^#SBATCH --job-name=c_bsse_nn1$' "$job"
grep -q '^#SBATCH --partition=p1$' "$job"
grep -q '^#SBATCH --nodes=1$' "$job"
grep -q '^#SBATCH --ntasks-per-node=1$' "$job"
grep -q '^#SBATCH --cpus-per-task=40$' "$job"
grep -q '^#SBATCH --mem=190000$' "$job"
grep -q 'shell-count 1' "$job"
grep -q 'set_input_key ntype 2' "$job"
grep -q 'set_input_key nelec 4' "$job"
grep -q 'set_input_key nspin 2' "$job"
grep -q 'set_input_key nupdown 2' "$job"
grep -q 'nbands=$((ao_per_atom \* total_sites))' "$job"
grep -q 'set_input_key exx_pca_threshold 1e-4' "$job"
grep -q 'set_input_key rpa_ccp_rmesh_times 1' "$job"
grep -q 'n_bands_chi0 = -1' "$job"
grep -q 'use_fullcoul_exx = t' "$job"
grep -q 'librpa_commit=d4810f73' "$job"
grep -q 'MKL_NUM_THREADS=$OMP_NUM_THREADS' "$job"
grep -q '"$counterpoise" collect' "$job"
grep -q -- '--raw-binding "$raw_binding"' "$job"
grep -q 'counterpoise_acceptance' "$job"
grep -q 'squeue -h -u "$USER" -n c_bsse_nn1' "$submitter"
grep -q 'sbatch --parsable' "$submitter"

echo success
