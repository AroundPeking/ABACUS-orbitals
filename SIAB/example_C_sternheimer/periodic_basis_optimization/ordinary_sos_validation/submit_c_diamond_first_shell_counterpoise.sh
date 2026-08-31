#!/bin/bash
set -euo pipefail

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
raw_root=${RAW_BINDING_ROOT:?missing completed raw atom-solid binding root}
run_root=${RUN_ROOT:?missing immutable counterpoise run root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
script=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_c_diamond_first_shell_counterpoise_d4810f73.slurm

test -s "$candidate_root/CANDIDATE.json"
test -s "$raw_root/binding/RESULT.json"
grep -qx success "$raw_root/binding/STATUS"
test ! -e "$run_root"
test -z "$(squeue -h -u "$USER" -n c_bsse_nn1 -o %A)"

job=$(sbatch --parsable \
  --export=ALL,CANDIDATE_ROOT="$candidate_root",RAW_BINDING_ROOT="$raw_root",RUN_ROOT="$run_root",SIAB_SOURCE_ROOT="$source_root" \
  "$script")
printf 'job_id=%s\nrun_root=%s\n' "$job" "$run_root"
