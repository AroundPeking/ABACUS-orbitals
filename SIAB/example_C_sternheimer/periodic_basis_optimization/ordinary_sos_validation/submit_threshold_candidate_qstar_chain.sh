#!/bin/bash

set -euo pipefail

if test "$#" -ne 3; then
  echo "usage: $0 CANDIDATE_ROOT RUN_ROOT SIAB_SOURCE_ROOT" >&2
  exit 2
fi
candidate_root=$1
run_root=$2
source_root=$3
source_commit=${SIAB_SOURCE_COMMIT:?missing exact SIAB source commit}
scripts=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation

test -d "$candidate_root" && test ! -e "$run_root" && test -d "$source_root"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test "${source_commit:0:8}" = "${source_root##*-}"
test -s "$candidate_root/TRUNCATION.json" || test -s "$candidate_root/CANDIDATE.json"
grep -qx 'status=success' "$candidate_root/provenance.txt"
for script in \
  run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm \
  run_threshold_candidate_solid_qstar_55d25e3c9.slurm \
  run_threshold_candidate_solid_qstar_sos_d4810f73.slurm \
  run_threshold_candidate_qstar_binding_collect.slurm; do
  test -s "$scripts/$script"
done
test -z "$(squeue -h -u "$USER" -n c_thr_atom_sos,c_thr_qstar,c_thr_qsos,c_thr_bind -o %A)"

common=ALL,CANDIDATE_ROOT="$candidate_root",RUN_ROOT="$run_root",SIAB_SOURCE_ROOT="$source_root",SIAB_SOURCE_COMMIT="$source_commit"
atom_job=$(sbatch --parsable --export="$common" "$scripts/run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm")
qstar_job=$(sbatch --parsable --export="$common" "$scripts/run_threshold_candidate_solid_qstar_55d25e3c9.slurm")
solid_sos_job=$(sbatch --parsable --dependency="afterok:$qstar_job" \
  --export="$common,QSTAR_ARRAY_JOB_ID=$qstar_job" \
  "$scripts/run_threshold_candidate_solid_qstar_sos_d4810f73.slurm")
binding_job=$(sbatch --parsable --dependency="afterok:$atom_job:$solid_sos_job" \
  --export="$common,ATOM_JOB_ID=$atom_job,QSTAR_ARRAY_JOB_ID=$qstar_job,SOLID_SOS_JOB_ID=$solid_sos_job" \
  "$scripts/run_threshold_candidate_qstar_binding_collect.slurm")

printf 'atom_job=%s\nqstar_array_job=%s\nsolid_sos_job=%s\nbinding_job=%s\n' \
  "$atom_job" "$qstar_job" "$solid_sos_job" "$binding_job"
