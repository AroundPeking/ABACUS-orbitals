#!/bin/bash

set -euo pipefail

if test "$#" -ne 3; then
  echo "usage: $0 CANDIDATE_ROOT RUN_ROOT SIAB_SOURCE_ROOT" >&2
  exit 2
fi
candidate_root=$1
run_root=$2
source_root=$3
scripts=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
receipt=${run_root}.SUBMISSION.txt

test -d "$candidate_root" && test ! -e "$run_root" && test ! -e "$receipt" && test -d "$source_root"
test -s "$candidate_root/CANDIDATE.json"
grep -q '"pre_pbe_gate": "pass"' "$candidate_root/CANDIDATE.json"
for script in \
  run_relaxed_dzp_pbe_gate_55d25e3c9.slurm \
  run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm \
  run_threshold_candidate_solid_qstar_55d25e3c9.slurm \
  run_threshold_candidate_solid_qstar_sos_d4810f73.slurm \
  run_threshold_candidate_qstar_binding_collect.slurm; do
  test -s "$scripts/$script"
done
test -z "$(squeue -h -u "$USER" -n c_relaxed_pbe,c_thr_atom_sos,c_thr_qstar,c_thr_qsos,c_thr_bind -o %A)"

common=ALL,CANDIDATE_ROOT="$candidate_root",RUN_ROOT="$run_root",SIAB_SOURCE_ROOT="$source_root"
for script in \
  run_relaxed_dzp_pbe_gate_55d25e3c9.slurm \
  run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm \
  run_threshold_candidate_solid_qstar_55d25e3c9.slurm \
  run_threshold_candidate_solid_qstar_sos_d4810f73.slurm \
  run_threshold_candidate_qstar_binding_collect.slurm; do
  sbatch --test-only --export="$common" "$scripts/$script" >/dev/null
done

pbe_job=$(sbatch --parsable --export="$common" "$scripts/run_relaxed_dzp_pbe_gate_55d25e3c9.slurm")
atom_job=$(sbatch --parsable --dependency="afterok:$pbe_job" --export="$common" "$scripts/run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm")
qstar_job=$(sbatch --parsable --dependency="afterok:$pbe_job" --export="$common" "$scripts/run_threshold_candidate_solid_qstar_55d25e3c9.slurm")
solid_sos_job=$(sbatch --parsable --dependency="afterok:$qstar_job" \
  --export="$common,QSTAR_ARRAY_JOB_ID=$qstar_job" \
  "$scripts/run_threshold_candidate_solid_qstar_sos_d4810f73.slurm")
binding_job=$(sbatch --parsable --dependency="afterok:$atom_job:$solid_sos_job" \
  --export="$common,ATOM_JOB_ID=$atom_job,QSTAR_ARRAY_JOB_ID=$qstar_job,SOLID_SOS_JOB_ID=$solid_sos_job" \
  "$scripts/run_threshold_candidate_qstar_binding_collect.slurm")

printf 'status submitted\npbe_job %s\natom_job %s\nqstar_array_job %s\nsolid_sos_job %s\nbinding_job %s\n' \
  "$pbe_job" "$atom_job" "$qstar_job" "$solid_sos_job" "$binding_job" > "$receipt"
echo "C_RELAXED_DZP_PBE_QSTAR_CHAIN_SUBMITTED receipt=$receipt"
