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
manifest_reader=$scripts/read_periodic_candidate_manifest.py
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
receipt=${run_root}.SUBMISSION.txt
lock=${receipt}.lock

test -d "$candidate_root" && test ! -e "$run_root" && test ! -e "$receipt" && test -d "$source_root"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test "${source_commit:0:8}" = "${source_root##*-}"
test -s "$candidate_root/CANDIDATE.json" && test -s "$manifest_reader"
grep -qx 'status=success' "$candidate_root/provenance.txt"
mkdir "$lock"
cleanup_lock() {
  rmdir "$lock" 2>/dev/null || true
}
trap cleanup_lock EXIT
mapfile -t candidate < <("$python" "$manifest_reader" "$candidate_root")
test "${#candidate[@]}" -eq 5
candidate_orbital_sha256=${candidate[2]}
for script in \
  run_relaxed_dzp_pbe_gate_55d25e3c9.slurm \
  run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm \
  run_threshold_candidate_solid_qstar_55d25e3c9.slurm \
  run_threshold_candidate_solid_qstar_sos_d4810f73.slurm \
  run_threshold_candidate_qstar_binding_collect.slurm; do
  test -s "$scripts/$script"
done

common=ALL,CANDIDATE_ROOT="$candidate_root",RUN_ROOT="$run_root",SIAB_SOURCE_ROOT="$source_root",SIAB_SOURCE_COMMIT="$source_commit"
atom_restart_source=$run_root/pbe-gate/atom/OUT.C_CANDIDATE_PBE_ATOM
for script in \
  run_relaxed_dzp_pbe_gate_55d25e3c9.slurm \
  run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm \
  run_threshold_candidate_solid_qstar_55d25e3c9.slurm \
  run_threshold_candidate_solid_qstar_sos_d4810f73.slurm \
  run_threshold_candidate_qstar_binding_collect.slurm; do
  sbatch --test-only --export="$common" "$scripts/$script" >/dev/null
done

pbe_job=$(sbatch --parsable --export="$common" "$scripts/run_relaxed_dzp_pbe_gate_55d25e3c9.slurm")
atom_job=$(sbatch --parsable --dependency="afterok:$pbe_job" \
  --export="$common,ATOM_RESTART_SOURCE=$atom_restart_source" \
  "$scripts/run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm")
qstar_job=$(sbatch --parsable --dependency="afterok:$pbe_job" --export="$common" "$scripts/run_threshold_candidate_solid_qstar_55d25e3c9.slurm")
solid_sos_job=$(sbatch --parsable --dependency="afterok:$qstar_job" \
  --export="$common,QSTAR_ARRAY_JOB_ID=$qstar_job" \
  "$scripts/run_threshold_candidate_solid_qstar_sos_d4810f73.slurm")
binding_job=$(sbatch --parsable --dependency="afterok:$atom_job:$solid_sos_job" \
  --export="$common,ATOM_JOB_ID=$atom_job,QSTAR_ARRAY_JOB_ID=$qstar_job,SOLID_SOS_JOB_ID=$solid_sos_job" \
  "$scripts/run_threshold_candidate_qstar_binding_collect.slurm")

printf 'status submitted\ncandidate_root %s\nrun_root %s\ncandidate_orbital_sha256 %s\npbe_job %s\natom_job %s\nqstar_array_job %s\nsolid_sos_job %s\nbinding_job %s\n' \
  "$candidate_root" "$run_root" "$candidate_orbital_sha256" \
  "$pbe_job" "$atom_job" "$qstar_job" "$solid_sos_job" "$binding_job" > "$receipt"
rmdir "$lock"
trap - EXIT
echo "C_RELAXED_DZP_PBE_QSTAR_CHAIN_SUBMITTED receipt=$receipt"
