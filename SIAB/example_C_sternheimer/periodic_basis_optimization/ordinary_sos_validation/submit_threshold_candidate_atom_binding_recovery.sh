#!/bin/bash

set -euo pipefail

if test "$#" -ne 6; then
  echo "usage: $0 CANDIDATE_ROOT RUN_ROOT SIAB_SOURCE_ROOT FAILED_ATOM_JOB_ID QSTAR_ARRAY_JOB_ID SOLID_SOS_JOB_ID" >&2
  exit 2
fi
candidate_root=$1
run_root=$2
source_root=$3
failed_atom_job_id=$4
qstar_array_job_id=$5
solid_sos_job_id=$6
source_commit=${SIAB_SOURCE_COMMIT:?missing exact SIAB source commit}
scripts=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
original_receipt=${run_root}.SUBMISSION.txt
receipt=${run_root}.ATOM_RECOVERY.txt
lock=${receipt}.lock
atom_restart_source=$run_root/pbe-gate/atom/OUT.C_CANDIDATE_PBE_ATOM
atom_recovery_root=$run_root/atom-recovery-${source_commit:0:8}

test -d "$candidate_root" && test -d "$run_root" && test -d "$source_root"
test -s "$candidate_root/CANDIDATE.json"
grep -qx 'status=success' "$candidate_root/provenance.txt"
grep -qx 'success' "$run_root/pbe-gate/STATUS"
for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do
  test -s "$atom_restart_source/$name"
done
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test "${source_commit:0:8}" = "${source_root##*-}"
[[ "$failed_atom_job_id" =~ ^[0-9]+$ ]]
[[ "$qstar_array_job_id" =~ ^[0-9]+$ ]]
[[ "$solid_sos_job_id" =~ ^[0-9]+$ ]]
test -s "$original_receipt"
grep -qx "atom_job $failed_atom_job_id" "$original_receipt"
grep -qx "qstar_array_job $qstar_array_job_id" "$original_receipt"
grep -qx "solid_sos_job $solid_sos_job_id" "$original_receipt"
failed_atom_state=$(sacct -X -j "$failed_atom_job_id" -n -P -o State | awk -F'|' 'NF {print $1; exit}')
test "$failed_atom_state" = FAILED
test ! -e "$atom_recovery_root" && test ! -e "$receipt"
mkdir "$lock"
cleanup_lock() {
  rmdir "$lock" 2>/dev/null || true
}
trap cleanup_lock EXIT

atom_script=$scripts/run_threshold_candidate_atom_sos_55d25e3c9_d4810f73.slurm
collector_script=$scripts/run_threshold_candidate_qstar_binding_collect.slurm
test -s "$atom_script" && test -s "$collector_script"
common=ALL,CANDIDATE_ROOT="$candidate_root",RUN_ROOT="$run_root",SIAB_SOURCE_ROOT="$source_root",SIAB_SOURCE_COMMIT="$source_commit"
atom_export=$common,ATOM_RESTART_SOURCE="$atom_restart_source",ATOM_SOS_ROOT="$atom_recovery_root"

sbatch --test-only --partition=48cp2,p1 --export="$atom_export" "$atom_script" >/dev/null
sbatch --test-only --partition=48cp2,p1 \
  --export="$common,ATOM_SOS_ROOT=$atom_recovery_root,ATOM_JOB_ID=1,QSTAR_ARRAY_JOB_ID=$qstar_array_job_id,SOLID_SOS_JOB_ID=$solid_sos_job_id" \
  "$collector_script" >/dev/null

atom_job=$(sbatch --parsable --partition=48cp2,p1 --export="$atom_export" "$atom_script")
binding_job=$(sbatch --parsable --partition=48cp2,p1 --dependency="afterok:$atom_job:$solid_sos_job_id" \
  --export="$common,ATOM_SOS_ROOT=$atom_recovery_root,ATOM_JOB_ID=$atom_job,QSTAR_ARRAY_JOB_ID=$qstar_array_job_id,SOLID_SOS_JOB_ID=$solid_sos_job_id" \
  "$collector_script")

printf 'status submitted\nsource_commit %s\nfailed_atom_job %s\natom_restart_source %s\natom_recovery_root %s\natom_recovery_job %s\nqstar_array_job %s\nsolid_sos_job %s\nbinding_recovery_job %s\n' \
  "$source_commit" "$failed_atom_job_id" "$atom_restart_source" "$atom_recovery_root" "$atom_job" \
  "$qstar_array_job_id" "$solid_sos_job_id" "$binding_job" > "$receipt"
rmdir "$lock"
trap - EXIT
echo "C_THRESHOLD_ATOM_BINDING_RECOVERY_SUBMITTED receipt=$receipt"
