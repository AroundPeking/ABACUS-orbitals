#!/bin/bash
set -euo pipefail

root=${SIAB_REFERENCE_ROOT:?}
source_dir=${SIAB_REFERENCE_SOURCE:?}
source_commit=${SIAB_REFERENCE_SOURCE_COMMIT:?}
test -s "$root/PREPARATION_MANIFEST.json"
test -s "$source_dir/run_siab_abfs_diag_server66.slurm"
test -s "$source_dir/run_siab_reference_server66.slurm"
test ! -e "$root/DIAG_JOB_ID.txt"
test ! -e "$root/SIAB_REFERENCE_JOB_ID.txt"
test ! -e "$root/.submission-claim"
grep -Eq '^sternheimer_mpi_layout[[:space:]]+global_equation$' "$root/INPUT"

campaign_id=$(printf '%s' "$root" | sha256sum | cut -c1-10)
diag_name=csd_${campaign_id}
reference_name=csr_${campaign_id}
mkdir -p "$root/logs"

queue_state=$(squeue -h -o '%i|%j|%T' 2>&1) || {
  echo "squeue failed; refusing an unobservable submission" >&2
  exit 2
}
history_state=$(sacct -n -X -S 2026-08-01 -o JobIDRaw,JobName,State 2>&1) || {
  echo "sacct failed; refusing an unobservable submission" >&2
  exit 2
}
if printf '%s\n%s\n' "$queue_state" "$history_state" | \
    grep -Eq "(^|[|[:space:]])(${diag_name}|${reference_name})([|[:space:]]|$)"; then
  echo "matching C SIAB reference job already exists" >&2
  exit 3
fi

mkdir "$root/.submission-claim"
trap 'rm -rf "$root/.submission-claim"' EXIT
printf '%s\n' "$queue_state" > "$root/.submission-claim/SQUEUE_BEFORE.txt"
printf '%s\n' "$history_state" > "$root/.submission-claim/SACCT_BEFORE.txt"

common_export=ALL,SIAB_REFERENCE_ROOT="$root",SIAB_REFERENCE_SOURCE_COMMIT="$source_commit",ABACUS_EXE="${ABACUS_EXE:?}",ABACUS_SHA256="${ABACUS_SHA256:?}",ABACUS_ENV_SCRIPT="${ABACUS_ENV_SCRIPT:?}"
sbatch --test-only \
  --job-name="$diag_name" \
  --export="$common_export",ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1 \
  "$source_dir/run_siab_abfs_diag_server66.slurm"
sbatch --test-only \
  --job-name="$reference_name" \
  --export="$common_export" \
  "$source_dir/run_siab_reference_server66.slurm"

diag_receipt=$(sbatch --parsable \
  --job-name="$diag_name" \
  --output="$root/logs/${diag_name}-%j.out" \
  --error="$root/logs/${diag_name}-%j.err" \
  --export="$common_export",ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY=1 \
  "$source_dir/run_siab_abfs_diag_server66.slurm")
diag_job=${diag_receipt%%;*}
test -n "$diag_job"

reference_receipt=$(sbatch --parsable \
  --dependency=afterok:"$diag_job" \
  --job-name="$reference_name" \
  --output="$root/logs/${reference_name}-%j.out" \
  --error="$root/logs/${reference_name}-%j.err" \
  --export="$common_export" \
  "$source_dir/run_siab_reference_server66.slurm")
reference_job=${reference_receipt%%;*}
test -n "$reference_job"

printf '%s\n' "$diag_job" > "$root/.submission-claim/DIAG_JOB_ID.txt"
printf '%s\n' "$reference_job" > "$root/.submission-claim/SIAB_REFERENCE_JOB_ID.txt"
mv "$root/.submission-claim/DIAG_JOB_ID.txt" "$root/DIAG_JOB_ID.txt"
mv "$root/.submission-claim/SIAB_REFERENCE_JOB_ID.txt" "$root/SIAB_REFERENCE_JOB_ID.txt"
trap - EXIT
echo "diag_job_id=$diag_job"
echo "siab_reference_job_id=$reference_job"
