#!/bin/bash
set -euo pipefail

root=${SIAB_REFERENCE_ROOT:?}
source_dir=${SIAB_REFERENCE_SOURCE:?}
source_commit=${SIAB_REFERENCE_SOURCE_COMMIT:?}
pilot_job_file=$root/DF_PILOT_JOB_ID.txt
reference_job_file=$root/DF_REFERENCE_JOB_ID.txt
recovery_job_file=$root/DF_REFERENCE_RECOVERY_JOB_ID.txt
claim=$root/.submission-claim-df-recovery

test -s "$root/PREPARATION_MANIFEST.json"
test -s "$source_dir/run_siab_reference_df.slurm"
test -s "$pilot_job_file"
test -s "$reference_job_file"
test ! -e "$recovery_job_file"
test ! -e "$root/SIAB_REFERENCE_COMPLETE.json"
test ! -e "$claim"

python3 - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
pilot = json.loads((root / "pilot_df/SIAB_REFERENCE_PILOT.json").read_text(encoding="ascii"))
assert pilot["status"] == "runtime_gate_failed"
assert pilot["abacus_exit_code"] == 124
assert pilot["completed_equations"] == 0
assert pilot["mpi_ranks"] == 10
assert pilot["omp_threads_per_rank"] == 40

progress = sorted((root / "pilot_df").glob("STERNHEIMER_SIAB_PROGRESS_rank*.dat"))
assert len(progress) == 10
for path in progress:
    text = path.read_text(encoding="ascii", errors="replace")
    assert "channel_workers_ready" in text
assert not list((root / "pilot_df").glob("STERNHEIMER_CHI0_FAILURE_rank*.dat"))
PY

original_reference_job=$(cat "$reference_job_file")
original_state=$(sacct -j "$original_reference_job" -X -n -o State | awk 'NF {print $1; exit}')
original_elapsed=$(sacct -j "$original_reference_job" -X -n -o Elapsed | awk 'NF {print $1; exit}')
case "$original_state" in
  CANCELLED*) ;;
  *) echo "original formal job is not cancelled: $original_state" >&2; exit 2 ;;
esac
test "$original_elapsed" = 00:00:00

campaign_id=$(printf '%s' "$root" | sha256sum | cut -c1-10)
recovery_name=csrr_${campaign_id}
queue_state=$(squeue -h -o '%i|%j|%T')
history_state=$(sacct -n -X -S 2026-08-01 -o JobIDRaw,JobName,State)
if printf '%s\n%s\n' "$queue_state" "$history_state" | \
    grep -Eq "(^|[|[:space:]])${recovery_name}([|[:space:]]|$)"; then
  echo "matching C SIAB recovery job already exists on df" >&2
  exit 3
fi

mkdir "$claim"
trap 'rm -rf "$claim"' EXIT
printf '%s\n' "$queue_state" > "$claim/SQUEUE_BEFORE.txt"
printf '%s\n' "$history_state" > "$claim/SACCT_BEFORE.txt"

common_export=ALL,SIAB_REFERENCE_ROOT="$root",SIAB_REFERENCE_SOURCE_COMMIT="$source_commit",SIAB_REFERENCE_TOTAL_EQUATIONS=15040,ABACUS_EXE="${ABACUS_EXE:?}",ABACUS_SHA256="${ABACUS_SHA256:?}",ABACUS_SOURCE_COMMIT="${ABACUS_SOURCE_COMMIT:?}"
sbatch --test-only --job-name="$recovery_name" --export="$common_export" \
  "$source_dir/run_siab_reference_df.slurm"

receipt=$(sbatch --parsable \
  --job-name="$recovery_name" \
  --output="$root/logs/${recovery_name}-%j.out" \
  --error="$root/logs/${recovery_name}-%j.err" \
  --export="$common_export" \
  "$source_dir/run_siab_reference_df.slurm")
job=${receipt%%;*}
test -n "$job"
printf '%s\n' "$job" > "$claim/DF_REFERENCE_RECOVERY_JOB_ID.txt"
mv "$claim/DF_REFERENCE_RECOVERY_JOB_ID.txt" "$recovery_job_file"
trap - EXIT
echo "df_reference_recovery_job_id=$job"
