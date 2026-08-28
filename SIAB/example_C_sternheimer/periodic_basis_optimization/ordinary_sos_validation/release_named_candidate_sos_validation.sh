#!/bin/bash

set -euo pipefail
trap 'status=$?; echo "C_NAMED_SOS_RELEASE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
validation=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
manifest=$candidate_root/CANDIDATE.json
atom_producer=$validation/run_named_candidate_atom_producer_55d25e3c9.slurm
atom_sos=$validation/run_named_candidate_atom_sos_d4810f73.slurm
solid_grid=$validation/run_named_candidate_solid_full_bz_55d25e3c9.slurm
solid_sos=$validation/run_named_candidate_solid_sos_d4810f73.slurm
binding=$validation/run_named_candidate_binding_collect.slurm

for file in "$manifest" "$atom_producer" "$atom_sos" "$solid_grid" "$solid_sos" "$binding"; do test -s "$file"; done
grep -Eq '^[[:space:]]*"pre_sos_gate": "pass",$' "$manifest"
test ! -e "$run_root"
mkdir -p "$run_root"
exec 9>"$run_root/.release.lock"
if ! flock -n 9; then
  echo "refusing duplicate: release lock is held" >&2
  exit 2
fi
for job_name in c2g_atom_prod c2g_atom_sos c2g_solid_q c2g_solid_sos c2g_bind; do
  if squeue -h -u "$USER" -o '%j' | grep -qx "$job_name"; then
    echo "refusing duplicate: active scheduler job $job_name already exists" >&2
    exit 2
  fi
done

exports=ALL,SIAB_SOURCE_ROOT=$code,CANDIDATE_ROOT=$candidate_root,RUN_ROOT=$run_root
for script in "$atom_producer" "$atom_sos" "$solid_grid" "$solid_sos" "$binding"; do
  sbatch --test-only --export="$exports" "$script" >> "$run_root/SBATCH_TEST_ONLY.txt" 2>&1
done

submit() {
  local output job_id
  output=$(sbatch --parsable "$@")
  job_id=${output%%;*}
  test "$job_id" -gt 0
  printf '%s\n' "$job_id"
}
atom_producer_job=$(submit --export="$exports" "$atom_producer")
atom_sos_job=$(submit --dependency=afterok:$atom_producer_job --export="$exports" "$atom_sos")
solid_grid_job=$(submit --export="$exports" "$solid_grid")
solid_sos_job=$(submit --dependency=afterok:$solid_grid_job --export="$exports,GRID_COULOMB_ARRAY_JOB_ID=$solid_grid_job" "$solid_sos")
binding_job=$(submit --dependency=afterok:$atom_sos_job:$solid_sos_job --export="$exports" "$binding")

cat > "$run_root/SUBMISSION.txt" <<EOF
status submitted
scope ordinary_all_band_sos_only
candidate_root $candidate_root
candidate_manifest_sha256 $(sha256sum "$manifest" | awk '{print $1}')
source_root $code
atom_producer_job $atom_producer_job
atom_sos_job $atom_sos_job
solid_grid_array_job $solid_grid_job
solid_sos_job $solid_sos_job
binding_job $binding_job
EOF
sha256sum "$0" "$manifest" "$atom_producer" "$atom_sos" "$solid_grid" "$solid_sos" "$binding" > "$run_root/SUBMISSION_SHA256.txt"
echo "C_NAMED_SOS_RELEASE_OK run_root=$run_root atom=$atom_producer_job atom_sos=$atom_sos_job solid_grid=$solid_grid_job solid_sos=$solid_sos_job binding=$binding_job"
