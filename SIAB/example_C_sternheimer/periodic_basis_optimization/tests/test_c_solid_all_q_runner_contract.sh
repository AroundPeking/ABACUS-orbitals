#!/bin/bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
runner=$root/galerkin_binding_workflow/run_c_solid_q123_reduced_df.slurm
standard_q1_runner=$root/galerkin_binding_workflow/run_c_solid_fd8_q13_standard_q1_df.slurm
standard_remaining_runner=$root/galerkin_binding_workflow/run_c_solid_fd8_q13_standard_remaining_df.slurm
standard_q_validator=$root/galerkin_binding_workflow/validate_c_solid_fd8_q_dataset.py

test -s "$runner"
grep -q '^#SBATCH --ntasks=1$' "$runner"
grep -q '^#SBATCH --cpus-per-task=40$' "$runner"
grep -q 'SIAB_SOURCE_ROOT' "$runner"
grep -q 'SIAB_SOURCE_COMMIT' "$runner"
grep -q 'SIAB_SCRIPT_SHA256' "$runner"
grep -q 'SIAB_CONFIG_SHA256' "$runner"
grep -q 'gitdir:' "$runner"
grep -q 'INITIAL_COEFFICIENTS' "$runner"
grep -q 'INITIAL_SHA256' "$runner"
grep -q 'Q1_PHYSICS_HASH' "$runner"
grep -q 'Q2_PHYSICS_HASH' "$runner"
grep -q 'Q3_PHYSICS_HASH' "$runner"
grep -q 'test ! -e "$output"' "$runner"
grep -q -- '--qstar "1=$q1"' "$runner"
grep -q -- '--qstar "2=$q2"' "$runner"
grep -q -- '--qstar "3=$q3"' "$runner"
grep -q 'c_diamond_solid_q123_reduced.json' "$runner"
grep -q 'physical_release_gate.*hold' "$runner"
if grep -Eqi 'atomic.response|atomic.source|c.atom' "$runner"; then
  echo "solid-only runner contains an atomic input" >&2
  exit 1
fi
if grep -Eq '(^|[ ;])git[[:space:]]' "$runner"; then
  echo "compute-node runner depends on an unavailable Git executable" >&2
  exit 1
fi

test -s "$standard_q1_runner"
grep -Fq 'source_head=$(cat "$SIAB_SOURCE_ROOT/.git/HEAD")' "$standard_q1_runner"
grep -Fq 'source_ref=${source_head#ref: }' "$standard_q1_runner"
if grep -Eq '(^|[ ;])git[[:space:]]' "$standard_q1_runner"; then
  echo "standard q1 compute-node runner depends on an unavailable Git executable" >&2
  exit 1
fi

test -s "$standard_remaining_runner"
grep -q '^#SBATCH --array=0-11%1$' "$standard_remaining_runner"
grep -Fq 'q_labels=(2 3 6 7 8 11 22 23 24 27 28 43)' "$standard_remaining_runner"
grep -Fq 'q_multiplicities=(6 3 6 12 6 3 2 6 6 6 6 1)' "$standard_remaining_runner"
grep -Fq 'set_input_key sternheimer_frequency_grid_file "$frequency_name"' "$standard_remaining_runner"
grep -Fq -- '--expected-frequency-grid "$frequency_name"' "$standard_remaining_runner"
if grep -Eq '(^|[ ;])git[[:space:]]' "$standard_remaining_runner"; then
  echo "standard remaining-q compute-node runner depends on an unavailable Git executable" >&2
  exit 1
fi

test -s "$standard_q_validator"
if grep -q '^from __future__ import annotations$' "$standard_q_validator"; then
  echo "standard q validator is incompatible with df Python 3.6" >&2
  exit 1
fi
