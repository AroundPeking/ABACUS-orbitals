#!/bin/bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
runner=$root/galerkin_binding_workflow/run_c_solid_q123_reduced_df.slurm

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
