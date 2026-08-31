#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
script=$root/run_sparse_qstar_librpa_replay_df.slurm

test -s "$script"
grep -qx '#SBATCH --partition=p1' "$script"
grep -qx '#SBATCH --nodes=1' "$script"
grep -qx '#SBATCH --ntasks-per-node=1' "$script"
grep -qx '#SBATCH --cpus-per-task=40' "$script"
grep -qx '#SBATCH --mem=190000' "$script"
grep -qx '#SBATCH --time=1-00:00:00' "$script"
grep -q 'test ! -e "$run_root"' "$script"
grep -q 'test -s "$source_root/librpa.in"' "$script"
grep -q 'test -d "$source_root/reader_v1"' "$script"
grep -q 'n_bands_chi0[[:space:]]*=[[:space:]]*-1' "$script"
grep -q 'librpa_debug=${LIBRPA_DEBUG:-f}' "$script"
grep -q 'source_commit=${SIAB_SOURCE_COMMIT:?' "$script"
grep -q 'test "$librpa_debug" = f || test "$librpa_debug" = t' "$script"
grep -q 'debug[[:space:]]*=[[:space:]]*$librpa_debug' "$script"
grep -q 'echo librpa_debug=$librpa_debug' "$script"
grep -q 'echo source_commit=$source_commit' "$script"
grep -q 'mpirun -np 1 "$librpa"' "$script"
grep -q 'libRPA finished successfully' "$script"
grep -q 'status=success' "$script"

echo SPARSE_QSTAR_LIBRPA_REPLAY_CONTRACT_OK
