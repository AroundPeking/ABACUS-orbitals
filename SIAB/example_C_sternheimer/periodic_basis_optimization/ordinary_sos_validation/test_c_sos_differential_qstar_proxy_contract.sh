#!/bin/bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner=$root/run_c_sos_differential_qstar_proxy_df.slurm

test -s "$runner"
grep -q '^#SBATCH --partition=p1$' "$runner"
grep -q '^#SBATCH --nodes=1$' "$runner"
grep -q '^#SBATCH --ntasks=1$' "$runner"
grep -q '^#SBATCH --cpus-per-task=1$' "$runner"
grep -q '^#SBATCH --mem=4000M$' "$runner"
grep -q '^#SBATCH --time=00:10:00$' "$runner"
grep -q 'module load python/3.9.22' "$runner"
grep -Fq 'test "$(cat "$REPO_ROOT/SOURCE_COMMIT")" = "$SIAB_COMMIT"' "$runner"
grep -Fq 'test ! -e "$RUN_ROOT"' "$runner"
grep -q 'c_sos_differential_qstar_proxy.py' "$runner"
grep -q 'c_sos_differential_qstar_proxy_current.json' "$runner"

echo C_SOS_DIFFERENTIAL_QSTAR_PROXY_CONTRACT_OK
