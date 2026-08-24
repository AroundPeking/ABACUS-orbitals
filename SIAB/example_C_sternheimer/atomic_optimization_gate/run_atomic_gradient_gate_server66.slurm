#!/bin/bash
#SBATCH --job-name=c_siab_grad
#SBATCH --partition=640
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=110G
#SBATCH --time=UNLIMITED
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${TARGET:?TARGET is required}"
: "${PYTHON_EXE:?PYTHON_EXE is required}"

GATE_DIR="$REPO_ROOT/SIAB/example_C_sternheimer/atomic_optimization_gate"
OPTIMIZER="$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi/main.py"
COEFFICIENTS="$REPO_ROOT/SG15_v1.0/Orbitals_v2.0/C_TZDP/info/10/ORBITAL_RESULTS.txt"
ORBITAL="$REPO_ROOT/SG15_v1.0/Orbitals_v2.0/C_TZDP/C_gga_10au_100Ry_3s3p2d.orb"
OUTPUT="$CAMPAIGN_ROOT/runs/atomic_gradient_gate"

for path in "$GATE_DIR/atomic_optimization_gate.py" "$OPTIMIZER" \
  "$COEFFICIENTS" "$ORBITAL" "$TARGET" "$PYTHON_EXE"; do
  test -f "$path" && test ! -L "$path" && test -s "$path"
done
test ! -e "$OUTPUT/ATOMIC_GRADIENT_GATE_RESULT.json"
if test -d "$OUTPUT" && test -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)"; then
  echo "refusing nonempty output directory: $OUTPUT" >&2
  exit 2
fi

source /etc/profile.d/modules.sh
module purge
module load gcc10.2
module load intel20u4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-40}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_MAX_ACTIVE_LEVELS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export PYTHONPATH="$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$OUTPUT/.mplconfig"

"$PYTHON_EXE" "$GATE_DIR/atomic_optimization_gate.py" prepare \
  --target "$TARGET" \
  --coefficients "$COEFFICIENTS" \
  --orbital "$ORBITAL" \
  --output "$OUTPUT"

cd "$OUTPUT"
/usr/bin/time -v "$PYTHON_EXE" "$OPTIMIZER" > run.log 2>&1
"$PYTHON_EXE" "$GATE_DIR/atomic_optimization_gate.py" audit --output "$OUTPUT"
