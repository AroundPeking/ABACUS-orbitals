#!/bin/bash
#SBATCH --partition=p1
#SBATCH --job-name=c_siab_pilot_df
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000M
#SBATCH --time=00:45:00
#SBATCH --exclusive

set -euo pipefail

# Match the 10-rank production shape so the runtime gate measures that layout.

root=${SIAB_REFERENCE_ROOT:?}
abacus=${ABACUS_EXE:?}
expected_abacus_sha=${ABACUS_SHA256:?}
expected_total_equations=${SIAB_REFERENCE_TOTAL_EQUATIONS:-15040}
pilot=$root/pilot_df

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NNODES:?}" -eq 10
test "${SLURM_NTASKS:?}" -eq 10
test "${SLURM_CPUS_PER_TASK:?}" -eq 40
test ! -e "$pilot"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
grep -Eq '^sternheimer_nfreq[[:space:]]+16$' "$root/INPUT"
grep -Eq '^sternheimer_mpi_layout[[:space:]]+global_equation$' "$root/INPUT"

mkdir "$pilot"
cp "$root"/INPUT "$root"/STRU "$root"/KPT "$root"/*.upf "$root"/*.orb \
  "$root"/*.abfs "$root"/fixed_frequency_grid_nfreq16.dat "$pilot"/
cp -R "$root/OUT.C_SIAB_REFERENCE" "$pilot/OUT.C_SIAB_REFERENCE"

module load oneapi/2024.2
export LD_LIBRARY_PATH=/data/home/df_iopcas_ghj/app/elpa-build/lib:/data/app/libxc/6.2.2-intel/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export OMP_STACKSIZE=2G
export MKL_DYNAMIC=FALSE
export I_MPI_FABRICS=shm:ofi
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export ABACUS_STERNHEIMER_CHANNEL_THREADS=40
export ABACUS_STERNHEIMER_FD_ST_FREQ_RANK_SHIFT=1
export ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL=1e-6
export ABACUS_STERNHEIMER_FD_ST_MAX_ITER=300
export ABACUS_STERNHEIMER_FD_ST_SOLVER_MAX_ITER=300
export ABACUS_STERNHEIMER_DELTA_A_BLOCK=reference_value_gradient

cd "$pilot"
ulimit -s unlimited
set +e
/usr/bin/time -v -o abacus.time \
  timeout --signal=TERM --kill-after=120s 30m \
  mpirun -ppn 1 -np "$SLURM_NTASKS" "$abacus" > abacus.out 2>&1
abacus_exit=$?
set -e

python3 - "$pilot" "$abacus_exit" "$expected_total_equations" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
abacus_exit = int(sys.argv[2])
total_equations = int(sys.argv[3])
equations = []
max_elapsed = 0.0
for path in sorted(root.glob("STERNHEIMER_SIAB_PROGRESS_rank*.dat")):
    for line in path.read_text(encoding="ascii", errors="replace").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 12:
            continue
        try:
            max_elapsed = max(max_elapsed, float(fields[11]))
        except ValueError:
            pass
        if fields[0] == "equation":
            equations.append(fields)

throughput = len(equations) / max_elapsed if max_elapsed > 0.0 else 0.0
projected_hours = (
    1.25 * total_equations / throughput / 3600.0 if throughput > 0.0 else None
)
passed = bool(equations) and projected_hours is not None and projected_hours <= 18.0
payload = {
    "status": "runtime_gate_passed" if passed else "runtime_gate_failed",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "abacus_exit_code": abacus_exit,
    "mpi_ranks": 10,
    "omp_threads_per_rank": 40,
    "completed_equations": len(equations),
    "observed_elapsed_seconds": max_elapsed,
    "aggregate_equations_per_second": throughput,
    "expected_total_equations": total_equations,
    "safety_factor": 1.25,
    "projected_hours": projected_hours,
    "acceptance_limit_hours": 18.0,
}
(root / "SIAB_REFERENCE_PILOT.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
)
if not passed:
    raise SystemExit(5)
PY
