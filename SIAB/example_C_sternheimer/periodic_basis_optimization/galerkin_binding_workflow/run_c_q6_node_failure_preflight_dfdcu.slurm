#!/bin/bash
#SBATCH --job-name=c_q6_node_recovery
#SBATCH --partition=long,normal
#SBATCH --nodes=48
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110000M
#SBATCH --time=12:00:00
#SBATCH --array=2%1
#SBATCH --no-requeue

set -euo pipefail
: "${C_SOLID_DFDCU_STAGE_ROOT:?missing immutable stage}"
: "${OUTPUT_ROOT:?missing new output root}"
: "${PREFLIGHT_WRAPPER_SHA256:?missing wrapper hash}"
: "${MPI_PREFLIGHT_SOURCE_SHA256:?missing probe hash}"
: "${RECOVERY_PARENT_JOB_ID:?missing terminal parent job}"
test "$RECOVERY_PARENT_JOB_ID" = 21870198_2
test "$SLURM_ARRAY_TASK_ID" = 2
test "$SLURM_NTASKS" = 48
test "$SLURM_NTASKS_PER_NODE" = 1
test "$SLURM_CPUS_PER_TASK" = 30
test -z "${C_SOLID_CHARGE_RESTART_CONTRACT:-}"
test "$(sacct -X -j "$RECOVERY_PARENT_JOB_ID" --noheader --format=State -P | tr -d ' |\n')" = NODE_FAIL
test "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" = "$PREFLIGHT_WRAPPER_SHA256"
probe=$C_SOLID_DFDCU_STAGE_ROOT/source/c_mpi_communication_preflight.c
test "$(sha256sum "$probe" | awk '{print $1}')" = "$MPI_PREFLIGHT_SOURCE_SHA256"
test ! -e "$OUTPUT_ROOT/q6"
mkdir "$C_SOLID_DFDCU_STAGE_ROOT/MPI_PREFLIGHT_EXECUTION"
cd "$C_SOLID_DFDCU_STAGE_ROOT/MPI_PREFLIGHT_EXECUTION"
export LANG=C LC_ALL=C
source "${DFDCU_ENV:-/public/home/ghj/app/src/env_60_245_intel2021.sh}"
export OMP_NUM_THREADS=30 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=omp I_MPI_FABRICS=shm:ofi
mpiicc -O2 -std=c99 "$probe" -o mpi_communication_preflight
timeout 180 mpirun -np 48 -ppn 1 ./mpi_communication_preflight > MPI_PREFLIGHT.log 2>&1
grep -q '^MPI_COMM_PREFLIGHT_OK ranks=48 ' MPI_PREFLIGHT.log
{
  echo status=success
  echo scope=communication_preflight_not_physical_acceptance
  echo parent_job=$RECOVERY_PARENT_JOB_ID
  echo parent_state=NODE_FAIL
  sha256sum "$probe" MPI_PREFLIGHT.log
} > PROVENANCE.txt

# The frozen standard runner owns all physics and final validation unchanged.
exec bash "$C_SOLID_DFDCU_STAGE_ROOT/source/run_c_solid_fd8_q13_standard_dfdcu.slurm"
