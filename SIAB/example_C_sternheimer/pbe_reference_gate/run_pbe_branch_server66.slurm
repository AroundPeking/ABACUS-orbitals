#!/usr/bin/env bash
#SBATCH --partition=640
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180000M
#SBATCH --time=24:00:00
#SBATCH --no-requeue

export C_PBE_GATE_PROFILE=server66
: "${C_PBE_GATE_ENTRYPOINT:?C_PBE_GATE_ENTRYPOINT is required}"
: "${C_PBE_GATE_COMMON_RUNNER:?C_PBE_GATE_COMMON_RUNNER is required}"
source "$C_PBE_GATE_COMMON_RUNNER"
