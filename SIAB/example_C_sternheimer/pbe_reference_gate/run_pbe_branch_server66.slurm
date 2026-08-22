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
export C_PBE_GATE_ENTRYPOINT="${BASH_SOURCE[0]}"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/run_pbe_branch_common.sh"
