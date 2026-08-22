#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc10.2
module load intel20u4
export LD_LIBRARY_PATH="/home/apps/gcc10.2/lib64:/home/apps/gcc10.2/lib:/home/apps/intel20u4/lib/intel64_lin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
