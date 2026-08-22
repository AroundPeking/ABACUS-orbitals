#!/bin/bash
#SBATCH --partition=640
#SBATCH --job-name=c_delta_librpa
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180000M
#SBATCH --time=1-00:00:00
#SBATCH --exclusive

set -euo pipefail

root=${DELTA_GATE_ROOT:?}
librpa=${LIBRPA_EXE:?}
expected_librpa_sha=${LIBRPA_SHA256:?}
runtime_env=${ABACUS_ENV_SCRIPT:?}
python=${PYTHON_EXE:-python3}
branches=(fixed free)
branch=${branches[${SLURM_ARRAY_TASK_ID:?}]}
source_dir=$root/branches/$branch
work=$source_dir/librpa

test "${SLURM_JOB_PARTITION:?}" = 640
test "${SLURM_NNODES:?}" -eq 1
test "${SLURM_NTASKS:?}" -eq 1
test "${SLURM_CPUS_PER_TASK:?}" -eq 48
test -x "$librpa"
test -s "$runtime_env"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -s "$source_dir/RESPONSE_COMPLETE.json"
test ! -e "$work"

source "$runtime_env"
mkdir -p "$work/input"
for name in stru_out bz_sampling_out basis_out basis_wfc_out basis_aux_out \
  v1_coulomb_full_iq_1_rank0.dat; do
  test -s "$source_dir/$name"
  ln -s "$source_dir/$name" "$work/input/$name"
done
mapfile -t responses < <(find "$source_dir" -maxdepth 1 -type f \
  -name 'v1_sternheimer_chi0_iq_1_ifreq_*.dat' | sort)
test "${#responses[@]}" -eq 6
for source in "${responses[@]}"; do
  ln -s "$source" "$work/input/${source##*/}"
done
printf '1 0.0 0.0 0.0 1.0\n' > "$work/input/qpoints.dat"

cat > "$work/librpa.in" <<EOF
task = sternheimer_rpa
input_dir = $work/input/
version_coul_reader = 1
nfreq = 6
sqrt_coulomb_threshold = 1e-5
prefix_coul_full = v1_coulomb_full_iq_
prefix_sternheimer_chi0 = v1_sternheimer_chi0_iq_
fn_sternheimer_qpoints = qpoints.dat
use_rpa_gamma = true
replace_w_head = false
EOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE

cd "$work"
/usr/bin/time -v -o librpa.time \
  mpirun -np "$SLURM_NTASKS" -ppn 1 "$librpa" > librpa.out 2>&1
grep -q 'libRPA finished successfully' librpa.out
test "$(grep -c 'Reading head/wing input from' librpa.out || true)" -eq 0
test "$(grep -Ec '^\|[[:space:]]+[0-9]+[[:space:]]+' librpa.out)" -eq 6
grep -q 'Total Sternheimer EcRPA' librpa.out

"$python" - "$branch" "$expected_librpa_sha" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

branch, exe_sha = sys.argv[1:]
text = Path("librpa.out").read_text(encoding="ascii", errors="replace")
row = re.compile(
    r"^\|\s+(\d+)\s+([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+"
    r"([+\-0-9.eE]+)(?:\s+([+\-0-9.eE]+))?",
    re.MULTILINE,
)
rows = []
for match in row.finditer(text):
    ifreq = int(match.group(1))
    if 1 <= ifreq <= 6:
        values = [float(value) for value in match.groups()[1:] if value is not None]
        assert all(math.isfinite(value) for value in values)
        rows.append({"ifreq": ifreq, "values": values})
assert sorted(item["ifreq"] for item in rows) == list(range(1, 7))

matches = re.findall(r"Total Sternheimer EcRPA:\s*([+\-0-9.eE]+)", text)
assert matches
energy = float(matches[-1])
assert math.isfinite(energy)
payload = {
    "status": "LIBRPA_COMPLETE",
    "branch": branch,
    "librpa_sha256": exe_sha,
    "task": "sternheimer_rpa",
    "nfreq": 6,
    "coulomb_kernel": "full",
    "sqrt_coulomb_threshold": 1.0e-5,
    "trace_log_rows": rows,
    "ecrpa_ha": energy,
    "librpa_out_sha256": hashlib.sha256(Path("librpa.out").read_bytes()).hexdigest(),
}
Path("LIBRPA_COMPLETE.json.tmp").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
)
Path("LIBRPA_COMPLETE.json.tmp").replace("LIBRPA_COMPLETE.json")
PY

echo "LIBRPA_OK branch=$branch"
