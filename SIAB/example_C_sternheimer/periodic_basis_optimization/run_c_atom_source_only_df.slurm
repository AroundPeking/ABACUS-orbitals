#!/bin/bash
#SBATCH --partition=p1
#SBATCH --job-name=c_atom_siab_source
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000M
#SBATCH --time=UNLIMITED
#SBATCH --exclusive

set -euo pipefail

: "${REPO_ROOT:?}"
: "${SOURCE_ROOT:?}"
: "${RUN_ROOT:?}"
: "${ABACUS_EXE:?}"
: "${ABACUS_SHA256:?}"
: "${EXPECTED_SOURCE_COMMIT:?}"
: "${EXPECTED_TARGET_SHA256:?}"
: "${PYTHON_EXE:?}"

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NNODES:?}" -eq 4
test "${SLURM_NTASKS:?}" -eq 4
test "${SLURM_NTASKS_PER_NODE:?}" -eq 1
test "${SLURM_CPUS_PER_TASK:?}" -eq 40
test -x "$ABACUS_EXE"
test -x "$PYTHON_EXE"
test "$(sha256sum "$ABACUS_EXE" | awk '{print $1}')" = "$ABACUS_SHA256"
test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test ! -e "$RUN_ROOT"

target="$SOURCE_ROOT/OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat"
test -s "$target"
test "$(sha256sum "$target" | awk '{print $1}')" = "$EXPECTED_TARGET_SHA256"
for name in INPUT STRU KPT fixed_frequency_grid_nfreq16.dat SOURCE_EIG_OCC.txt \
  C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_3s3p2d.orb \
  C_10au_3s3p2d1f1g_1e-4.abfs; do
  test -s "$SOURCE_ROOT/$name"
done
for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do
  test -s "$SOURCE_ROOT/OUT.C_SIAB_REFERENCE/$name"
done

mkdir -p "$RUN_ROOT/OUT.C_ATOM_SIAB_SOURCE"
cp "$SOURCE_ROOT"/{STRU,KPT,fixed_frequency_grid_nfreq16.dat,SOURCE_EIG_OCC.txt} "$RUN_ROOT/"
cp "$SOURCE_ROOT"/{C_ONCV_PBE-1.0.upf,C_gga_10au_100Ry_3s3p2d.orb,C_10au_3s3p2d1f1g_1e-4.abfs} "$RUN_ROOT/"
cp "$SOURCE_ROOT/OUT.C_SIAB_REFERENCE"/{wfs1_nao.txt,wfs2_nao.txt,chgs1.cube,chgs2.cube} \
  "$RUN_ROOT/OUT.C_ATOM_SIAB_SOURCE/"

awk '
  $1 == "suffix" { $2 = "C_ATOM_SIAB_SOURCE" }
  $1 == "out_sternheimer_siab" { next }
  $1 == "out_sternheimer_basis_opt" { next }
  $1 == "sternheimer_siab_source_only" { next }
  { print }
  END {
    print "out_sternheimer_basis_opt 1"
    print "sternheimer_siab_source_only 1"
  }
' "$SOURCE_ROOT/INPUT" > "$RUN_ROOT/INPUT"

grep -Eq '^out_sternheimer_librpa[[:space:]]+0$' "$RUN_ROOT/INPUT"
grep -Eq '^out_sternheimer_basis_opt[[:space:]]+1$' "$RUN_ROOT/INPUT"
grep -Eq '^sternheimer_siab_source_only[[:space:]]+1$' "$RUN_ROOT/INPUT"
grep -Eq '^sternheimer_nfreq[[:space:]]+16$' "$RUN_ROOT/INPUT"
grep -Eq '^sternheimer_mpi_layout[[:space:]]+global_equation$' "$RUN_ROOT/INPUT"

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

cd "$RUN_ROOT"
{
  echo "status=running"
  echo "job_id=${SLURM_JOB_ID:?}"
  echo "source_commit=$EXPECTED_SOURCE_COMMIT"
  echo "abacus_sha256=$ABACUS_SHA256"
  echo "target_sha256=$EXPECTED_TARGET_SHA256"
  echo "nodes=$SLURM_NNODES"
  echo "mpi_ranks=$SLURM_NTASKS"
  echo "omp_threads=$OMP_NUM_THREADS"
  sha256sum INPUT STRU KPT fixed_frequency_grid_nfreq16.dat ./*.upf ./*.orb ./*.abfs
} > SOURCE_ONLY_PROVENANCE.txt

ulimit -s unlimited
/usr/bin/time -v -o abacus.time \
  mpirun -ppn 1 -np "$SLURM_NTASKS" "$ABACUS_EXE" > abacus.out 2>&1

status=OUT.C_ATOM_SIAB_SOURCE/STERNHEIMER_SIAB_STATUS.dat
source_file=OUT.C_ATOM_SIAB_SOURCE/STERNHEIMER_SIAB_SOURCE_V1.dat
test -s "$status"
test -s "$source_file"
grep -Eq '^status[[:space:]]+success$' "$status"
grep -Eq '^format[[:space:]]+siab_source_v1$' "$status"
grep -Eq '^sternheimer_siab_source_only[[:space:]]+yes$' "$status"

export PYTHONPATH="$REPO_ROOT/SIAB/opt_orb_pytorch_dpsi${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_EXE" - "$target" "$RUN_ROOT/$source_file" "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from IO.read_sternheimer import read_sternheimer
from IO.read_sternheimer_source import read_sternheimer_source
from sternheimer_source_pair import pair_response_and_source


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


target = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
root = Path(sys.argv[3]).resolve()
pair = pair_response_and_source(
    read_sternheimer(target),
    read_sternheimer_source(source),
)
payload = {
    "status": "success",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "target": {"path": str(target), "sha256": sha256(target), "size": target.stat().st_size},
    "source": {"path": str(source), "sha256": sha256(source), "size": source.stat().st_size},
    "provenance_warnings": list(pair.provenance_warnings),
    "physical_provenance": pair.response.provenance,
}
(root / "SOURCE_ONLY_COMPLETE.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="ascii",
)
PY

test -s SOURCE_ONLY_COMPLETE.json

