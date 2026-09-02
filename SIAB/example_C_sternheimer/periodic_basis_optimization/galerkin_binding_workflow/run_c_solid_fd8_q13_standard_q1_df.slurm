#!/bin/bash
#SBATCH --job-name=c_fd8_q1_nf12_p1e6
#SBATCH --partition=p1
#SBATCH --nodes=48
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=12:00:00

set -euo pipefail
trap 'status=$?; echo "C_SOLID_FD8_Q1_STANDARD_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${SIAB_SOURCE_ROOT:?missing immutable SIAB source deployment}"
: "${SIAB_SOURCE_COMMIT:?missing SIAB source commit}"
: "${SIAB_RUNNER_SHA256:?missing q1 runner SHA256}"
: "${SIAB_CONFIG_SHA256:?missing standard config SHA256}"
: "${OUTPUT_ROOT:?missing immutable output root}"

module load oneapi/2024.2

workflow=$SIAB_SOURCE_ROOT/SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow
runner=$workflow/run_c_solid_fd8_q13_standard_q1_df.slurm
config=$workflow/c_diamond_solid_fd8_q13_standard.json
reference_grid=$workflow/c_diamond_fd8_nfreq12_reference.tsv
abacus_root=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825
abacus=$abacus_root/artifacts/job3117942/abacus_3p
template=$abacus_root/gate-template-product-pca
abacus_commit=711af860c125b9757c344a1961b63524c550cfe4
expected_abacus_sha=5a7ab8f2fc14424c8b47c5a9cbe22d1784016140cc9fa302f7e032662eed58b2
orbital_sha256=7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d
pseudopotential_sha256=e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99
work=$OUTPUT_ROOT
suffix=C_DIAMOND_FD8_Q1_NFREQ12_PCA1E6_BASIS_OPT

resolve_source_head() {
  local source_head source_ref ref_path
  source_head=$(cat "$SIAB_SOURCE_ROOT/.git/HEAD")
  if [[ "$source_head" == "ref: "* ]]; then
    source_ref=${source_head#ref: }
    ref_path=$SIAB_SOURCE_ROOT/.git/$source_ref
    if [[ -s "$ref_path" ]]; then
      cat "$ref_path"
    else
      awk -v ref="$source_ref" '$2 == ref {print $1; found=1} END {exit !found}' \
        "$SIAB_SOURCE_ROOT/.git/packed-refs"
    fi
  else
    printf '%s\n' "$source_head"
  fi
}

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 48
test "$SLURM_NTASKS_PER_NODE" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test -d "$SIAB_SOURCE_ROOT/.git"
test "$(resolve_source_head)" = "$SIAB_SOURCE_COMMIT"
test "$(sha256sum "$runner" | awk '{print $1}')" = "$SIAB_RUNNER_SHA256"
test "$(sha256sum "$config" | awk '{print $1}')" = "$SIAB_CONFIG_SHA256"
test -s "$reference_grid"
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
grep -Fq "$abacus_commit" "$abacus_root/artifacts/job3117942/commit_metadata.txt"
test -d "$template"
test ! -e "$work"

mkdir -p "$work"
cp -a "$template/." "$work/"
cd "$work"

set_input_key() {
  local key=$1 value=$2
  if awk -v key="$key" '$1 == key {found=1} END {exit !found}' INPUT; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]+.*|${key} ${value}|" INPUT
  else
    printf '%-36s %s\n' "$key" "$value" >> INPUT
  fi
}

sed -i -E 's|^[[:space:]]*2[[:space:]]+2[[:space:]]+2[[:space:]]+0[[:space:]]+0[[:space:]]+0[[:space:]]*$|4 4 4 0 0 0|' KPT
set_input_key suffix "$suffix"
set_input_key ecutwfc 45
set_input_key nx 24
set_input_key ny 24
set_input_key nz 24
set_input_key nbands 44
set_input_key kpar 4
set_input_key rpa 1
set_input_key out_librpa_reader_version 1
set_input_key out_sternheimer_librpa 0
set_input_key out_sternheimer_basis_opt 1
set_input_key sternheimer_fd_order 8
set_input_key sternheimer_nfreq 12
set_input_key sternheimer_frequency_mpi 1
set_input_key sternheimer_channel_mpi 0
set_input_key sternheimer_mpi_layout frequency_grouped
set_input_key sternheimer_q_index 1
set_input_key exx_pca_threshold 1e-6
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 2
sed -i -E '/^[[:space:]]*sternheimer_frequency_grid_file[[:space:]]+/d' INPUT

! grep -q '^ABFS_ORBITAL' STRU
test -z "$(find . -maxdepth 1 -type f -name '*.abfs' -print -quit)"
test "$(sha256sum C_gga_10au_100Ry_3s3p2d.orb | awk '{print $1}')" = "$orbital_sha256"
test "$(sha256sum C_ONCV_PBE-1.0.upf | awk '{print $1}')" = "$pseudopotential_sha256"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export OMP_STACKSIZE=2G
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=omp
export I_MPI_FABRICS=shm:ofi
export ABACUS_STERNHEIMER_CHANNEL_THREADS=$OMP_NUM_THREADS
unset ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL

{
  echo status=running
  echo purpose=c_solid_fd8_q13_standard_q1_timing_and_size_gate
  echo siab_source_commit=$SIAB_SOURCE_COMMIT
  echo abacus_commit=$abacus_commit
  echo abacus_sha256=$expected_abacus_sha
  echo qstar_scheme=fd8_discrete_13
  echo q_index=1
  echo q_multiplicity=1
  echo q_count=64
  echo frequency_count=12
  echo frequency_grid=greenx_minimax
  echo auxiliary_basis_source=threshold_only_product_pca
  echo pca_threshold=1e-6
  echo coulomb=full_periodic
  echo grid=24,24,24
  echo kmesh=4,4,4
  echo fd_order=8
  echo mpi_layout=frequency_grouped
  echo response_kpoint_groups=4
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$runner" "$config" "$reference_grid" "$abacus" INPUT STRU KPT \
    C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_3s3p2d.orb
} > RUN_PROVENANCE.txt

/usr/bin/time -v -o abacus.time \
  mpirun -np "$SLURM_NTASKS" -ppn "$SLURM_NTASKS_PER_NODE" \
  "$abacus" > abacus.out 2>&1

status=OUT.${suffix}/STERNHEIMER_SIAB_STATUS.dat
dataset=OUT.${suffix}/STERNHEIMER_BASIS_OPT_V1
test -s "$status"
grep -qx 'status success' "$status"
grep -qx 'format basis_opt_v1' "$status"
grep -qx 'abfs_source product_pca' "$status"
grep -qx 'sternheimer_nested_k_frequency_mpi yes' "$status"
grep -qx 'sternheimer_kpoint_groups 4' "$status"
grep -qx 'sternheimer_frequency_groups 12' "$status"
grep -qx 'mpi_ranks 48' "$status"
grep -qx 'status success' "$dataset/status.dat"
grep -qx 'all_converged yes' "$dataset/status.dat"
grep -qx 'auxiliary_basis_source product_pca' "$dataset/manifest.dat"
grep -qx 'kernel full_coulomb' "$dataset/manifest.dat"
grep -qx 'q_count 64' "$dataset/manifest.dat"
grep -qx 'selected_iq 1' "$dataset/manifest.dat"
grep -qx 'frequency_count 12' "$dataset/manifest.dat"

python3 "$abacus_root/validate_periodic_basis_opt.py" \
  "$dataset" --commit "$abacus_commit" | tee BASIS_OPT_VALIDATION.json

python3 - "$dataset/manifest.dat" "$reference_grid" "$orbital_sha256" "$pseudopotential_sha256" "$expected_abacus_sha" \
  > STANDARD_PROTOCOL_VALIDATION.json <<'PY'
import hashlib
import json
import math
import pathlib
import struct
import sys

(
    manifest_path,
    reference_grid_path,
    orbital_sha256,
    pseudopotential_sha256,
    executable_sha256,
) = sys.argv[1:]
metadata = {}
frequencies = []
with open(manifest_path, encoding="ascii") as handle:
    for line in handle:
        fields = line.split()
        if fields and fields[0] == "frequency":
            frequencies.append(fields[1:])
        elif len(fields) >= 2 and fields[0] not in {"kpoint", "eigenvalues_ry", "entry"}:
            metadata[fields[0]] = fields[1:]

def scalar(key):
    values = metadata.get(key, [])
    if len(values) != 1:
        raise RuntimeError("missing or non-scalar manifest field: " + key)
    return values[0]

assert scalar("abacus_commit") == "711af860c125b9757c344a1961b63524c550cfe4"
assert scalar("executable_sha256") == executable_sha256
assert scalar("orbital_sha256") == orbital_sha256
assert scalar("pseudopotential_sha256") == pseudopotential_sha256
assert scalar("auxiliary_basis_source") == "product_pca"
assert scalar("kernel") == "full_coulomb"
assert int(scalar("q_count")) == 64
assert int(scalar("selected_iq")) == 1
assert math.isclose(float(scalar("q_weight")), 1.0 / 64.0, rel_tol=0.0, abs_tol=1.0e-15)
assert int(scalar("frequency_count")) == 12
assert len(frequencies) == 12

digest = hashlib.sha256()
digest.update(b"ABACUS_STERNHEIMER_AUXILIARY_BASIS_DEFINITION_V1")
digest.update(orbital_sha256.encode("ascii"))
digest.update(struct.pack(">d", 1.0e-6))
digest.update(struct.pack(">d", 4.0))
digest.update(struct.pack(">Q", 0))
assert scalar("auxiliary_basis_sha256") == digest.hexdigest()

raw_dimension = int(scalar("raw_auxiliary_dimension"))
retained_rank = int(scalar("whitened_auxiliary_rank"))
assert raw_dimension == 236
assert 0 < retained_rank <= raw_dimension

grid_path = pathlib.Path("FREQUENCY_GRID.tsv")
grid_path.write_text(
    "\n".join(
        "\t".join([str(int(values[0]) + 1), values[1], values[2]])
        for values in frequencies
    ) + "\n",
    encoding="ascii",
)
reference_grid = pathlib.Path(reference_grid_path)
reference_rows = [
    tuple(float(value) for value in line.split())
    for line in reference_grid.read_text(encoding="ascii").splitlines()
]
actual_rows = [
    tuple(float(value) for value in line.split())
    for line in grid_path.read_text(encoding="ascii").splitlines()
]
if len(reference_rows) != 12 or len(actual_rows) != 12:
    raise RuntimeError("frequency grid differs from accepted standard reference")
for reference, actual in zip(reference_rows, actual_rows):
    if reference[0] != actual[0] or any(
        not math.isclose(reference[index], actual[index], rel_tol=1.0e-12, abs_tol=1.0e-14)
        for index in (1, 2)
    ):
        raise RuntimeError("frequency grid differs from accepted standard reference")
print(json.dumps({
    "status": "success",
    "frequency_count": len(frequencies),
    "frequency_grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
    "frequency_reference_sha256": hashlib.sha256(reference_grid.read_bytes()).hexdigest(),
    "product_pca_threshold": 1.0e-6,
    "raw_auxiliary_dimension": raw_dimension,
    "whitened_auxiliary_rank": retained_rank,
    "q_index": 1,
    "q_weight": float(scalar("q_weight")),
}, indent=2, sort_keys=True))
PY

find "$dataset" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUT_SHA256.txt
dataset_bytes=$(du -sb "$dataset" | awk '{print $1}')
{
  echo status=success
  echo dataset_bytes=$dataset_bytes
  echo final_energy_headwing_protocol=qavg_deferred_not_applied_to_training_dataset
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$reference_grid" FREQUENCY_GRID.tsv STANDARD_PROTOCOL_VALIDATION.json "$dataset/manifest.dat" "$dataset/status.dat"
} >> RUN_PROVENANCE.txt

echo "C_SOLID_FD8_Q1_STANDARD_OK work=$work"
