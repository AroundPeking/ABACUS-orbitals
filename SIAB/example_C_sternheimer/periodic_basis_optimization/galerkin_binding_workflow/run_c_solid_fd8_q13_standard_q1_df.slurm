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
validator=$workflow/validate_c_solid_fd8_q_dataset.py
basis_opt_validator=$workflow/validate_periodic_basis_opt_streaming.py
validator_python=${C_BASIS_PYTHON:-/data/home/df_iopcas_ghj/app/miniconda3/bin/python}
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
test -s "$validator"
test -s "$basis_opt_validator"
test -x "$validator_python"
"$validator_python" -c 'import numpy'
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
  echo qstar_scheme=fd8_symmetry_8
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
  sha256sum "$runner" "$config" "$validator" "$basis_opt_validator" "$abacus" INPUT STRU KPT \
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

"$validator_python" "$basis_opt_validator" \
  "$dataset" --commit "$abacus_commit" | tee BASIS_OPT_VALIDATION.json

expected_auxiliary_sha=$(python3 - "$orbital_sha256" <<'PY'
import hashlib
import struct
import sys

digest = hashlib.sha256()
digest.update(b"ABACUS_STERNHEIMER_AUXILIARY_BASIS_DEFINITION_V1")
digest.update(sys.argv[1].encode("ascii"))
digest.update(struct.pack(">d", 1.0e-6))
digest.update(struct.pack(">d", 4.0))
digest.update(struct.pack(">Q", 0))
print(digest.hexdigest())
PY
)

python3 "$validator" \
  --manifest "$dataset/manifest.dat" \
  --status "$dataset/status.dat" \
  --input INPUT \
  --frequency-grid-output FREQUENCY_GRID.tsv \
  --frequency-grid-source greenx_minimax \
  --expected-abacus-commit "$abacus_commit" \
  --expected-executable-sha256 "$expected_abacus_sha" \
  --expected-orbital-sha256 "$orbital_sha256" \
  --expected-pseudopotential-sha256 "$pseudopotential_sha256" \
  --expected-auxiliary-basis-sha256 "$expected_auxiliary_sha" \
  --expected-iq 1 \
  --expected-q-weight 0.015625 \
  > STANDARD_PROTOCOL_VALIDATION.json

find "$dataset" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUT_SHA256.txt
dataset_bytes=$(du -sb "$dataset" | awk '{print $1}')
{
  echo status=success
  echo dataset_bytes=$dataset_bytes
  echo final_energy_headwing_protocol=qavg_deferred_not_applied_to_training_dataset
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$validator" "$basis_opt_validator" FREQUENCY_GRID.tsv \
    STANDARD_PROTOCOL_VALIDATION.json "$dataset/manifest.dat" "$dataset/status.dat"
} >> RUN_PROVENANCE.txt

echo "C_SOLID_FD8_Q1_STANDARD_OK work=$work"
