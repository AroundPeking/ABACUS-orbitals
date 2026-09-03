#!/bin/bash
#SBATCH --job-name=c_fd8_q13_nf12_dcu
#SBATCH --partition=normal
#SBATCH --nodes=48
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=110000
#SBATCH --time=12:00:00

set -euo pipefail
trap 'status=$?; echo "C_SOLID_FD8_DFDCU_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

: "${C_SOLID_DFDCU_STAGE_ROOT:?missing df_dcu stage root}"
: "${OUTPUT_ROOT:?missing immutable output root}"
: "${RUNNER_SHA256:?missing runner SHA256}"
: "${CONFIG_SHA256:?missing standard config SHA256}"
: "${VALIDATOR_SHA256:?missing standard validator SHA256}"
: "${ABACUS_SHA256:?missing df_dcu ABACUS SHA256}"
: "${DF_Q1_FREQUENCY_GRID_SHA256:?missing accepted df q1 frequency-grid SHA256}"
: "${DF_Q1_STANDARD_VALIDATION_SHA256:?missing accepted df q1 validation SHA256}"
: "${DF_Q1_RECOVERY_ACCEPTANCE_SHA256:?missing accepted df q1 recovery SHA256}"

DFDCU_ENV=${DFDCU_ENV:-/public/home/ghj/app/src/env_60_245_intel2021.sh}
DFDCU_ELPA_ROOT=${DFDCU_ELPA_ROOT:-/public/home/ghj/app/deps/elpa-2021.11.002-intelmpi2021}
source "$DFDCU_ENV"

source_commit=711af860c125b9757c344a1961b63524c550cfe4
runner=$C_SOLID_DFDCU_STAGE_ROOT/source/run_c_solid_fd8_q13_standard_dfdcu.slurm
config=$C_SOLID_DFDCU_STAGE_ROOT/source/c_diamond_solid_fd8_q13_standard.json
validator=$C_SOLID_DFDCU_STAGE_ROOT/source/validate_c_solid_fd8_q_dataset.py
abacus_root=$C_SOLID_DFDCU_STAGE_ROOT/artifacts/abacus-711af860c-intel2021
abacus=$abacus_root/abacus_3p
template=$C_SOLID_DFDCU_STAGE_ROOT/gate-template-product-pca
q1_root=$C_SOLID_DFDCU_STAGE_ROOT/inputs/q1
frequency_name=Q1_FREQUENCY_GRID.tsv
orbital_sha256=7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d
pseudopotential_sha256=e95d682a8b918557fb57e2e0ec11b2f48cf693cb72a11d078cf07ec489a8fa99
q_labels=(1 2 3 6 7 8 11 22 23 24 27 28 43)
q_multiplicities=(1 6 3 6 12 6 3 2 6 6 6 6 1)
slot=${SLURM_ARRAY_TASK_ID:?missing array slot}
test "$slot" -ge 0
test "$slot" -lt "${#q_labels[@]}"
iq=${q_labels[$slot]}
multiplicity=${q_multiplicities[$slot]}
work=$OUTPUT_ROOT/q${iq}
suffix=C_DIAMOND_FD8_Q${iq}_NFREQ12_PCA1E6_BASIS_OPT_DFDCU

test "$SLURM_JOB_PARTITION" = normal
test "$SLURM_NTASKS" = 48
test "$SLURM_NTASKS_PER_NODE" = 1
test "$SLURM_CPUS_PER_TASK" = 30
test "$(sha256sum "$runner" | awk '{print $1}')" = "$RUNNER_SHA256"
test "$(sha256sum "$config" | awk '{print $1}')" = "$CONFIG_SHA256"
test "$(sha256sum "$validator" | awk '{print $1}')" = "$VALIDATOR_SHA256"
test -x "$abacus"
test "$(sha256sum "$abacus" | awk '{print $1}')" = "$ABACUS_SHA256"
grep -Fxq 'status=success' "$abacus_root/BUILD_PROVENANCE.txt"
test -d "$template"
test "$(sha256sum "$q1_root/FREQUENCY_GRID.tsv" | awk '{print $1}')" = "$DF_Q1_FREQUENCY_GRID_SHA256"
test "$(sha256sum "$q1_root/STANDARD_PROTOCOL_VALIDATION.json" | awk '{print $1}')" = "$DF_Q1_STANDARD_VALIDATION_SHA256"
test "$(sha256sum "$q1_root/Q1_RECOVERY_ACCEPTANCE.json" | awk '{print $1}')" = "$DF_Q1_RECOVERY_ACCEPTANCE_SHA256"

mkdir -p "$OUTPUT_ROOT"
mkdir "$work"
cp -a "$template/." "$work/"
cp "$q1_root/FREQUENCY_GRID.tsv" "$work/$frequency_name"
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
set_input_key sternheimer_frequency_grid_file "$frequency_name"
set_input_key sternheimer_frequency_mpi 1
set_input_key sternheimer_channel_mpi 0
set_input_key sternheimer_mpi_layout frequency_grouped
set_input_key sternheimer_q_index "$iq"
set_input_key exx_pca_threshold 1e-6
set_input_key exx_ccp_rmesh_times 1
set_input_key rpa_ccp_rmesh_times 2

! grep -q '^ABFS_ORBITAL' STRU
test -z "$(find . -maxdepth 1 -type f -name '*.abfs' -print -quit)"
test "$(sha256sum C_gga_10au_100Ry_3s3p2d.orb | awk '{print $1}')" = "$orbital_sha256"
test "$(sha256sum C_ONCV_PBE-1.0.upf | awk '{print $1}')" = "$pseudopotential_sha256"
test "$(sha256sum "$frequency_name" | awk '{print $1}')" = "$DF_Q1_FREQUENCY_GRID_SHA256"

export LD_LIBRARY_PATH=$DFDCU_ELPA_ROOT/lib:$C_SOLID_DFDCU_STAGE_ROOT/deps/lib:${LD_LIBRARY_PATH:-}
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

if [ "$iq" = 1 ]; then
  cross_host_anchor=yes
else
  cross_host_anchor=no
fi

{
  echo status=running
  echo purpose=c_solid_fd8_q13_standard_df_dcu
  echo source_commit=$source_commit
  echo abacus_sha256=$ABACUS_SHA256
  echo cross_host_anchor=$cross_host_anchor
  echo df_q1_standard_validation_sha256=$DF_Q1_STANDARD_VALIDATION_SHA256
  echo df_q1_recovery_acceptance_sha256=$DF_Q1_RECOVERY_ACCEPTANCE_SHA256
  echo qstar_scheme=fd8_discrete_13
  echo q_index=$iq
  echo q_multiplicity=$multiplicity
  echo q_count=64
  echo frequency_count=12
  echo frequency_grid=frozen_df_q1_greenx_minimax
  echo frequency_grid_sha256=$DF_Q1_FREQUENCY_GRID_SHA256
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
  sha256sum "$runner" "$config" "$validator" "$abacus" INPUT STRU KPT \
    C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_3s3p2d.orb "$frequency_name"
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
grep -qx "selected_iq $iq" "$dataset/manifest.dat"
grep -qx 'frequency_count 12' "$dataset/manifest.dat"

python3 "$C_SOLID_DFDCU_STAGE_ROOT/validate_periodic_basis_opt.py" \
  "$dataset" --commit "$source_commit" | tee BASIS_OPT_VALIDATION.json

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
  --frequency-grid-source frozen_df_q1_greenx_minimax \
  --expected-frequency-grid "$frequency_name" \
  --expected-abacus-commit "$source_commit" \
  --expected-executable-sha256 "$ABACUS_SHA256" \
  --expected-orbital-sha256 "$orbital_sha256" \
  --expected-pseudopotential-sha256 "$pseudopotential_sha256" \
  --expected-auxiliary-basis-sha256 "$expected_auxiliary_sha" \
  --expected-iq "$iq" \
  --expected-q-weight 0.015625 \
  > STANDARD_PROTOCOL_VALIDATION.json

test "$(sha256sum FREQUENCY_GRID.tsv | awk '{print $1}')" = "$DF_Q1_FREQUENCY_GRID_SHA256"
find "$dataset" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUT_SHA256.txt
dataset_bytes=$(du -sb "$dataset" | awk '{print $1}')
{
  echo status=success
  echo dataset_bytes=$dataset_bytes
  echo final_energy_headwing_protocol=qavg_deferred_not_applied_to_training_dataset
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' abacus.time
  sha256sum "$validator" "$frequency_name" FREQUENCY_GRID.tsv \
    STANDARD_PROTOCOL_VALIDATION.json "$dataset/manifest.dat" "$dataset/status.dat"
} >> RUN_PROVENANCE.txt

echo "C_SOLID_FD8_DFDCU_OK q=$iq multiplicity=$multiplicity work=$work"
