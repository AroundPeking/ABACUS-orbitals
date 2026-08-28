#!/bin/bash
#SBATCH --job-name=c_atom_delta_rpa
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-delta-rpa-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-delta-rpa-%j.err

set -euo pipefail
trap 'status=$?; echo "C_SELECTED_ATOM_DELTA_RPA_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
base=$campaign/runs/product-pca-20260825
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
validation=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation
atom_root=$base/selected-c-atom-response-aware-product-pca-55d25e3c9
producer=$atom_root/producer-final
sos=$atom_root/sos-nfreq6-d4810f73
response=$atom_root/matched-delta-nfreq6-55d25e3c9
frequency=$sos/ATOM_SOS_FREQUENCY_GRID.dat
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
expected_naux=261
expected_orbital_sha=3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$atom_root/matched-delta-reader-nfreq6-d4810f73
reader=$work/reader_v1

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -d "$validation" && test -s "$frequency"
test -x "$python" && test -x "$librpa" && test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -d "$response" && test -d "$producer" && test -d "$sos"
grep -qx 'status=success' "$response/provenance.txt"
grep -qx "abfs_channels=$expected_naux" "$response/provenance.txt"
grep -qx 'all_converged=yes' "$response/provenance.txt"
frequency_sha=$(sha256sum "$frequency" | awk '{print $1}')
grep -qx "frequency_grid_sha256=$frequency_sha" "$response/provenance.txt"

mkdir -p "$reader"
for name in basis_aux_out basis_out basis_wfc_out; do
  test -s "$response/$name"
  ln -s "$response/$name" "$reader/$name"
done
test -s "$producer/v1_coulomb_full_iq_1_rank0.dat"
ln -s "$producer/v1_coulomb_full_iq_1_rank0.dat" "$reader/v1_coulomb_full_iq_1_rank0.dat"
for matrix in "$response"/v1_sternheimer_chi0_iq_1_ifreq_*_rank*.dat; do
  test -s "$matrix"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done
test "$(find -L "$reader" -maxdepth 1 -type f -name 'v1_sternheimer_chi0_iq_1_ifreq_*_rank*.dat' | wc -l)" -eq 6
cmp -s "$response/basis_aux_out" "$producer/basis_aux_out"
cmp -s "$response/basis_out" "$producer/basis_out"
cmp -s "$response/basis_wfc_out" "$producer/basis_wfc_out"

cat > "$work/librpa.in" <<EOF
task = sternheimer_rpa
input_dir = $reader/
version_coul_reader = 1
nfreq = 6
sqrt_coulomb_threshold = 1e-5
prefix_coul_full = v1_coulomb_full_iq_
prefix_sternheimer_chi0 = v1_sternheimer_chi0_iq_
sternheimer_iq = 1
sternheimer_qweight = 1
EOF

find -L "$reader" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work/INPUT_SHA256.txt"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=selected_c_atom_matched_delta_st_body_rpa
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo selected_orbital_sha256=$expected_orbital_sha
  echo frequency_grid_source=atom_sos_exact
  echo frequency_grid_sha256=$frequency_sha
  echo auxiliary_dimension=$expected_naux
  echo nfreq=6
  echo spin_state=fixed_triplet_nup3_ndown1
  echo qpoint_count=1
  echo coulomb_kernel=full_periodic_poisson
  echo coulomb_matrix_source=producer_full_periodic_ewald
  echo analytic_headwing=no_atom_supercell
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$frequency" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
grep -q 'libRPA finished successfully' librpa.out
grep -q 'Total Sternheimer EcRPA:' librpa.out
ecrpa_ha=$(awk '/Total Sternheimer EcRPA:/ {value=$(NF-1)} END {print value}' librpa.out)
reference_ha=$(awk -F= '$1=="reference_ha" {print $2}' "$response/provenance.txt")
test -n "$ecrpa_ha" && test -n "$reference_ha"
"$python" - "$reference_ha" "$ecrpa_ha" "$frequency_sha" <<'PY' > RESULT_SUMMARY.txt
import math
import sys
reference = float(sys.argv[1])
ecrpa = float(sys.argv[2])
assert math.isfinite(reference) and math.isfinite(ecrpa)
print("status success")
print("side atom")
print("method delta_st")
print("scope body_only_no_analytic_headwing")
print("coulomb_kernel full_periodic_poisson")
print("selected_orbital_sha256 3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1")
print(f"frequency_grid_sha256 {sys.argv[3]}")
print("frequency_grid_source atom_sos_exact")
print("naux 261")
print(f"reference_ha {reference:.15f}")
print(f"ecrpa_ha {ecrpa:.15f}")
PY

{
  echo status=success
  echo ecrpa_ha=$ecrpa_ha
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt
} >> provenance.txt

echo "C_SELECTED_ATOM_DELTA_RPA_OK work=$work ecrpa_ha=$ecrpa_ha"
