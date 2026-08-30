#!/bin/bash
#SBATCH --job-name=c_nog_lri_v
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nog-lri-v-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nog-lri-v-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NO_G_LRI_COULOMB_CONTROL_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source=$campaign/runs/product-pca-20260825/joint-atom-solid-no-g-threshold-qstar-sos-ce2801f3
reader=$source/solid-grid/q1
grid_reference=$source/solid-sos
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
work=$campaign/runs/product-pca-20260825/joint-atom-solid-no-g-lri-coulomb-control

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -d "$reader"
test -s "$grid_reference/librpa.in"
test -s "$grid_reference/librpa.out"
grep -q 'libRPA finished successfully' "$grid_reference/librpa.out"
test "$(find "$reader" -maxdepth 1 -type f -name 'v1_coulomb_full_iq_*_rank0.dat' -size +0c | wc -l)" -eq 64
test -s "$reader/v1_Cs_data_0.txt"
test -s "$reader/KS_eigenvector_0.dat"
test -x "$librpa"
test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
grep -qx "driver=$librpa" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "${#expected_librpa_sha}" = 64
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"

mkdir -p "$work"
cd "$work"
cp "$grid_reference/librpa.in" librpa.in
sed -i -E \
  -e "s|^[[:space:]]*input_dir[[:space:]]*=.*|input_dir = $reader/|" \
  -e 's|^[[:space:]]*prefix_coul_full[[:space:]]*=.*|prefix_coul_full = v1_coulomb_full_iq_|' \
  librpa.in

grep -Fqx "input_dir = $reader/" librpa.in
grep -Eq '^nfreq[[:space:]]*=[[:space:]]*6$' librpa.in
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_full_iq_$' librpa.in
grep -Eq '^n_bands_chi0[[:space:]]*=[[:space:]]*-1$' librpa.in
grep -Eq '^replace_w_head[[:space:]]*=[[:space:]]*f$' librpa.in
grep -Eq '^sqrt_coulomb_threshold[[:space:]]*=[[:space:]]*1e-5$' librpa.in

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=no_g_grid_vs_lri_coulomb_control
  echo changed_variable=coulomb_matrix_source
  echo source_reader=$reader
  echo reference_grid_librpa_output=$grid_reference/librpa.out
  echo reference_grid_ecrpa_ha=-3.318544531
  echo exx_pca_threshold=1e-4
  echo nfreq=6
  echo auxiliary_dimension=322
  echo ao_basis=joint_atom_solid_3s3p2d1f_no_g
  echo coulomb_kernel=abacus_lri_full_periodic
  echo comparison_kernel=exact_rhs_full_periodic_poisson
  echo headwing=body_only
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$grid_reference/librpa.in" "$grid_reference/librpa.out" librpa.in \
    "$reader/v1_Cs_data_0.txt" "$reader/KS_eigenvector_0.dat" \
    "$build_provenance" "$librpa"
} > provenance.txt

/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1

grep -q 'libRPA finished successfully' librpa.out
grep -q '^RPA correlation energy (Hartree)' librpa.out
grep -q '^| Total EcRPA:' librpa.out
/data/home/df_iopcas_ghj/app/miniconda3/bin/python - librpa.out > RESULT_SUMMARY.txt <<'PY'
import math
import re

lines = open("librpa.out", encoding="utf-8", errors="replace").read().splitlines()
contributions = []
total = None
inside = False
for line in lines:
    if line.strip() == "RPA correlation energy (Hartree)":
        inside = True
        continue
    if not inside:
        continue
    total_match = re.match(r"^\| Total EcRPA:\s+([-+0-9.eE]+)\s*$", line)
    if total_match:
        total = float(total_match.group(1))
        break
    contribution = re.search(r": \(([-+0-9.eE]+),([-+0-9.eE]+)\)\s*$", line)
    if contribution:
        contributions.append((float(contribution.group(1)), float(contribution.group(2))))

assert total is not None and math.isfinite(total)
assert len(contributions) == 64, len(contributions)
maximum_imaginary = max(abs(value[1]) for value in contributions)
assert maximum_imaginary <= 1.0e-10, maximum_imaginary
print("status success")
print("scope no_g_grid_vs_lri_coulomb_control")
print("coulomb_kernel abacus_lri_full_periodic")
print("qpoint_count 64")
print("gamma_contribution_ha {:.12f}".format(contributions[0][0]))
print("maximum_q_imaginary_ha {:.16e}".format(maximum_imaginary))
print("ecrpa_ha {:.12f}".format(total))
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt
} >> provenance.txt

echo "C_NO_G_LRI_COULOMB_CONTROL_OK work=$work"
