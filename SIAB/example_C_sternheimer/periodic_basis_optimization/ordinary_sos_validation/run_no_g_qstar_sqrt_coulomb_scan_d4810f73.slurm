#!/bin/bash
#SBATCH --job-name=c_nog_sqrtv
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --array=1-3
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nog-sqrtv-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nog-sqrtv-%A_%a.err

set -euo pipefail
trap 'status=$?; echo "C_NO_G_SQRT_COULOMB_SCAN_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

thresholds=("1.1e-5" "2e-5" "1e-4")
labels=("1p1e-5" "2e-5" "1e-4")
threshold=${thresholds[$((SLURM_ARRAY_TASK_ID - 1))]}
label=${labels[$((SLURM_ARRAY_TASK_ID - 1))]}
campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source=$campaign/runs/product-pca-20260825/joint-atom-solid-no-g-threshold-qstar-sos-ce2801f3/solid-sos
reader=$source/reader_v1
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
work=$campaign/runs/product-pca-20260825/joint-atom-solid-no-g-sqrtv-${label}-control

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test "$SLURM_ARRAY_TASK_ID" -ge 1
test "$SLURM_ARRAY_TASK_ID" -le 3
test ! -e "$work"
test -d "$reader"
test -s "$source/librpa.in"
test -s "$source/librpa.out"
grep -q 'libRPA finished successfully' "$source/librpa.out"
test -x "$librpa"
test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
grep -qx "driver=$librpa" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "${#expected_librpa_sha}" = 64
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"

mkdir -p "$work"
cd "$work"
cp "$source/librpa.in" librpa.in
sed -i -E "s|^[[:space:]]*sqrt_coulomb_threshold[[:space:]]*=.*|sqrt_coulomb_threshold = $threshold|" librpa.in

grep -Fqx "input_dir = $reader/" librpa.in
grep -Eq '^nfreq[[:space:]]*=[[:space:]]*6$' librpa.in
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_grid_iq_$' librpa.in
grep -Eq '^n_bands_chi0[[:space:]]*=[[:space:]]*-1$' librpa.in
grep -Eq '^replace_w_head[[:space:]]*=[[:space:]]*f$' librpa.in
grep -Fqx "sqrt_coulomb_threshold = $threshold" librpa.in

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=no_g_qstar_sqrt_coulomb_threshold_diagnosis
  echo changed_variable=sqrt_coulomb_threshold
  echo sqrt_coulomb_threshold=$threshold
  echo source_reader=$reader
  echo source_librpa_output=$source/librpa.out
  echo source_total_ecrpa_ha=-3.318544531
  echo exx_pca_threshold=1e-4
  echo nfreq=6
  echo auxiliary_dimension=322
  echo ao_basis=joint_atom_solid_3s3p2d1f_no_g
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo headwing=body_only
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$source/librpa.in" "$source/librpa.out" librpa.in \
    "$build_provenance" "$librpa"
} > provenance.txt

/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1

grep -q 'libRPA finished successfully' librpa.out
grep -q '^RPA correlation energy (Hartree)' librpa.out
grep -q '^| Total EcRPA:' librpa.out
/data/home/df_iopcas_ghj/app/miniconda3/bin/python - librpa.out "$threshold" > RESULT_SUMMARY.txt <<'PY'
import math
import re
import sys

lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
threshold = sys.argv[2]
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
print("scope no_g_qstar_sqrt_coulomb_threshold_diagnosis")
print(f"sqrt_coulomb_threshold {threshold}")
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

echo "C_NO_G_SQRT_COULOMB_SCAN_OK threshold=$threshold work=$work"
