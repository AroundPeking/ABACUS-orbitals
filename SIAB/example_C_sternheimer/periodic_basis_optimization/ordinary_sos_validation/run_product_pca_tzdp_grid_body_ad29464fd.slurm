#!/bin/bash
#SBATCH --job-name=c24_tzdp_gridrpa
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-tzdp-gridrpa-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-tzdp-gridrpa-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_TZDP_GRID_RPA_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
grid_root=$campaign/runs/product-pca-20260825/grid-coulomb-original-tzdp-ad29464fd
template=$campaign/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/librpa_tzdp_grid_nfreq6.in
librpa=/data/home/df_iopcas_ghj/app/oml-pinned/LibRPA-v0.7.0-dd169fa/build-df-oml/chi0_main.exe
expected_librpa_sha=7af5e426933283a2aa08b8771692bc770506ce3b6f00ccf8545108cb8e58d3bc
expected_full_q_indices=(1 22 43 6 27 23 11 55)
work=$campaign/runs/product-pca-20260825/ordinary-sos-original-tzdp-grid-body-nfreq6-ad29464fd
reader=$work/reader_v1

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -x "$librpa"
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -s "$template"
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_grid_iq_$' "$template"

mkdir -p "$reader"
q1=$grid_root/q1
grep -qx 'status=success' "$q1/provenance.txt"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out; do
  test -s "$q1/$name"
  ln -s "$q1/$name" "$reader/$name"
done
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(find "$q1" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c -print0 | sort -z)

for ordinal in $(seq 1 8); do
  qdir=$grid_root/q${ordinal}
  full_q_index=${expected_full_q_indices[$((ordinal - 1))]}
  matrix=$qdir/v1_coulomb_grid_iq_${ordinal}_rank0.dat
  test -s "$matrix"
  grep -qx 'status=success' "$qdir/provenance.txt"
  grep -qx "irreducible_q_ordinal=$ordinal" "$qdir/provenance.txt"
  grep -qx "full_q_index=$full_q_index" "$qdir/provenance.txt"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done

grid_coulomb_file_count=$(find -L "$reader" -maxdepth 1 -type f -name 'v1_coulomb_grid_iq_*_rank*.dat' -size +0c | wc -l)
cs_file_count=$(find -L "$reader" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)
test "$grid_coulomb_file_count" -eq 8
test "$cs_file_count" -ge 8
python3 - "$reader/bz_sampling_out" <<'PY'
import sys

lines = open(sys.argv[1], encoding="ascii").read().splitlines()
counts = [int(value) for value in lines[1].split()]
assert counts == [8, 8]
labels = [(int(row.split()[-2]), int(row.split()[-1])) for row in lines[2:]]
assert labels == [(index, index) for index in range(1, 9)]
PY

sed "s|__INPUT_DIR__|$reader|" "$template" > "$work/librpa.in"
find -L "$reader" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work/INPUT_SHA256.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi

{
  echo status=running
  echo purpose=periodic_c_original_tzdp_all_band_sos_exact_grid_poisson_body
  echo grid_coulomb_producer_job=${GRID_COULOMB_ARRAY_JOB_ID:?missing grid Coulomb array job id}
  echo abacus_commit=ad29464fdc9125114cc052840415647a1b5a6ef1
  echo librpa_commit=dd169fa11fa920d580d4f39dc11e218a7f17f7b5
  echo orbital_basis=C_TZDP_10au_3s3p2d
  echo auxiliary_basis_source=product_pca
  echo pca_threshold=1e-4
  echo nfreq=6
  echo kmesh=4,4,4
  echo irreducible_q_count=8
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo analytic_headwing=no_body_screen
  echo grid_coulomb_file_count=$grid_coulomb_file_count
  echo cs_file_count=$cs_file_count
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  sha256sum "$0" "$template" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1

grep -q '^RPA correlation energy (Hartree)' librpa.out
grep -q '^| Total EcRPA:' librpa.out
python3 - librpa.out > RESULT_SUMMARY.txt <<'PY'
import math
import re
import sys

lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
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
assert len(contributions) == 8, len(contributions)
maximum_imaginary = max(abs(value[1]) for value in contributions)
assert maximum_imaginary <= 1.0e-10, maximum_imaginary
assert abs(total) < 2.0, total
print("status success")
print("scope body_only_no_analytic_headwing")
print("coulomb_kernel exact_rhs_full_periodic_poisson")
print("qpoint_count 8")
print("maximum_q_imaginary_ha {:.16e}".format(maximum_imaginary))
print("ecrpa_ha {:.12f}".format(total))
PY

{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt
} >> provenance.txt

echo "C24_TZDP_GRID_RPA_OK work=$work"
