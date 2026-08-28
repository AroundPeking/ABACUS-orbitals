#!/bin/bash
#SBATCH --job-name=c24_rsp_pca_gridrpa
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-response-pca-gridrpa-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c24-response-pca-gridrpa-%j.err

set -euo pipefail
trap 'status=$?; echo "C24_RESPONSE_PCA_READER_RPA_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
selection_root=$campaign/runs/product-pca-20260825/selected-fixed-prefix-checkpoint-500
selection=$selection_root/SELECTED_CANDIDATE.json
q1_root=$campaign/runs/product-pca-20260825/response-aware-product-pca-q1-55d25e3c9
q1_job=3137632
grid_root=$campaign/runs/product-pca-20260825/response-aware-product-pca-full-bz-55d25e3c9
expected_naux=522
template=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/librpa_tzdp_grid_full_bz_nfreq6.in
frequency_extractor=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/extract_librpa_frequency_grid.py
frequency_python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
librpa_base_commit=dd169fa11fa920d580d4f39dc11e218a7f17f7b5
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
maximum_reported_discrepancy=7.82805267e-06
work=$campaign/runs/product-pca-20260825/ordinary-sos-response-aware-full-bz-nfreq6-d4810f73
reader=$work/reader_v1

test "$SLURM_JOB_PARTITION" = p1
test "$SLURM_NTASKS" = 1
test "$SLURM_CPUS_PER_TASK" = 40
test ! -e "$work"
test -s "$selection"
test -x "$librpa"
test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
grep -qx "driver=$librpa" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "${#expected_librpa_sha}" = 64
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
test -s "$template"
test -s "$frequency_extractor"
test -x "$frequency_python"
grep -Eq '^use_scalapack_ecrpa[[:space:]]*=[[:space:]]*t$' "$template"
grep -Eq '^use_symmetry_rpa[[:space:]]*=[[:space:]]*f$' "$template"
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_grid_iq_$' "$template"
grep -Eq '^replace_w_head[[:space:]]*=[[:space:]]*f$' "$template"

read -r q1_state q1_exit < <(sacct -n -X -j "$q1_job" --format=State,ExitCode -P | awk -F'|' 'NF>=2 && $1!="" {print $1, $2; exit}')
test "$q1_state" = COMPLETED
test "$q1_exit" = 0:0
grep -qx 'status=success' "$q1_root/provenance.txt"
python3 - "$q1_root/GRID_MATRIX_HEADER.json" "$expected_naux" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert payload["status"] == "success"
assert payload["iq"] == 1
assert payload["naux"] == int(sys.argv[2])
PY

mkdir -p "$reader"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out; do
  test -s "$q1_root/$name"
  ln -s "$q1_root/$name" "$reader/$name"
done
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(find "$q1_root" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c -print0 | sort -z)

for ordinal in $(seq 1 64); do
  if test "$ordinal" -eq 1; then
    qdir=$q1_root
  else
    qdir=$grid_root/q${ordinal}
    grep -qx 'status=success' "$qdir/provenance.txt"
    grep -qx "full_q_ordinal=$ordinal" "$qdir/provenance.txt"
    grep -qx "full_q_index=$ordinal" "$qdir/provenance.txt"
  fi
  matrix=$qdir/v1_coulomb_grid_iq_${ordinal}_rank0.dat
  test -s "$matrix"
  test -s "$qdir/GRID_MATRIX_HEADER.json"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done

grid_coulomb_file_count=$(find -L "$reader" -maxdepth 1 -type f -name 'v1_coulomb_grid_iq_*_rank*.dat' -size +0c | wc -l)
cs_file_count=$(find -L "$reader" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)
test "$grid_coulomb_file_count" -eq 64
test "$cs_file_count" -eq 1
auxiliary_dimension=$(python3 - "$q1_root" "$grid_root" "$reader/bz_sampling_out" "$expected_naux" <<'PY'
import json
import pathlib
import sys

q1_root = pathlib.Path(sys.argv[1])
grid_root = pathlib.Path(sys.argv[2])
expected_naux = int(sys.argv[4])
dimensions = {
    json.loads(
        ((q1_root if ordinal == 1 else grid_root / f"q{ordinal}") / "GRID_MATRIX_HEADER.json").read_text(encoding="ascii")
    )["naux"]
    for ordinal in range(1, 65)
}
assert dimensions == {expected_naux}
lines = pathlib.Path(sys.argv[3]).read_text(encoding="ascii").splitlines()
counts = [int(value) for value in lines[1].split()]
assert counts == [64, 64]
labels = [(int(row.split()[-2]), int(row.split()[-1])) for row in lines[2:]]
assert labels == [(index, index) for index in range(1, 65)]
print(expected_naux)
PY
)

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
  echo purpose=periodic_c_response_aware_product_pca_all_band_sos
  echo q1_grid_coulomb_job=$q1_job
  echo remaining_grid_coulomb_job=${GRID_COULOMB_ARRAY_JOB_ID:?missing q2-q64 grid Coulomb array job id}
  echo abacus_commit=55d25e3c9bb8255fb17bcdf127c653a72402a5d5
  echo librpa_commit=$librpa_commit
  echo librpa_base_commit=$librpa_base_commit
  echo librpa_feature_branch_exception=rpa_logdet_bz_exact
  echo librpa_sha256=$expected_librpa_sha
  echo auxiliary_basis_source=response_aware_fixed_prefix_times_all_ao_product_pca
  echo auxiliary_dimension=$auxiliary_dimension
  echo fixed_nu=2,2,1,0,0
  echo pca_threshold=1e-4
  echo nfreq=6
  echo kmesh=4,4,4
  echo symmetry_rpa=false
  echo full_q_count=64
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo determinant_route=scalapack_pivot_parity_fixed
  echo bz_coordinate_source=fractional_columns
  echo maximum_reported_discrepancy=$maximum_reported_discrepancy
  echo analytic_headwing=no_body_screen
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  echo mkl_threads=$MKL_NUM_THREADS
  sha256sum "$0" "$selection" "$template" "$work/librpa.in" \
    "$work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1

grep -Fq "Canonicalized BZ Cartesian k-vectors from fractional coordinates; maximum reported discrepancy = $maximum_reported_discrepancy" librpa.out
grep -q '^RPA correlation energy (Hartree)' librpa.out
grep -q '^| Total EcRPA:' librpa.out
"$frequency_python" "$frequency_extractor" \
  --source librpa.out \
  --output SELECTED_SOS_FREQUENCY_GRID.dat \
  --manifest SELECTED_SOS_FREQUENCY_GRID.json \
  --expected-nfreq 6 > frequency_extract.out
python3 - SELECTED_SOS_FREQUENCY_GRID.json <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert payload["status"] == "success"
assert payload["nfreq"] == 6
assert len(payload["frequencies_ha"]) == 6
assert len(payload["weights_ha"]) == 6
PY
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
assert len(contributions) == 64, len(contributions)
maximum_imaginary = max(abs(value[1]) for value in contributions)
assert maximum_imaginary <= 1.0e-10, maximum_imaginary
print("status success")
print("scope body_only_no_analytic_headwing")
print("auxiliary_basis response_aware_fixed_prefix_product_pca")
print("auxiliary_dimension 522")
print("coulomb_kernel exact_rhs_full_periodic_poisson")
print("determinant_route scalapack_pivot_parity_fixed")
print("bz_coordinate_source fractional_columns")
print("qpoint_count 64")
print("maximum_q_imaginary_ha {:.16e}".format(maximum_imaginary))
print("q_real_sum_ha {:.12f}".format(sum(value[0] for value in contributions)))
print("ecrpa_ha {:.12f}".format(total))
PY

{
  echo status=success
  echo frequency_grid_source=response_aware_selected_candidate_sos
  echo matched_delta_st_frequency_file=SELECTED_SOS_FREQUENCY_GRID.dat
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt \
    SELECTED_SOS_FREQUENCY_GRID.dat SELECTED_SOS_FREQUENCY_GRID.json \
    frequency_extract.out "$frequency_extractor"
} >> provenance.txt

echo "C24_RESPONSE_PCA_READER_RPA_OK work=$work"
