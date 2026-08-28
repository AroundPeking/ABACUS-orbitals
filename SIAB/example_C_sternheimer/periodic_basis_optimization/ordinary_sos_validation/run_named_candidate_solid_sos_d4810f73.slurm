#!/bin/bash
#SBATCH --job-name=c_basis_ssos
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-basis-solid-sos-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-basis-solid-sos-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NAMED_SOLID_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
code=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
manifest=$candidate_root/CANDIDATE.json
grid_root=$run_root/solid-grid
work=$run_root/solid-sos
reader=$work/reader_v1
template=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/librpa_tzdp_grid_full_bz_nfreq6.in
extractor=$code/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/extract_librpa_frequency_grid.py
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt
maximum_reported_discrepancy=7.82805267e-06

test "${SLURM_JOB_PARTITION:?}" = p1 && test "${SLURM_NTASKS:?}" = 1 && test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -s "$manifest" && test -x "$librpa" && test -x "$python" && test -s "$template" && test -s "$extractor"
grep -Eq '^prefix_coul_full[[:space:]]*=[[:space:]]*v1_coulomb_grid_iq_$' "$template"
grep -Eq '^use_symmetry_rpa[[:space:]]*=[[:space:]]*f$' "$template"
grep -Eq '^replace_w_head[[:space:]]*=[[:space:]]*f$' "$template"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"
orbital_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["exported_orbital_sha256"])' "$manifest")

mkdir -p "$reader"
q1=$grid_root/q1
grep -qx 'status=success' "$q1/provenance.txt"
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out; do
  test -s "$q1/$name"
  ln -s "$q1/$name" "$reader/$name"
done
while IFS= read -r -d '' path; do ln -s "$path" "$reader/$(basename "$path")"; done < <(find "$q1" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c -print0 | sort -z)
for ordinal in $(seq 1 64); do
  qdir=$grid_root/q${ordinal}
  grep -qx 'status=success' "$qdir/provenance.txt"
  grep -qx "full_q_ordinal=$ordinal" "$qdir/provenance.txt"
  matrix=$qdir/v1_coulomb_grid_iq_${ordinal}_rank0.dat
  test -s "$matrix" && test -s "$qdir/GRID_MATRIX_HEADER.json"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done
test "$(find -L "$reader" -maxdepth 1 -type f -name 'v1_coulomb_grid_iq_*_rank*.dat' -size +0c | wc -l)" -eq 64
test "$(find -L "$reader" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)" -eq 1
naux=$(python3 - "$grid_root" "$reader/bz_sampling_out" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
dimensions = {json.loads((root / f"q{ordinal}" / "GRID_MATRIX_HEADER.json").read_text(encoding="ascii"))["naux"] for ordinal in range(1, 65)}
assert len(dimensions) == 1
naux = dimensions.pop()
assert naux > 0 and naux % 2 == 0
lines = pathlib.Path(sys.argv[2]).read_text(encoding="ascii").splitlines()
assert [int(value) for value in lines[1].split()] == [64, 64]
assert [(int(row.split()[-2]), int(row.split()[-1])) for row in lines[2:]] == [(index, index) for index in range(1, 65)]
print(naux)
PY
)

sed "s|__INPUT_DIR__|$reader|" "$template" > "$work/librpa.in"
if grep -Eq '^n_bands_chi0[[:space:]]*=' "$work/librpa.in"; then
  sed -i -E 's|^n_bands_chi0[[:space:]]*=.*|n_bands_chi0 = -1|' "$work/librpa.in"
else
  echo 'n_bands_chi0 = -1' >> "$work/librpa.in"
fi
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
  echo purpose=named_candidate_c_solid_all_band_sos
  echo grid_coulomb_array_job=${GRID_COULOMB_ARRAY_JOB_ID:?missing grid array job id}
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  echo selected_orbital_sha256=$orbital_sha
  echo naux=$naux
  echo pca_threshold=1e-4
  echo nfreq=6
  echo n_bands_chi0=-1
  echo kmesh=4,4,4
  echo full_q_count=64
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo mpi_ranks=$SLURM_NTASKS
  echo omp_threads=$OMP_NUM_THREADS
  echo mkl_threads=$MKL_NUM_THREADS
  sha256sum "$0" "$manifest" "$template" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
grep -Fq "Canonicalized BZ Cartesian k-vectors from fractional coordinates; maximum reported discrepancy = $maximum_reported_discrepancy" librpa.out
grep -q 'libRPA finished successfully' librpa.out
grep -q '^| Total EcRPA:' librpa.out
"$python" "$extractor" --source librpa.out --output SOLID_SOS_FREQUENCY_GRID.dat --manifest SOLID_SOS_FREQUENCY_GRID.json --expected-nfreq 6 > frequency_extract.out
reference_ha=$(awk '/Etot_without_rpa\(Ha\):/ {value=$2} END {print value}' "$q1/abacus.out")
ecrpa_ha=$(awk '/^\| Total EcRPA:/ {value=$NF} END {print value}' librpa.out)
frequency_sha=$(sha256sum SOLID_SOS_FREQUENCY_GRID.dat | awk '{print $1}')
python3 - "$reference_ha" "$ecrpa_ha" librpa.out <<'PY'
import math, re, sys
assert math.isfinite(float(sys.argv[1])) and math.isfinite(float(sys.argv[2]))
lines = open(sys.argv[3], encoding="utf-8", errors="replace").read().splitlines()
inside, contributions = False, []
for line in lines:
    if line.strip() == "RPA correlation energy (Hartree)":
        inside = True
        continue
    if inside:
        match = re.search(r": \(([-+0-9.eE]+),([-+0-9.eE]+)\)\s*$", line)
        if match:
            contributions.append((float(match.group(1)), float(match.group(2))))
        if line.startswith("| Total EcRPA:"):
            break
assert len(contributions) == 64, len(contributions)
assert max(abs(value[1]) for value in contributions) <= 1e-10
PY
cat > RESULT_SUMMARY.txt <<EOF
status success
side solid
method sos
scope body_only_no_analytic_headwing
coulomb_kernel full_periodic_poisson
selected_orbital_sha256 $orbital_sha
frequency_grid_sha256 $frequency_sha
naux $naux
reference_ha $reference_ha
ecrpa_ha $ecrpa_ha
EOF
{
  echo status=success
  echo frequency_grid_sha256=$frequency_sha
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt SOLID_SOS_FREQUENCY_GRID.dat SOLID_SOS_FREQUENCY_GRID.json frequency_extract.out
} >> provenance.txt
echo "C_NAMED_SOLID_SOS_OK work=$work ecrpa_ha=$ecrpa_ha"
