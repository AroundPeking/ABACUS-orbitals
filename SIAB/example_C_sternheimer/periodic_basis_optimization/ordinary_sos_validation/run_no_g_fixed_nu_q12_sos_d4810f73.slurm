#!/bin/bash
#SBATCH --job-name=c_nog_fixnu_sos
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nog-fixnu-sos-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nog-fixnu-sos-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NO_G_FIXED_NU_Q12_SOS_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-ce2801f3
run_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-g-fixed-nu-q12-control
grid_root=$run_root/solid-grid
work=$run_root/solid-sos
reader=$work/reader_v1
zero_writer=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/sparse_qstar_coulomb.py
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
librpa_root=/data/home/df_iopcas_ghj/app/librpa/LibRPA-rpa-logdet-bz-exact-d4810f73/build-df-intel
librpa=$librpa_root/chi0_main.exe
librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9
build_provenance=$librpa_root/PROVENANCE-d4810f73.txt

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -s "$zero_writer"
test -x "$librpa" && test -s "$build_provenance"
grep -qx "source_commit=$librpa_commit" "$build_provenance"
expected_librpa_sha=$(tail -1 "$build_provenance" | awk '{print $1}')
test "$(sha256sum "$librpa" | awk '{print $1}')" = "$expected_librpa_sha"

mkdir -p "$reader" "$work/zero-manifests"
q1=$grid_root/q1
q2=$grid_root/q2
for qdir in "$q1" "$q2"; do
  grep -qx 'success' "$qdir/STATUS"
  grep -Eq '^rpa_pca_fixed_nu[[:space:]]+2,2,1,0$' "$qdir/INPUT"
done
for name in band_out basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat stru_out; do
  test -s "$q1/$name"
  ln -s "$q1/$name" "$reader/$name"
done
while IFS= read -r -d '' path; do
  ln -s "$path" "$reader/$(basename "$path")"
done < <(find "$q1" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c -print0 | sort -z)
test "$(find -L "$reader" -maxdepth 1 -type f -name 'v1_Cs_data_*.txt' -size +0c | wc -l)" -eq 1

for q_index in 1 2; do
  qdir=$grid_root/q${q_index}
  matrix=$qdir/v1_coulomb_grid_iq_${q_index}_rank0.dat
  test -s "$matrix"
  ln -s "$matrix" "$reader/$(basename "$matrix")"
done
template=$q1/v1_coulomb_grid_iq_1_rank0.dat
for q_index in $(seq 3 64); do
  "$python" "$zero_writer" \
    --source "$template" \
    --output "$reader/v1_coulomb_grid_iq_${q_index}_rank0.dat" \
    --iq "$q_index" \
    --manifest "$work/zero-manifests/q${q_index}.json" > /dev/null
done
test "$(find -L "$reader" -maxdepth 1 -type f -name 'v1_coulomb_grid_iq_*_rank0.dat' -size +0c | wc -l)" -eq 64

cat > "$work/librpa.in" <<EOF
input_dir = $reader/
task = rpa
nfreq = 6
tfgrids_type = minimax
parallel_routing = libri
version_coul_reader = 1
version_lri_reader = 1
prefix_coul_full = v1_coulomb_grid_iq_
prefix_lri_coeff = v1_Cs_data_
use_scalapack_ecrpa = t
use_shrink_abfs = f
use_shrink_chi = f
n_bands_chi0 = -1
vq_threshold = 0
sqrt_coulomb_threshold = 1e-5
cs_threshold = 0
use_fullcoul_exx = t
use_symmetry_exx = f
use_symmetry_rpa = f
use_symmetry_gw = f
replace_w_head = f
debug = f
EOF
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
  echo purpose=no_g_fixed_nu_q1_q2_sparse_sos_control
  echo q12_array_job=${Q12_ARRAY_JOB_ID:?missing q1/q2 array job id}
  echo changed_variable=product_pca_fixed_radial_profiles
  echo rpa_pca_fixed_nu=2,2,1,0
  echo pca_threshold=1e-4
  echo nfreq=6
  echo n_bands_chi0=-1
  echo coulomb_kernel=exact_rhs_full_periodic_poisson
  echo librpa_commit=$librpa_commit
  echo librpa_sha256=$expected_librpa_sha
  sha256sum "$0" "$zero_writer" "$work/librpa.in" "$work/INPUT_SHA256.txt" "$build_provenance" "$librpa"
} > "$work/provenance.txt"

cd "$work"
/usr/bin/time -v -o librpa.time mpirun -np 1 "$librpa" > librpa.out 2>&1
grep -q 'libRPA finished successfully' librpa.out
grep -q '^| Total EcRPA:' librpa.out
"$python" - librpa.out > RESULT_SUMMARY.txt <<'PY'
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
assert max(abs(value[1]) for value in contributions) <= 1.0e-10
assert max(abs(value[0]) for value in contributions[2:]) == 0.0
print("status success")
print("scope no_g_fixed_nu_q1_q2_sparse_sos_control")
print("rpa_pca_fixed_nu 2,2,1,0")
print("gamma_contribution_ha {:.12f}".format(contributions[0][0]))
print("q2_contribution_ha {:.12f}".format(contributions[1][0]))
print("sparse_total_ecrpa_ha {:.12f}".format(total))
PY
{
  echo status=success
  sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*/wall_clock=/p; s/^[[:space:]]*Maximum resident set size (kbytes):[[:space:]]*/max_rss_kb=/p' librpa.time
  sha256sum librpa.in librpa.out RESULT_SUMMARY.txt INPUT_SHA256.txt
} >> provenance.txt
printf 'success\n' > STATUS

echo "C_NO_G_FIXED_NU_Q12_SOS_OK work=$work"
