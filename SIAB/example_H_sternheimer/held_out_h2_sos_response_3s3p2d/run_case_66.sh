#!/bin/bash

set -euo pipefail
source "$HOME/.bashrc" >/dev/null 2>&1 || true

campaign_root=${1:?campaign root is required}
lane=${2:?lane is required}
case_name=${3:?case name is required}
threads=${4:?OpenMP thread count is required}
runtime=${RUNTIME_ROOT:-/home/ghj/sternheimer_abacus/verification/runtime-h2-sos-80a606f5-d70b5571}
python=${PYTHON:-/home/ghj/app/miniconda3/envs/abacus-orbitals/bin/python}
case_dir="$campaign_root/$lane/$case_name"
manifest="$campaign_root/campaign.json"
abacus="$runtime/bin/abacus-80a606f57a26"
librpa="$runtime/bin/chi0_main.exe"

test -d "$case_dir"
test -s "$manifest"
test -x "$python"
test -x "$abacus"
test -x "$librpa"
test "$(sha256sum "$abacus" | awk '{print $1}')" = \
  2e6441a67a1ad19c18538bd4134a97ca6f7b028cd5ccbc46fabea946d899728d
test "$(sha256sum "$librpa" | awk '{print $1}')" = \
  defb442582891a0ceeb3618b95f13f863bfacdac28ca01ecdf5f06ba278a6a9c

read -r nbands nspin nelec suffix orbital pseudo auxiliary < <(
  "$python" - "$manifest" "$case_dir" "$lane" "$case_name" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest_path = Path(sys.argv[1])
case_dir = Path(sys.argv[2]).resolve()
lane = sys.argv[3]
case_name = sys.argv[4]
data = json.loads(manifest_path.read_text(encoding="ascii"))
matches = [
    item for item in data["cases"]
    if item["lane"] == lane and item["case"] == case_name
]
if len(matches) != 1:
    raise SystemExit("campaign manifest case lookup failed")
entry = matches[0]
if Path(entry["directory"]).resolve() != case_dir:
    raise SystemExit("campaign manifest directory mismatch")
for name, expected in entry["input_sha256"].items():
    actual = sha256(case_dir / name)
    if actual != expected:
        raise SystemExit(f"{name} SHA256 expected {expected}, got {actual}")

filenames = data["asset_filenames"]
hashes = data["asset_sha256"]
orbital_key = entry["orbital_asset_key"]
for key in (orbital_key, "pseudopotential", "auxiliary_basis"):
    path = case_dir / filenames[key]
    actual = sha256(path)
    if actual != hashes[key]:
        raise SystemExit(f"{key} SHA256 expected {hashes[key]}, got {actual}")

values = {}
for line in (case_dir / "INPUT").read_text(encoding="ascii").splitlines():
    fields = line.split()
    if len(fields) >= 2 and fields[0] != "INPUT_PARAMETERS":
        values[fields[0]] = fields[1]
if int(values["nbands"]) != entry["nbands"]:
    raise SystemExit("INPUT nbands mismatch")
if int(values.get("nspin", "1")) != entry["nspin"]:
    raise SystemExit("INPUT nspin mismatch")
if int(values["nelec"]) != entry["nelec"]:
    raise SystemExit("INPUT nelec mismatch")
print(
    entry["nbands"],
    entry["nspin"],
    entry["nelec"],
    values["suffix"],
    filenames[orbital_key],
    filenames["pseudopotential"],
    filenames["auxiliary_basis"],
)
PY
)

output_dir="OUT.$suffix"
test ! -e "$case_dir/$output_dir"
test ! -e "$case_dir/band_out"
test ! -e "$case_dir/basis_wfc_out"
test ! -e "$case_dir/basis_aux_out"
test ! -e "$case_dir/sos_full_nfreq16_dump"

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
export LD_LIBRARY_PATH="$runtime/lib:${LD_LIBRARY_PATH:-}"

cd "$case_dir"
{
  echo "date_start=$(date +%F_%T)"
  echo "host=$(hostname)"
  echo "lane=$lane"
  echo "case=$case_name"
  echo "omp_threads=$threads"
  echo "nbands=$nbands"
  echo "nspin=$nspin"
  echo "nelec=$nelec"
  echo "abacus_source_commit=80a606f57a2610bc2532468661b687b01f58074c"
  echo "librpa_reference_commit=d70b55714515c4c6446a87fa96fd1fd62c7f6fde"
  sha256sum INPUT STRU KPT librpa.in "$orbital" "$pseudo" "$auxiliary" \
    "$abacus" "$librpa"
} > RUN_PROVENANCE.txt

/usr/bin/time -v -o abacus.time \
  mpirun -np 1 -ppn 1 "$abacus" > abacus.stdout 2> abacus.stderr
grep -q '!FINAL_ETOT_IS' "$output_dir/running_scf.log"
grep -q 'rpa_lcao_exx(Ha):' abacus.stdout
test -s band_out
test -s basis_wfc_out
test -s basis_aux_out
test -s v1_coulomb_full_iq_1_rank0.dat
test -s v1_Cs_data_0.txt
test -s KS_eigenvector_0.dat

awk -v expected_bands="$nbands" -v expected_spins="$nspin" \
    -v expected_electrons="$nelec" '
  NR == 2 && $1 != expected_spins { exit 2 }
  NR == 3 && $1 != expected_bands { exit 3 }
  NR >= 7 && NF >= 4 {
    rows += 1
    occupation = $2 + 0.0
    occupation_sum += occupation
    if (occupation < -1.0e-8 || occupation > 2.0 + 1.0e-8) { exit 4 }
    nearest = int(occupation + 0.5)
    difference = occupation - nearest
    if (difference < 0) { difference = -difference }
    if (difference > 1.0e-8) { exit 5 }
  }
  END {
    if (rows != expected_bands * expected_spins) { exit 6 }
    difference = occupation_sum - expected_electrons
    if (difference < 0) { difference = -difference }
    if (difference > 1.0e-8) { exit 7 }
  }
' band_out

/usr/bin/time -v -o librpa.time \
  mpirun -np 1 -ppn 1 "$librpa" > librpa.stdout 2> librpa.stderr
grep -q 'libRPA finished successfully' librpa.stdout
grep -q 'Total EcRPA:' librpa.stdout
sha256sum band_out basis_wfc_out basis_aux_out v1_coulomb_full_iq_1_rank0.dat \
  v1_Cs_data_0.txt KS_eigenvector_0.dat > PRODUCTION_OUTPUTS.sha256
echo "date_end=$(date +%F_%T)" >> RUN_PROVENANCE.txt

