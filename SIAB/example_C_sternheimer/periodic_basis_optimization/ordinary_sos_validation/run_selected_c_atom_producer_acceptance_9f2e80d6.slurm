#!/bin/bash
#SBATCH --job-name=c_atom_pca_accept
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-pca-accept-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-atom-pca-accept-%j.err

set -euo pipefail
trap 'status=$?; echo "C_SELECTED_ATOM_ACCEPTANCE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
base=$campaign/runs/product-pca-20260825
atom_root=$base/selected-c-atom-response-aware-product-pca-55d25e3c9
source_job_id=3144040
source_deployment=/data/home/df_iopcas_ghj/app/siab/periodic-c-9f2e80d6
source_script=$source_deployment/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_selected_c_atom_response_pca_producer_55d25e3c9.slurm
source_log=$campaign/logs/c-atom-pca-final-${source_job_id}.err
source_producer=$atom_root/producer-final
accepted=$atom_root/producer-accepted
temporary=$atom_root/.producer-accepted.${SLURM_JOB_ID:?}
expected_source_script_sha=6f6531963bf03cc60b0d93fe8431bbefefb11941e7e25fe24f7458f7b2704a17
expected_source_log_sha=f58c37c5588b410ac0bec4f47b18e3eb719963c553fb8e58f3b6b72c77c36d74
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
expected_orbital_sha=3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1
expected_naux=261

test "${SLURM_JOB_PARTITION:?}" = p1
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 1
test ! -e "$accepted" && test ! -e "$temporary"
test -d "$source_producer" && test -s "$source_script" && test -s "$source_log"
test "$(sha256sum "$source_script" | awk '{print $1}')" = "$expected_source_script_sha"
test "$(sha256sum "$source_log" | awk '{print $1}')" = "$expected_source_log_sha"
grep -Fqx 'C_SELECTED_ATOM_PRODUCER_FAILED line=245 command=test -s "$name" status=1' "$source_log"

read -r source_state source_exit < <(
  sacct -n -X -j "$source_job_id" --format=State,ExitCode -P \
    | awk -F'|' 'NF>=2 && $1!="" {print $1, $2; exit}'
)
test "$source_state" = FAILED
test "$source_exit" = 1:0
grep -Fq 'Exit status: 0' "$source_producer/abacus.time"
grep -Fq 'FINISH Time' "$source_producer/abacus.out"
grep -Fq 'TOTAL  Time  : 377' "$source_producer/abacus.out"
grep -Fq 'Response-aware RPA product PCA fixed radial profiles: 2,2,1,0,0' "$source_producer/abacus.out"
grep -Fq 'rpa_pca_threshold: 0.0001' "$source_producer/abacus.out"
grep -Fqx 'status=running' "$source_producer/provenance.txt"
grep -Fqx "abacus_sha256=$expected_abacus_sha" "$source_producer/provenance.txt"
grep -Fqx "selected_orbital_sha256=$expected_orbital_sha" "$source_producer/provenance.txt"
grep -Fqx "expected_naux=$expected_naux" "$source_producer/provenance.txt"
grep -Fqx "$expected_source_script_sha  /var/spool/slurmd/job${source_job_id}/slurm_script" "$source_producer/provenance.txt"

mkdir "$temporary"
python3 - "$source_producer" "$expected_naux" "$expected_orbital_sha" "$temporary/READER_HEADER.json" <<'PY'
import json
import math
import pathlib
import re
import struct
import sys

source = pathlib.Path(sys.argv[1])
expected_naux = int(sys.argv[2])
expected_orbital_sha = sys.argv[3]
output = pathlib.Path(sys.argv[4])

grid = json.loads((source / "GRID_MATRIX_HEADER.json").read_text(encoding="ascii"))
assert grid == {
    "atom_auxiliary_dimensions": [expected_naux],
    "iq": 1,
    "marker": -20129433,
    "natoms": 1,
    "naux": expected_naux,
    "nblocks": 1,
    "status": "success",
    "value_flag": 1,
}

with (source / "KS_eigenvector_0.dat").open("rb") as handle:
    header = struct.unpack("<8i", handle.read(32))
assert header == (-12345679, 28, 1, 2, 56, 56, 1, 36)
assert not (source / "KS_eigenvector_1.dat").exists()

band_header = [int(value) for value in (source / "band_out").read_text(
    encoding="ascii"
).splitlines()[:4]]
assert band_header == [1, 2, 56, 56]

states = {1: [], 2: []}
spin = None
eig_occ = source / "OUT.C_ATOM_SELECTED_RESPONSE_PCA/eig_occ.txt"
for line in eig_occ.read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"spin=(\d+)", line)
    if match:
        spin = int(match.group(1))
        continue
    fields = line.split()
    if spin in states and len(fields) == 3 and fields[0].isdigit():
        states[spin].append(float(fields[2]))
assert all(len(values) == 56 for values in states.values())
assert states[1][:3] == [1.0, 1.0, 1.0]
assert all(abs(value) <= 1.0e-12 for value in states[1][3:])
assert states[2][0] == 1.0
assert all(abs(value) <= 1.0e-12 for value in states[2][1:])
assert math.isclose(sum(states[1]), 3.0)
assert math.isclose(sum(states[2]), 1.0)

atom_reference = dict(
    line.split(maxsplit=1)
    for line in (source / "ATOM_REFERENCE.txt").read_text(encoding="ascii").splitlines()
)
assert atom_reference["status"] == "success"
assert atom_reference["selected_orbital_sha256"] == expected_orbital_sha
assert int(atom_reference["naux"]) == expected_naux
assert math.isfinite(float(atom_reference["reference_ha"]))

payload = {
    "status": "success",
    "reader_file_count": 1,
    "nk": header[2],
    "nspin": header[3],
    "nbands_per_spin": header[4],
    "nlocal": header[5],
    "naux": expected_naux,
    "occupations": {"spin_1": 3.0, "spin_2": 1.0},
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

required=(
  INPUT STRU KPT C_ONCV_PBE-1.0.upf C_gga_10au_100Ry_selected_product_pca.orb
  ATOM_REFERENCE.txt GRID_MATRIX_HEADER.json abacus.out abacus.time band_out
  basis_aux_out basis_out basis_wfc_out bz_sampling_out KS_eigenvector_0.dat
  stru_out v1_Cs_data_0.txt v1_coulomb_cut_iq_1_rank0.dat
  v1_coulomb_full_iq_1_rank0.dat
)
for name in "${required[@]}"; do
  test -s "$source_producer/$name"
  ln -s "$source_producer/$name" "$temporary/$name"
done

reference_ha=$(awk '$1=="reference_ha" {print $2}' "$source_producer/ATOM_REFERENCE.txt")
wall_clock=$(sed -n 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss):[[:space:]]*//p' "$source_producer/abacus.time")
cat > "$temporary/PRODUCER_ACCEPTANCE.txt" <<EOF
status success
acceptance_kind completed_physics_wrapper_only_failure
source_job_id $source_job_id
source_scheduler_state $source_state
source_scheduler_exit_code $source_exit
source_abacus_exit_status 0
reader_file_count 1
reader_nk 1
reader_nspin 2
nbands_per_spin 56
naux $expected_naux
selected_orbital_sha256 $expected_orbital_sha
reference_ha $reference_ha
wall_clock $wall_clock
wrapper_failure reader_v1_single_kpoint_file_contains_two_spins
EOF
find -L "$temporary" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$temporary/INPUT_SHA256.txt"
{
  echo status=success
  echo source_job_id=$source_job_id
  echo source_scheduler_state=$source_state
  echo source_scheduler_exit_code=$source_exit
  echo source_abacus_exit_status=0
  echo acceptance_kind=completed_physics_wrapper_only_failure
  echo selected_orbital_sha256=$expected_orbital_sha
  echo naux=$expected_naux
  echo reference_ha=$reference_ha
  echo wall_clock=$wall_clock
  sha256sum "$0" "$source_script" "$source_log" "$temporary/READER_HEADER.json" "$temporary/PRODUCER_ACCEPTANCE.txt" "$temporary/INPUT_SHA256.txt"
} > "$temporary/provenance.txt"

mv "$temporary" "$accepted"
echo "C_SELECTED_ATOM_ACCEPTANCE_OK source_job=$source_job_id accepted=$accepted reference_ha=$reference_ha"
