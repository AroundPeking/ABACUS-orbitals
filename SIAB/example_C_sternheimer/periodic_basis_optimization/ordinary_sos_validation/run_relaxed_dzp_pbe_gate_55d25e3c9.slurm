#!/bin/bash
#SBATCH --job-name=c_relaxed_pbe
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-relaxed-pbe-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-relaxed-pbe-%j.err

set -euo pipefail
trap 'status=$?; echo "C_RELAXED_DZP_PBE_GATE_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR
module load oneapi/2024.2

candidate_root=${CANDIDATE_ROOT:?missing immutable candidate root}
run_root=${RUN_ROOT:?missing immutable run root}
source_root=${SIAB_SOURCE_ROOT:?missing exact SIAB source deployment}
atom_restart_source=${ATOM_RESTART_SOURCE:-}
manifest=$candidate_root/CANDIDATE.json
collector=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/collect_relaxed_dzp_pbe_gate.py
manifest_reader=$source_root/SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/read_periodic_candidate_manifest.py
atom_source=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/runs/product-pca-20260825/original-tzdp-atom-allband-sos-recovery-8e5cfb5d
solid_source=/data/home/df_iopcas_ghj/app/abacus/unified-basis-opt-aux-source-711af860c-20260825/runs/c-periodic-basis-opt-product-pca-20260825/retry1/c-diamond-g24-fd8-nfreq6-q1-product-pca-711af860c-kpar8-retry1
abacus=/data/home/df_iopcas_ghj/app/abacus/response-pca-55d25e3c9-20260827/artifacts/job3137534/abacus_3p
expected_abacus_sha=25a9133c3facfe44d64d34f2696fe899b82e55be6e22c15c20ade0c00721a3d9
python=/data/home/df_iopcas_ghj/app/miniconda3/bin/python
work=$run_root/pbe-gate

case "${SLURM_JOB_PARTITION:?}" in
  p1|48cp2) ;;
  *) echo "unsupported df partition: $SLURM_JOB_PARTITION" >&2; exit 1 ;;
esac
test "${SLURM_NTASKS:?}" = 1
test "${SLURM_CPUS_PER_TASK:?}" = 40
test ! -e "$work"
test -s "$manifest" && test -s "$collector" && test -s "$manifest_reader"
test -x "$abacus" && test "$(sha256sum "$abacus" | awk '{print $1}')" = "$expected_abacus_sha"
mapfile -t candidate < <("$python" "$manifest_reader" "$candidate_root")
test "${candidate[0]}" = "$manifest"
orbital_name=${candidate[1]}
orbital_sha=${candidate[2]}
nao_atom=${candidate[3]}
orbital_layout=${candidate[4]}
orbital=$candidate_root/$orbital_name
if test -n "$atom_restart_source"; then
  test -d "$atom_restart_source"
  for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do
    test -s "$atom_restart_source/$name"
  done
fi

export LD_LIBRARY_PATH=/data/home/df_iopcas_ghj/app/elpa-build/lib:/data/app/libxc/6.2.2-intel/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=PASSIVE
export OMP_STACKSIZE=2G
export MKL_DYNAMIC=FALSE
export I_MPI_FABRICS=shm:ofi

set_input_key() {
  local input=$1 key=$2 value=$3 temporary
  if awk -v key="$key" '$1 == key {found=1} END {exit !found}' "$input"; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]+.*|${key} ${value}|" "$input"
  else
    printf '%-36s %s\n' "$key" "$value" >> "$input"
  fi
}

remove_stru_abfs_section() {
  "$python" - "$1" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sections = {"ATOMIC_SPECIES", "NUMERICAL_ORBITAL", "ABFS_ORBITAL", "LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS"}
output, skip = [], False
for line in path.read_text(encoding="ascii").splitlines():
    stripped = line.strip()
    if stripped == "ABFS_ORBITAL":
        skip = True
        continue
    if skip and stripped in sections:
        skip = False
    if not skip:
        output.append(line)
path.write_text("\n".join(output).rstrip() + "\n", encoding="ascii")
PY
}

replace_orbital() {
  "$python" - "$1" "$orbital_name" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
name = sys.argv[2]
text, count = re.subn(r"^C_gga_.*\.orb$", name, path.read_text(encoding="ascii"), count=1, flags=re.MULTILINE)
if count != 1:
    raise SystemExit("STRU must contain exactly one C numerical orbital line")
path.write_text(text, encoding="ascii")
PY
}

prepare_side() {
  local side=$1 source=$2 suffix=$3 nbands=$4 nx=$5 ny=$6 nz=$7
  local side_work=$work/$side input=$work/$side/INPUT stru=$work/$side/STRU
  local out=$side_work/OUT.$suffix
  mkdir -p "$side_work"
  cp "$source/INPUT" "$source/STRU" "$source/KPT" "$source/C_ONCV_PBE-1.0.upf" "$orbital" "$side_work/"
  remove_stru_abfs_section "$stru"
  replace_orbital "$stru"
  set_input_key "$input" suffix "$suffix"
  set_input_key "$input" calculation scf
  set_input_key "$input" nbands "$nbands"
  set_input_key "$input" scf_thr 1e-10
  set_input_key "$input" scf_nmax 300
  set_input_key "$input" rpa 0
  set_input_key "$input" out_librpa_reader_version 0
  set_input_key "$input" out_sternheimer_librpa 0
  set_input_key "$input" out_sternheimer_basis_opt 0
  set_input_key "$input" nx "$nx"
  set_input_key "$input" ny "$ny"
  set_input_key "$input" nz "$nz"
  if test "$side" = atom; then
    set_input_key "$input" nelec 4
    set_input_key "$input" nspin 2
    set_input_key "$input" nupdown 2
    set_input_key "$input" symmetry 0
    set_input_key "$input" gamma_only 1
    set_input_key "$input" kpar 1
    set_input_key "$input" smearing_method fixed
    set_input_key "$input" ocp 1
    set_input_key "$input" ocp_set "3*1 $((nbands - 3))*0 1*1 $((nbands - 1))*0"
    set_input_key "$input" out_chg 1
    set_input_key "$input" out_wfc_lcao 1
    set_input_key "$input" out_app_flag 1
    if test -n "$atom_restart_source"; then
      mkdir -p "$out" "$side_work/restart_input_snapshot"
      for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do
        cp "$atom_restart_source/$name" "$out/$name"
        cp "$atom_restart_source/$name" "$side_work/restart_input_snapshot/$name"
      done
      set_input_key "$input" init_wfc file
      set_input_key "$input" init_chg file
      set_input_key "$input" mixing_beta 0.2
      set_input_key "$input" mixing_beta_mag 0.2
      set_input_key "$input" mixing_gg0 0.0
      set_input_key "$input" mixing_gg0_mag 0.0
    else
      set_input_key "$input" init_wfc atomic
      set_input_key "$input" init_chg atomic
    fi
  else
    set_input_key "$input" nspin 1
    set_input_key "$input" symmetry -1
    set_input_key "$input" kpar 1
    set_input_key "$input" init_wfc atomic
    set_input_key "$input" init_chg atomic
  fi
  (
    cd "$side_work"
    /usr/bin/time -v -o abacus.time mpirun -np 1 "$abacus" > abacus.out 2>&1
  )
  test -s "$out/running_scf.log"
  grep -q '#SCF IS CONVERGED#' "$out/running_scf.log"
  ! grep -q '!!SCF IS NOT CONVERGED!!' "$out/running_scf.log"
  if test "$side" = atom; then
    "$python" - "$out/eig_occ.txt" "$nbands" <<'PY'
import math
import pathlib
import re
import sys

states = {1: [], 2: []}
spin = None
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"spin=(\d+)", line)
    if match:
        spin = int(match.group(1))
        continue
    fields = line.split()
    if spin in states and len(fields) == 3 and fields[0].isdigit():
        states[spin].append(float(fields[2]))
nbands = int(sys.argv[2])
valid = (
    all(len(values) == nbands for values in states.values())
    and states[1][:3] == [1.0, 1.0, 1.0]
    and all(abs(value) <= 1.0e-12 for value in states[1][3:])
    and states[2][0] == 1.0
    and all(abs(value) <= 1.0e-12 for value in states[2][1:])
    and math.isclose(sum(states[1]), 3.0)
    and math.isclose(sum(states[2]), 1.0)
)
if not valid:
    raise SystemExit("atom occupations are not the intended 3/1 state")
PY
    if test -n "$atom_restart_source"; then
      for spin in 1 2; do
        grep -Eq "Read NAO wave functions from .*wfs${spin}_nao.txt" "$side_work/abacus.out"
        grep -Eq "Read in electron density: .*chgs${spin}\.cube|Read electron density from file: .*chgs${spin}\.cube" "$out/running_scf.log"
      done
    fi
  fi
  "$python" - "$out/running_scf.log" "$side_work/PBE_ENERGY_EV" <<'PY'
import math
import pathlib
import re
import sys

values = []
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)\s+eV", line)
    if match:
        values.append(float(match.group(1)))
if len(values) != 1 or not math.isfinite(values[0]):
    raise SystemExit(f"expected one finite !FINAL_ETOT_IS value, got {values}")
pathlib.Path(sys.argv[2]).write_text(f"{values[0]:.16g}\n", encoding="ascii")
PY
}

mkdir -p "$work"
{
  echo status=running
  echo purpose=candidate_atom_solid_pbe_reference_gate
  echo reference_basis=original_unoptimized_sg15_tzdp
  echo orbital_layout=$orbital_layout
  echo nao_atom=$nao_atom
  echo selected_orbital_sha256=$orbital_sha
  echo atom_restart_source=${atom_restart_source:-none}
  echo atom_restart_mixing_beta=$(test -n "$atom_restart_source" && echo 0.2 || echo original)
  echo tolerance_ev=0.010
  sha256sum "$0" "$manifest" "$collector" "$manifest_reader" "$abacus" "$orbital"
  if test -n "$atom_restart_source"; then
    sha256sum "$atom_restart_source"/wfs1_nao.txt "$atom_restart_source"/wfs2_nao.txt \
      "$atom_restart_source"/chgs1.cube "$atom_restart_source"/chgs2.cube
  fi
} > "$work/provenance.txt"

prepare_side atom "$atom_source" C_CANDIDATE_PBE_ATOM "$nao_atom" 135 135 135
prepare_side solid "$solid_source" C_CANDIDATE_PBE_SOLID "$((2 * nao_atom))" 24 24 24
atom_energy=$(cat "$work/atom/PBE_ENERGY_EV")
solid_energy=$(cat "$work/solid/PBE_ENERGY_EV")
"$python" "$collector" \
  --atom-energy-ev "$atom_energy" \
  --solid-c2-energy-ev "$solid_energy" \
  --tolerance-ev 0.010 \
  --output "$work/PBE_GATE.json" > "$work/collect.out"
grep -q '"pbe_gate": "pass"' "$work/PBE_GATE.json"
{
  echo status=success
  echo atom_pbe_ev=$atom_energy
  echo solid_c2_pbe_ev=$solid_energy
  sha256sum "$work/PBE_GATE.json" "$work/atom/PBE_ENERGY_EV" "$work/solid/PBE_ENERGY_EV"
} >> "$work/provenance.txt"
printf 'success\n' > "$work/STATUS"
echo "C_RELAXED_DZP_PBE_GATE_OK work=$work"
