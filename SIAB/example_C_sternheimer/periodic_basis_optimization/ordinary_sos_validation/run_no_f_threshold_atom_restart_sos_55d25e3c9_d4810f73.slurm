#!/bin/bash
#SBATCH --job-name=c_nof_atom_r
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=190000
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-atom-restart-%j.out
#SBATCH --error=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c/logs/c-nof-atom-restart-%j.err

set -euo pipefail
trap 'status=$?; echo "C_NO_F_ATOM_RESTART_FAILED line=$LINENO command=$BASH_COMMAND status=$status" >&2; exit $status' ERR

campaign=/data/home/df_iopcas_ghj/app/siab/periodic-c-efc24c2c
source_root=/data/home/df_iopcas_ghj/app/siab/periodic-c-ce2801f3
candidate_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-candidate-570a6d79
run_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-threshold-q12-control
restart_source=$run_root/atom/OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA
base=$run_root/patched-atom.sh
patched=$run_root/patched-atom-restart-v4.sh

test -s "$base"
for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do
  test -s "$restart_source/$name"
done
test ! -e "$patched"
test ! -e "$run_root/atom-restart-v3"

sed -E \
  -e 's|work=\$run_root/atom|work=$run_root/atom-restart-v3|' \
  -e '/^cd "\$work"$/a\
mkdir -p OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA restart_input_snapshot\
for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do\
  cp '"$restart_source"'/"$name" OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA/"$name"\
  cp '"$restart_source"'/"$name" restart_input_snapshot/"$name"\
done' \
  -e 's/set_input_key init_wfc atomic/set_input_key init_wfc file/' \
  -e 's/set_input_key init_chg atomic/set_input_key init_chg file/' \
  -e 's/purpose=threshold_only_product_pca_c_atom_all_band_sos/purpose=no_f_threshold_c_atom_exact_broyden_restart_all_band_sos/' \
  "$base" > "$patched"

grep -q 'work=\$run_root/atom-restart-v3' "$patched"
grep -q 'OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA/"\$name"' "$patched"
grep -q 'set_input_key init_wfc file' "$patched"
grep -q 'set_input_key init_chg file' "$patched"
! grep -Eq '^[[:space:]]*set_input_key mixing_' "$patched"
grep -q 'purpose=no_f_threshold_c_atom_exact_broyden_restart_all_band_sos' "$patched"

env CANDIDATE_ROOT="$candidate_root" RUN_ROOT="$run_root" SIAB_SOURCE_ROOT="$source_root" bash "$patched"

work=$run_root/atom-restart-v3
grep -qx 'success' "$work/STATUS"
grep -Eq '^mixing_beta[[:space:]]+0.3$' "$work/INPUT"
grep -Eq '^mixing_beta_mag[[:space:]]+0.3$' "$work/INPUT"
grep -q '#SCF IS CONVERGED#' "$work/OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA/running_scf.log"
! grep -q '!!SCF IS NOT CONVERGED!!' "$work/OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA/running_scf.log"
for spin in 1 2; do
  grep -Eq "Read NAO wave functions from .*wfs${spin}_nao.txt" "$work/abacus.out"
  grep -Eq "Read in electron density: .*chgs${spin}\.cube|Read electron density from file: .*chgs${spin}\.cube" "$work/OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA/running_scf.log"
done
{
  echo restart_source_job=3158450
  echo restart_source_scf_converged=no
  echo restart_change=reset_broyden_history_only
  echo mixing_beta=0.3
  echo mixing_beta_mag=0.3
  echo restart_physical_state=fixed_triplet_nup3_ndown1
  sha256sum "$0" "$base" "$patched" "$restart_source/wfs1_nao.txt" "$restart_source/wfs2_nao.txt" "$restart_source/chgs1.cube" "$restart_source/chgs2.cube"
} >> "$work/provenance.txt"

echo "C_NO_F_ATOM_RESTART_OK work=$work"
