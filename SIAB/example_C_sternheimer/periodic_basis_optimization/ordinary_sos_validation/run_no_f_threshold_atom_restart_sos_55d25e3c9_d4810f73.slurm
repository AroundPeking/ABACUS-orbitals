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
run_root=$campaign/runs/product-pca-20260825/joint-atom-solid-no-f-threshold-q12-control
restart_source=$run_root/atom/OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA
base=$run_root/patched-atom.sh
patched=$run_root/patched-atom-restart.sh

test -s "$base"
for name in wfs1_nao.txt wfs2_nao.txt chgs1.cube chgs2.cube; do
  test -s "$restart_source/$name"
done
test ! -e "$patched"
test ! -e "$run_root/atom-restart"

sed -E \
  -e 's|work=\$run_root/atom|work=$run_root/atom-restart|' \
  -e '/^cd "\$work"$/a\
cp '"$restart_source"'/wfs1_nao.txt '"$restart_source"'/wfs2_nao.txt '"$restart_source"'/chgs1.cube '"$restart_source"'/chgs2.cube "$work"/' \
  -e 's/set_input_key init_wfc atomic/set_input_key init_wfc file/' \
  -e 's/set_input_key init_chg atomic/set_input_key init_chg file\
set_input_key mixing_beta 0.1\
set_input_key mixing_beta_mag 0.1\
set_input_key mixing_gg0 0.0\
set_input_key mixing_gg0_mag 0.0/' \
  -e 's/purpose=threshold_only_product_pca_c_atom_all_band_sos/purpose=no_f_threshold_c_atom_restart_all_band_sos/' \
  "$base" > "$patched"

grep -q 'work=\$run_root/atom-restart' "$patched"
grep -q 'set_input_key init_wfc file' "$patched"
grep -q 'set_input_key init_chg file' "$patched"
grep -q 'set_input_key mixing_beta 0.1' "$patched"
grep -q 'set_input_key mixing_gg0_mag 0.0' "$patched"
grep -q 'purpose=no_f_threshold_c_atom_restart_all_band_sos' "$patched"

bash "$patched"

work=$run_root/atom-restart
grep -qx 'success' "$work/STATUS"
grep -q '#SCF IS CONVERGED#' "$work/OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA/running_scf.log"
! grep -q '!!SCF IS NOT CONVERGED!!' "$work/OUT.C_ATOM_THRESHOLD_CANDIDATE_PCA/running_scf.log"
{
  echo restart_source_job=3158450
  echo restart_source_scf_converged=no
  echo restart_physical_state=fixed_triplet_nup3_ndown1
  sha256sum "$0" "$base" "$patched" "$restart_source/wfs1_nao.txt" "$restart_source/wfs2_nao.txt" "$restart_source/chgs1.cube" "$restart_source/chgs2.cube"
} >> "$work/provenance.txt"

echo "C_NO_F_ATOM_RESTART_OK work=$work"
