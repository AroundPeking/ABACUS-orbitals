#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
producer=$root/run_selected_c_atom_response_pca_producer_55d25e3c9.slurm
sos=$root/run_selected_c_atom_sos_d4810f73.slurm
delta=$root/run_selected_c_atom_matched_delta_55d25e3c9.slurm
reader=$root/run_selected_c_atom_matched_delta_reader_d4810f73.slurm
release=$root/release_selected_c_atom_binding_chain.sh
collector=$root/collect_selected_c_binding_energy.py

for file in "$producer" "$sos" "$delta" "$reader" "$release" "$collector"; do
  test -s "$file"
done

grep -q '^#SBATCH --partition=p1$' "$producer"
grep -q '^#SBATCH --nodes=1$' "$producer"
grep -q '^#SBATCH --ntasks=1$' "$producer"
grep -q '^#SBATCH --cpus-per-task=40$' "$producer"
grep -q '^#SBATCH --mem=190000$' "$producer"
grep -q '^#SBATCH --time=1-00:00:00$' "$producer"
grep -q '^expected_orbital_sha=3e7e31072a0a388b12397f9957e75502d4c75755534cb2db1c3cfe12e8f132b1$' "$producer"
grep -q '^expected_naux=261$' "$producer"
grep -q '^fixed_nu=2,2,1,0,0$' "$producer"
grep -q 'set_input_key nbands 56' "$producer"
grep -q 'set_input_key nspin 2' "$producer"
grep -q 'set_input_key nupdown 2' "$producer"
grep -q 'set_input_key ocp_set "3\*1 53\*0 1\*1 55\*0"' "$producer"
grep -q 'set_input_key nx 135' "$producer"
grep -q 'set_input_key ny 135' "$producer"
grep -q 'set_input_key nz 135' "$producer"
grep -q 'set_input_key sternheimer_fd_order 8' "$producer"
grep -q 'set_input_key exx_pca_threshold 1e-4' "$producer"
grep -q 'set_input_key rpa_pca_fixed_nu "$fixed_nu"' "$producer"
grep -q 'set_input_key out_sternheimer_basis_opt 1' "$producer"
grep -q 'remove_stru_abfs_section' "$producer"
grep -q 'full_periodic_poisson' "$producer"
grep -q 'assert header\["natoms"\] == 1' "$producer"
grep -q 'assert header\["naux"\] == expected_naux' "$producer"
! grep -q 'C_10au_3s3p2d1f1g_1e-4.abfs' "$producer"

grep -q '^#SBATCH --nodes=1$' "$sos"
grep -q '^librpa_commit=d4810f73aab20c36e69b1c353c945b77f40931c9$' "$sos"
grep -q '^expected_naux=261$' "$sos"
grep -q 'task = rpa' "$sos"
grep -q 'nfreq = 6' "$sos"
grep -q 'tfgrids_type = minimax' "$sos"
grep -q 'use_rpa_gamma = true' "$sos"
grep -q 'replace_w_head = false' "$sos"
grep -q 'prefix_coul_full = v1_coulomb_grid_iq_' "$sos"
grep -q 'extract_librpa_frequency_grid.py' "$sos"
grep -q 'ATOM_SOS_FREQUENCY_GRID.dat' "$sos"
grep -q 'side atom' "$sos"
grep -q 'method sos' "$sos"
grep -q 'scope body_only_no_analytic_headwing' "$sos"

grep -q '^#SBATCH --partition=p1$' "$delta"
grep -q '^#SBATCH --nodes=16$' "$delta"
grep -q '^#SBATCH --ntasks=16$' "$delta"
grep -q '^#SBATCH --cpus-per-task=40$' "$delta"
grep -q '^#SBATCH --mem=190000$' "$delta"
grep -q '^expected_naux=261$' "$delta"
grep -q 'set_input_key nbands 56' "$delta"
grep -q 'set_input_key ocp_set "3\*1 53\*0 1\*1 55\*0"' "$delta"
grep -q 'set_input_key sternheimer_delta 1' "$delta"
grep -q 'set_input_key sternheimer_fd_order 8' "$delta"
grep -q 'set_input_key sternheimer_frequency_mpi 1' "$delta"
grep -q 'set_input_key sternheimer_channel_mpi 1' "$delta"
grep -q 'set_input_key sternheimer_mpi_layout global_equation' "$delta"
grep -q 'set_input_key sternheimer_frequency_grid_file "$frequency_name"' "$delta"
grep -q 'ATOM_SOS_FREQUENCY_GRID.dat' "$delta"
grep -q 'all_converged yes' "$delta"
grep -q 'max_relative_residual' "$delta"
grep -q 'cmp -s basis_aux_out' "$delta"
! grep -q 'sternheimer_solver_tolerance' "$delta"

grep -q '^#SBATCH --nodes=1$' "$reader"
grep -q '^expected_naux=261$' "$reader"
grep -q 'task = sternheimer_rpa' "$reader"
grep -q 'nfreq = 6' "$reader"
grep -q 'use_rpa_gamma = true' "$reader"
grep -q 'replace_w_head = false' "$reader"
grep -q 'side atom' "$reader"
grep -q 'method delta_st' "$reader"
grep -q 'frequency_grid_source atom_sos_exact' "$reader"

grep -q 'flock' "$release"
grep -q 'sbatch --test-only' "$release"
grep -q 'squeue' "$release"
grep -q 'sacct' "$release"
grep -q 'status success' "$release"
grep -q 'refusing duplicate' "$release"
grep -q 'collect_selected_c_binding_energy.py' "$release"

echo "C_SELECTED_ATOM_BINDING_CONTRACT_OK"
