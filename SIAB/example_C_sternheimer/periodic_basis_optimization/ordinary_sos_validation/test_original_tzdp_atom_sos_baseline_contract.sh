#!/bin/bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
script=$root/run_original_tzdp_atom_sos_baseline_55d25e3c9_d4810f73.slurm

test -s "$script"
grep -Fq '#SBATCH --partition=p1' "$script"
grep -Fq '#SBATCH --ntasks=1' "$script"
grep -Fq '#SBATCH --cpus-per-task=40' "$script"
grep -Fq 'C_gga_10au_100Ry_3s3p2d.orb' "$script"
grep -Fq 'expected_orbital_sha=7ba114ee382d50ed831a0c90919ce291f97a08075e0e18851977d3217597289d' "$script"
grep -Fq 'set_input_key nbands 22' "$script"
grep -Fq 'set_input_key rpa_pca_fixed_nu 2,2,1,0,0' "$script"
grep -Fq 'set_input_key exx_pca_threshold 1e-4' "$script"
grep -Fq 'set_input_key sternheimer_fd_order 8' "$script"
grep -Fq 'set_input_key nx 135' "$script"
grep -Fq 'n_bands_chi0 = -1' "$script"
grep -Fq 'prefix_coul_full = v1_coulomb_full_iq_' "$script"
grep -Fq 'use_rpa_gamma = true' "$script"
grep -Fq 'libRPA finished successfully' "$script"
grep -Fq 'status success' "$script"
