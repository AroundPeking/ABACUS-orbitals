#!/bin/bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
script=$root/run_named_candidate_band_window_scan_d4810f73.slurm

test -s "$script"
grep -Fq '#SBATCH --partition=p1' "$script"
grep -Fq '#SBATCH --ntasks=1' "$script"
grep -Fq '#SBATCH --cpus-per-task=40' "$script"
grep -Fq '10:20,14:28,18:36,22:44,26:52,30:60,34:68' "$script"
grep -Fq 'run_named_candidate_band_window_scan.py' "$script"
grep -Fq 'n_bands_chi0' "$root/run_named_candidate_band_window_scan.py"
grep -Fq 'minimax_min_gap' "$root/run_named_candidate_band_window_scan.py"
grep -Fq 'minimax_max_transition' "$root/run_named_candidate_band_window_scan.py"
