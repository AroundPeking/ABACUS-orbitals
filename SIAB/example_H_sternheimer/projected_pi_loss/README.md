# Source-aware projected-Pi feasibility gate

This directory contains a read-only analysis. It pairs ABACUS source-v1 and
response-v1 files, reconstructs the full-Coulomb symmetrized projected
response, and ranks three fixed-size H TZDP coefficient sets. It does not
modify the optimizer and does not use SOS-RPA energy or H+ghost data.

Required physical inputs are one paired response/source and one validated
zero-order audit for H and H2, plus the initial TZDP, fixed-DZP joint, and
low-frequency-guarded `ORBITAL_RESULTS.txt` files. The coefficient interface is
exactly H `3s2p` with 25 radial primitives; `1s,2s,1p` must agree with the
initial basis within `1e-12`.

The command writes JSON, Markdown, PNG, and PDF diagnostics atomically. Exit
status `0` means both optimized bases improve the initial total loss at all
three primitive-overlap thresholds and every family is stable within 1%.
Exit status `2` means the complete diagnostics were written but the method
must stop before optimizer integration and move to a Galerkin-Sternheimer
design. Any malformed input, failed audit, or pairing error exits `1` before
creating the output directory.

The validated `df_dcu` runtime is Python 3.10, PyTorch 2.1.0+cpu, and NumPy
1.26.4. Matplotlib 3.10.3 is installed separately under
`/work1/ghj/runtime/siab-projected-pi-mpl-20260801`; add that directory to
`PYTHONPATH` instead of modifying the fixed SIAB runtime. The complete SIAB
Python regression suite passed 276 tests with this environment.

```bash
export PYTHONPATH=/work1/ghj/runtime/siab-projected-pi-mpl-20260801
python analyze_projected_pi.py \
  --h-response H/sternheimer_matrix.dat \
  --h-source H/STERNHEIMER_SIAB_SOURCE_V1.dat \
  --h-audit H_zero_order_identity.json \
  --h2-response H2/sternheimer_matrix.dat \
  --h2-source H2/STERNHEIMER_SIAB_SOURCE_V1.dat \
  --h2-audit H2_zero_order_identity.json \
  --initial initial_tzdp_ORBITAL_RESULTS.txt \
  --joint fixed_dzp_joint_ORBITAL_RESULTS.txt \
  --guarded low_frequency_guarded_ORBITAL_RESULTS.txt \
  --output-dir projected_pi_result
```
