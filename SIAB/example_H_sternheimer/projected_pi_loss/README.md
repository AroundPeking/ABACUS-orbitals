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

## Physical ranking result

Commit `ecedcd80` was staged under
`/work1/ghj/sternheimer_abacus_tests/siab_projected_pi_feasibility_20260801`.
Job `21464513` completed on one `normal` node with 30 CPUs and 110610 MB. The
analysis took 74.30 s and used 1,905,692 KiB maximum resident memory according
to `/usr/bin/time -v`. Both strict pairs and zero-order audits passed without
warnings, fixed `1s,2s,1p` differences were exactly zero, and the decision was
`pass`.

At the nominal primitive-overlap rank threshold `1e-12`:

| basis | H loss | H2 loss | equal-family total | H/H2 rank | max condition | held-out D CP |
|---|---:|---:|---:|---:|---:|---:|
| initial TZDP | 0.1422168377 | 0.0912063047 | 0.2334231424 | 534/1032 | 3.073394e4 | 105.556881 |
| fixed-DZP joint `3s2p` | 0.1329357582 | 0.0816364120 | 0.2145721702 | 534/1032 | 5.562937e3 | 105.853882 |
| low-frequency-guarded `3s2p` | 0.1333449050 | 0.0821021186 | 0.2154470237 | 534/1032 | 5.397578e3 | 105.843252 |

The CP values are independent held-out SOS-RPA results in kcal/mol; they were
not read by the projected-Pi command. The projected-Pi order is joint,
guarded, initial from best to worst, matching the held-out CP order. The same
strict order holds at all 16 frequencies for both H and H2. Across rank
thresholds `1e-10`, `1e-11`, and `1e-12`, the maximum relative loss spread is
`9.6355e-7`, well below the 1% gate. Candidate and reference Hermitian errors
are zero at stored precision.

This result validates projected-Pi as a training-target candidate; it does not
create or validate a new orbital basis. The next implementation is specified
in `docs/superpowers/plans/2026-08-01-siab-pi-dpsi-joint.md`. Raw JSON,
Markdown, plots, scheduler records, and input/code hashes are preserved under
`results/`.
