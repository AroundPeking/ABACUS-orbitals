# Held-out H2 SOS gate for the 3s3p2d response basis

This campaign compares the unmodified SG15 TZDP H basis with the atomic
Delta-Sternheimer response-optimized 3s3p2d basis.  It is an independent
physical test: the optimization target is not reused in the SOS energy.

Both lanes use the same 20 Angstrom cell, 0.74085 Angstrom H-H distance,
100 Ry wave-function cutoff, 16 minimax frequencies, full Coulomb matrix,
and the explicit `H_sg15_3s2p1d1f1g_gaus_pca1e-4.abfs` auxiliary basis.
Because the ABFS is explicit, ABACUS uses `exx_pca_threshold 10`; this screens
out an additional generated auxiliary basis and does not change the ABFS
provenance threshold of `1e-4`.

The primary observable is the uncorrected binding energy
`D_raw = 2 E(H) - E(H2)`.  The ghost calculation provides only the diagnostic
`D_CP = 2 E(H+ghost) - E(H2)` and `BSSE = D_raw - D_CP`.

`prepare_campaign.py` creates six immutable full-band cases.  On server 66,
`run_campaign_66.sh CAMPAIGN_ROOT 10` runs all six concurrently using one MPI
rank and ten OpenMP threads per case, occupying 60 of the 64 logical CPUs.
