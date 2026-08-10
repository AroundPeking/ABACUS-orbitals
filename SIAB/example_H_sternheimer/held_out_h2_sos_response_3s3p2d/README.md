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
`run_campaign_66.sh CAMPAIGN_ROOT 1 6` runs the six independent physical cases
concurrently, using one MPI rank per ABACUS case.

Each MPI rank is forced to one OpenMP thread.  This is a correctness condition
for the validated 20 Angstrom, 100 Ry molecular grid: using more than one
OpenMP thread produced a nonphysical negative integrated density during the
first SCF density construction, while one-rank and eight-rank calculations at
one thread per rank give the same converged zero-order energy.  Eight MPI ranks
accelerate the SCF iterations but take longer than the complete 923-second
one-rank producer because the auxiliary-basis/Coulomb postprocessing does not
scale in this molecular path.  Concurrent one-rank cases therefore give the
shortest measured campaign throughput and remain well below the memory limit.
