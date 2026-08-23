# C atomic SIAB Sternheimer reference producer

This producer starts only after the C fixed-occupation and weak-field/free
routes pass both the zero-order PBE and full Delta-ST response gates.  It uses
the accepted, field-free fixed-occupation restart because the two routes have
already been shown to represent the same triplet state and the same response.

The physical contract follows the successful H workflow:

- 20-Angstrom centered atom, 30 Ry, explicit `135^3` grid and FD8;
- neutral triplet C with integer `3 up / 1 down` occupation;
- 16 fixed GreenX frequencies;
- 10-au spherical-Bessel primitives through `l=4` at 100 Ry;
- explicit C TZDP-derived ABFS generated with PCA threshold `1e-4`;
- `exx_pca_threshold=10` in ABACUS, so the explicit ABFS is not truncated a
  second time;
- SIAB Coulomb whitening threshold `1e-10`, matching the H target producer;
- solver tolerance `1e-6`.  Server 66's
  `640` partition contains ten nodes, so the global-equation scheduler assigns
  the 16 frequencies across ten ranks; some ranks process two frequencies.
  `sternheimer_mpi_layout=global_equation` is explicit: the default grouped
  layout would incorrectly require the rank count to be a multiple of 16.
  The partition permits an unlimited wall time, which is requested explicitly
  because the 15040 response equations may approach a one-day limit.

The df production lane uses the normal `p1` partition: 16 nodes, one MPI rank
and 40 OpenMP threads per node, 190000 MB per node and an exclusive allocation.
This gives one rank per GreenX frequency while `global_equation` remains the
authoritative work layout; the physical equation set is unchanged.

If the 30-minute pilot reaches `channel_workers_ready` on all 16 ranks but
finishes no complete low-frequency channel batch, do not submit a longer pilot
that repeats the full precompute.  After cancelling the dependency-blocked
formal job at zero elapsed time, `submit_siab_reference_df_recovery.sh` audits
the pilot progress and submits exactly one unlimited formal producer.  It
refuses a failed precompute, any Sternheimer failure file, a non-cancelled
original formal job, or an existing recovery receipt.

The df production wrapper is pinned to ABACUS commit `8cf890e8c`, which
restores the streamed full-Coulomb-whitened auxiliary perturbations used by
the SIAB Sternheimer equations.  The raw auxiliary potentials are sampled in
1024-point grid chunks and transformed as
`V_tilde(r) = V_raw(r) W`; they are not stored as a dense raw-channel by grid
array.

`prepare_siab_reference.py` refuses an unpassed or tampered response source and
creates a new immutable target directory.  Restart wavefunctions and densities
are read from the response manifest's original zero-field PBE phase, not from
the response output directory: the latter may be rewritten when Delta-ST runs.
The submitter first runs a one-node
ABFS/whitening diagnostic in an isolated copy, then launches the 10-node SIAB
producer only after that diagnostic passes.  A completed producer must contain
`sternheimer_matrix.dat`, a successful all-converged status, the ABFS channel
map, Coulomb-whitening diagnostics and a SHA256 completion manifest.

`generate_frequency_grid.py` creates the 16-point GreenX grid from the accepted
field-free fixed-occupation `eig_occ.txt`.  The field/free equivalence has
already been established by the six-frequency response gate, so the production
target does not mix two slightly different numerical transition windows.

The atomic target is a pipeline gate, not the final transferable C training
set.  Once it is accepted, optimization freezes the C DZP core
(`1s,2s,1p,2p,1d`) and first varies the remaining TZDP shells
(`3s,3p,2d`).  The formal basis training and validation must then add C-C
bonding environments and held-out diamond volumes before any solid-state RPA
claim is accepted.
