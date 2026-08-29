# C Atom-Solid Balanced Response Implementation Plan

**Goal:** Optimize one compact C basis against both the isolated C atom and
diamond-C response, then validate the ordinary all-band SOS binding energy
against the fixed Delta-ST reference.

## 1. Family-balanced loss

- Add tests showing that two periodic datasets in one family retain the legacy
  weighted loss, while atom and solid families contribute equal normalized
  weight regardless of matrix dimension.
- Add family labels and per-family diagnostics to the periodic optimizer.
- Preserve the legacy single-family command line and output behavior.

## 2. Atomic projected-Pi adapter

- Read the accepted C `sternheimer_matrix.dat` and a matching source-v1 file.
- Reuse `ProjectedPiEvaluator` so the atomic contribution is a normalized
  response-matrix loss, not an unscaled sum of first-order-wavefunction rows.
- Permit a different atomic frequency grid from the periodic six-point grid.
- Report atomic and solid losses independently in every checkpoint and final
  result.

## 3. Source-only atom gate

- Reuse the accepted 20-A, FD8, integer-occupation atom producer and its exact
  restart files.
- Run `sternheimer_siab_source_only=1` with the same Bessel primitives, ABFS,
  Coulomb whitening threshold, and 16-frequency metadata.
- Require response/source primitive blocks, overlap, occupations, auxiliary
  channels, frequencies, PP, orbital, and Coulomb provenance to match before
  pairing.  Do not solve or submit a new Delta-ST response.

## 4. Low-cost joint campaign

- Train with `C_atom` plus solid `q1` and `q2`, freezing `2s2p1d`.
- Start from a compact nested basis that already passes overlap and virtual-tail
  gates; do not reuse the rejected 3g/2g/1g results as accepted checkpoints.
- Evaluate solid `q3`, atomic family loss, occupied capture, overlap condition,
  radial smoothness, and virtual spectrum before promotion.

## 5. Final SOS validation

- Export one selected basis.
- Recalculate the integer-occupation C atom and the two-atom diamond primitive
  cell with ordinary all-band SOS.
- Run the full 64-q LibRPA chain once and compute zero-order, correlation, and
  total binding contributions separately.
- Accept only below `0.1 kcal/mol/C`; otherwise reject the basis and diagnose a
  single low-cost variable before proposing another full-BZ run.

