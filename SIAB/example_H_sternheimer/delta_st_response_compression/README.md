# H Delta-ST Response Compression

The gradient gate checks that the original H `3s2p` SIAB basis has a usable
descent direction toward an immutable uniform-grid Delta-ST response. The full
runner performs bounded optimization and can promote new radial shells from
the Bessel primitive parent. Neither path fits the scalar RPA correlation
energy.

The candidate response contains only the compact `3s2p` LCAO space. The
occupied state is obtained from the fixed ABACUS LCAO generalized eigenproblem.
The `1s`, `2s`, and `1p` radial columns are frozen; only `3s` and `2p` move.
Both candidate and reference responses use the same full-Coulomb transform.

Run on server66 with the `abacus-orbitals` Python environment:

```bash
python run_h_gradient_gate.py \
  /path/to/grid-reference \
  /path/to/sternheimer_galerkin_primitive.dat \
  /path/to/sternheimer_galerkin_fixed_ao.dat \
  /path/to/SG15/H_TZDP/info/8/ORBITAL_RESULTS.txt \
  /path/to/SG15/H_TZDP/H_gga_8au_100Ry_3s2p.orb \
  /path/to/new-output-directory \
  --reference-commit 142b090e2babbc0d1cf1831c165d19a03ef56526 \
  --sidecar-commit bc720617aa058ab14823b5104b6657dc549b2d7d \
  --siab-commit COMMIT
```

The explicit orbital path is a protocol gate: its SHA256 must equal the
orbital hash stored in the grid-reference and sidecar metadata. This prevents
coefficients from a different pseudopotential family from being optimized
against an unrelated fixed-AO reference.

The output directory contains the initial and accepted SIAB coefficient files
and `gradient_gate.json`. The JSON records protocol hashes, full-frequency
losses, frozen and variable gradient norms, the accepted step, response-space
rank and condition diagnostics, timing, memory, and RPA-energy diagnostics.

After this gate passes, run the bounded full optimization with the same six
positional inputs and provenance commits:

```bash
python run_h_response_optimization.py \
  /path/to/grid-reference \
  /path/to/sternheimer_galerkin_primitive.dat \
  /path/to/sternheimer_galerkin_fixed_ao.dat \
  /path/to/SG15/H_TZDP/info/8/ORBITAL_RESULTS.txt \
  /path/to/SG15/H_TZDP/H_gga_8au_100Ry_3s2p.orb \
  /path/to/new-output-directory \
  --reference-commit 142b090e2babbc0d1cf1831c165d19a03ef56526 \
  --sidecar-commit bc720617aa058ab14823b5104b6657dc549b2d7d \
  --siab-commit COMMIT
```

Add deterministic radial shells by listing their angular momenta in order.
For example, this constructs `3s3p2d` from the original `3s2p` coefficients and
then optimizes all non-frozen radial columns:

```bash
python run_h_response_optimization.py \
  /path/to/grid-reference \
  /path/to/sternheimer_galerkin_primitive.dat \
  /path/to/sternheimer_galerkin_fixed_ao.dat \
  /path/to/SG15/H_TZDP/info/8/ORBITAL_RESULTS.txt \
  /path/to/SG15/H_TZDP/H_gga_8au_100Ry_3s2p.orb \
  /path/to/new-output-directory \
  --reference-commit 142b090e2babbc0d1cf1831c165d19a03ef56526 \
  --sidecar-commit bc720617aa058ab14823b5104b6657dc549b2d7d \
  --siab-commit COMMIT \
  --append-l 1 2 2 \
  --max-steps 500 \
  --maximum-step 20
```

For each requested `l`, the runner forms the metric-orthogonal complement of
the current radial columns in the Bessel parent. It temporarily appends every
complement direction, evaluates the full-frequency Pi-matrix loss, and keeps
the lowest-loss direction. Repeated values append multiple radials in the same
angular channel. The radial metric is averaged equally over all magnetic
components because one radial coefficient column is shared by every `m`.
`optimization.json` records every candidate loss, selected mode, metric
condition, magnetic-component metric deviation, and orthogonality errors under
`basis_extensions`.

The optimizer uses a monotone Armijo backtracking line search. Frozen columns
are restored exactly at every trial. Before optimization, the original three
SG15 s orbitals are rotated within their unchanged span so the first radial
column is the exact occupied atomic eigenstate. This column remains frozen;
therefore the active-spin response space always has eight virtual directions
and cannot gain an unphysical tenth direction by moving the fixed occupied
state outside the nine-AO basis. `optimization.json` records this rotation and
every accepted step, including the full-frequency response loss, maximum
frequency error, gradient norms, retained response ranks, overlap condition
number, and the RPA correlation energy as a diagnostic. A converged optimization
does not by itself prove that `3s2p` has enough response capacity; that is
decided from the final response residual and held-out molecular tests.
