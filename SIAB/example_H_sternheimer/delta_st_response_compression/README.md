# H Delta-ST Response Compression Gradient Gate

This gate checks that the original H `3s2p` SIAB basis has a usable descent
direction toward an immutable uniform-grid Delta-ST response. It does not run a
long optimization and it does not fit the scalar RPA correlation energy.

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

The optimizer uses a monotone Armijo backtracking line search. Frozen columns
are restored exactly at every trial. `optimization.json` stores every accepted
step, including the full-frequency response loss, maximum frequency error,
gradient norms, retained response ranks, overlap condition number, and the RPA
correlation energy as a diagnostic. A converged optimization does not by itself
prove that `3s2p` has enough response capacity; that is decided from the final
response residual and held-out molecular tests.
