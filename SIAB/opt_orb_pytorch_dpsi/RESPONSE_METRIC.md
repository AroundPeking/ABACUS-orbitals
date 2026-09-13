# Response metric diagnostic

This NumPy-only postprocessor does not change the Galerkin solver or the
production RPA path. It measures compression of response projections within
a supplied mother basis, not an RPA energy error.

For overlap `S = U diag(s) U^H` and bra rows `Q = <psi|B>`, retain
`s > rtol * max(s)` and recover orthonormal coordinates
`X = diag(s)^(-1/2) U^H Q^H`. Ket coefficients are
`c = U diag(s)^(-1) U^H Q^H`. The conjugation of Q is essential.
`Metric.restore` reports the residual against `Q^H`; this does not measure
unknown response outside the mother span. Check several relative cutoffs.

`covariance(X, weights)` and `spectrum(covariance)` report weighted response
norms and the optimal RMS compression error by rank. Snapshot weights must
be finite and nonnegative. Use separately documented frequency, occupied,
k and q conventions. Do not combine different Bloch sectors as if they
shared a single physical Hilbert space.

Materially indefinite or non-Hermitian inputs fail. Tiny negative covariance
eigenvalues within the numerical tolerance are clipped only for the tail
calculation; the raw minimum is retained in the report.

The reported rank is not a number of atom-centered radial functions.
Truncating a POD expansion is not the same operation as solving a Galerkin
equation in that truncated space. Frozen-H Galerkin observables and local
NAO fitting are subsequent, separately validated stages.

Run from the repository root:

```sh
PYTHONPATH=SIAB/opt_orb_pytorch_dpsi python -m unittest discover -s SIAB/tests -p test_response_metric.py
```

The initial diagnostic was tested on one destination k sector of the frozen
C Gamma data, twelve frequencies, in read-only DF job 3333617. Its production
inputs and case-specific provenance remain separate from this reusable core.

## Frozen-H observable diagnostic

`response_galerkin.py` consumes an orthonormal-coordinate Hamiltonian, source
bras, and fixed occupied projections. `occupied_frames` preserves that
occupied span; `virtual_pod` orders the covariance within its complement.
`galerkin_pi` solves the virtual reduced problem spectrally and applies k and
occupation weights once. It does not include q or frequency quadrature weights
and does not add a spin factor. Nonpositive virtual gaps fail explicitly.

The total trial dimension is occupied rank plus virtual POD rank. This is
different from the unconstrained response-rank diagnostic above. The helper
does not convert those dimensions into an atom-centered NAO basis.

`trace_log` checks positive I-Pi and evaluates Tr(log(I-Pi)+Pi). A single-k
Pi must be compared with its same-k reference. If only total-q Pi is known,
replacing a mother-space k contribution with a compressed k contribution
defines a hypothetical sensitivity, not a physical full-q RPA error.

DF read-only job 3333756 verified a 44-dimensional trial against the unchanged
production Galerkin implementation at one frequency (relative Pi difference
6.81e-15). The full retained 378-dimensional virtual complement is recovered
at the end of the rank sweep (relative Pi difference 1.07e-14).

```sh
PYTHONPATH=SIAB/opt_orb_pytorch_dpsi python -m unittest discover -s SIAB/tests -p test_response_galerkin.py
```
