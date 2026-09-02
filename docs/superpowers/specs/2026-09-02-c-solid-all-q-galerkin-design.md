# C Solid All-Q Galerkin Basis Workflow

## Goal

Optimize a diamond-C NAO basis for solid RPA and GW observables without using
an isolated C atom or an atom-solid binding-energy target.  The inner loop uses
frozen Galerkin response data from every symmetry-inequivalent q star of the
`4 x 4 x 4` mesh.  Ordinary all-band SOS remains the physical promotion gate.

## Physical Boundary

The eight q-star representatives are `1, 2, 3, 6, 7, 8, 11, 28` with frozen
star multiplicities.  Their weighted sum is the complete 64-q result for this
mesh; it is not proof of q-mesh convergence.  A reduced q set may be used only
for regression tests or an explicitly labelled screening estimate.

The workflow requires a named solid-only reference observable.  A large-basis
ordinary-SOS energy may be recorded as a numerical baseline, but it is not an
independent physical reference.  A binding-energy reference may not be
substituted for a solid correlation energy or a GW quasiparticle observable.

## Architecture

Add a solid-only adapter beside the existing atom-solid workflow.  It reads an
immutable config, an initial orbital, and one periodic Galerkin dataset for
each required q star.  It does not accept atomic response or atomic source
arguments.

The adapter has four responsibilities:

1. validate the q-star identity, multiplicity, frequency grid, orbital,
   pseudopotential, auxiliary basis, executable and source provenance;
2. evaluate the existing q-weighted periodic Galerkin loss as one `C_solid`
   family, preserving the core implementation's exact q and frequency weights;
3. generate a bounded deterministic solid descent candidate while preserving
   the configured fixed radial prefix and occupied-capture constraint;
4. write status, provenance, gradient, candidate and target-reference records
   with input and output hashes.

The core periodic loss already performs q-weighted accumulation.  This stage
therefore adds an adapter and stricter dataset contract, not a second loss
implementation.

## Objective And Candidate Policy

The first implementation uses the existing normalized projected-response loss
over all eight q stars.  It records enough per-q information to add an
energy-sensitivity weighting later without changing the dataset contract.
The scalar ordinary-SOS energy is never the only response constraint.

Start from the accepted reverse-p3 `3s3p2d` basis.  Reoptimize its available
s/p/d directions under the all-q solid objective before assuming that an f
channel is required.  A nested single-f direction may be proposed only when
its all-q gradient is linearly independent and predicts a larger stable
decrease than the s/p/d directions.  Use a small trust radius and freeze one
candidate at a time.

## Dataset Contract

A production all-q run must contain exactly the configured eight logical
q-star labels, with no duplicates or omissions.  Every dataset must have:

- `status success` and `all_converged yes`;
- the configured star weight and six-frequency grid;
- identical orbital, pseudopotential, auxiliary basis, primitive-block,
  ABACUS commit and executable identities where the format provides them;
- full periodic Coulomb and the same `4 x 4 x 4` k/q definition;
- finite matrices and a finite solver residual below the configured limit.

A reduced-set mode is allowed only when the config explicitly names the
expected subset.  Its output must state `coverage = reduced` and cannot set a
physical-release gate.

## Promotion And Validation

The expensive path is monotone:

1. complete eight-q Galerkin evaluation and candidate freeze;
2. solid PBE per-C energy, gap, band order and virtual-spectrum gate;
3. q2/q6 six-frequency high-tail stability gate;
4. direct solid ordinary-SOS proxy from the configured q subset;
5. complete eight-q ordinary all-band SOS and LibRPA energy;
6. q-mesh, frequency and Gamma head/wing convergence of the accepted basis;
7. GW band-edge validation when the intended output is GW.

No atom job is part of this route.  Scheduler completion, numerical validity,
Galerkin improvement, ordinary-SOS agreement and GW acceptance remain separate
states.

## Efficiency

The missing frozen q-star responses are produced once.  Candidate gradients,
trust-region steps and complete eight-q Galerkin evaluations then reuse those
files without rerunning ABACUS.  Most rejected candidates stop at the offline
or q2/q6 gates.  Only a candidate whose conservative solid-only proxy reaches
the target interval receives the remaining ordinary-SOS q stars.

## Tests

Tests are written before implementation and must prove:

- a valid eight-q manifest passes with the exact labels and weights;
- missing, duplicate or misweighted q stars fail;
- mismatched source, orbital, auxiliary or frequency provenance fails;
- reduced coverage is explicit and cannot release physics;
- no atomic input is accepted or required;
- all eight datasets form one `C_solid` loss family;
- output ordering and hashes are deterministic;
- the existing atom-solid adapter remains unchanged.
