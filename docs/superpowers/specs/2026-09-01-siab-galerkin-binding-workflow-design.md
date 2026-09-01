# Reusable Galerkin Binding-Basis Workflow

## Goal

Replace material-specific basis searches with one manifest-driven workflow that
uses Galerkin response calculations to generate and screen orbital candidates,
then uses ordinary all-band SOS only for physical promotion and acceptance.
The same workflow must support molecular targets such as `2 H - H2` and solid
targets such as `C_atom - 0.5 C2_solid`.

## Boundary

Galerkin response is an inner-loop model.  It may generate candidates, reject
invalid candidates, and rank candidates when its ranking has been calibrated.
It may not publish a basis.  The final observable remains the ordinary SOS
binding energy formed from independently recalculated fragments and bonded
systems.  Delta-ST is a frozen external reference and is not rerun.

## Architecture

The workflow has three reusable units.

1. A Galerkin candidate generator evaluates each named physical family's loss
   and tangent-space gradient at one immutable starting orbital.  It creates a
   small deterministic Pareto bank from normalized family gradients while
   preserving the fixed radial prefix and orthonormality constraints.
2. A pure state evaluator reads immutable evidence manifests and assigns one
   state: `galerkin_screen`, `pbe_gate`, `tail_gate`, `proxy_gate`,
   `full_q_gate`, `accepted`, or `rejected`.  It never submits a job.
3. A system adapter supplies stoichiometry, open-shell occupation rules,
   q-point representatives and multiplicities, PBE thresholds, and the final
   binding-energy tolerance.  Submission wrappers execute only the single next
   action returned by the state evaluator after duplicate checks.

## Candidate Generation

For family loss `L_F(C)`, compute its Euclidean gradient with respect to every
unfrozen radial coefficient block, then project that gradient onto the tangent
space orthogonal to the fixed prefix and to the current variable frame.  The
projected family gradients are normalized as a complete coefficient vector.

For two families `A` and `B`, create the fixed Pareto set

```text
w = 0.25, 0.50, 0.75
d_w = -(w grad(L_A)/||grad(L_A)||
        + (1-w) grad(L_B)/||grad(L_B)||).
```

Use a bounded, configured trust radius and retract every trial back to the
fixed-prefix Stiefel manifold.  A candidate enters the bank only when it is
finite, preserves the prefix exactly, satisfies occupied capture and overlap
conditioning, and improves at least one family without exceeding the configured
degradation allowance for the other family.  The bank records every input
hash, family loss, gradient norm and cosine, direction weight, trust radius,
and output orbital hash.

The family loss is a candidate generator, not a binding-energy surrogate.  In
particular, no signed atom-solid cancellation is inferred from squared Pi loss.

## Promotion Sequence

Every system adapter uses the same monotone sequence.

1. Galerkin training and held-out response gates.
2. Radial smoothness, occupied-space, overlap and virtual-spectrum gates.
3. Pure PBE gate against the immutable unoptimized basis.
4. A small numerical-stability gate when the system needs one.
5. A calibrated ordinary-SOS proxy gate.
6. The complete ordinary-SOS physical gate.

A failed gate is terminal for that candidate.  A missing gate yields exactly
one next action.  A completed fingerprint is reused and an active fingerprint
is reported instead of resubmitted.

## C Adapter

The first adapter is diamond C with an isolated open-shell atom.

```text
candidate layout: 3s3p2d, 22 AO/C
fixed prefix:      2s2p1d
families:          C_atom and C_solid
PBE gate:          10 meV for atom, solid per C, and binding
stability gate:    q2/q6 six-frequency high-tail screen
ranking proxy:     atom plus q6/q7/q8 ordinary SOS
final q stars:     1,2,3,6,7,8,11,28
reference:         6.902326 eV/C
acceptance:        absolute error below 0.1 eV/C
```

The adapter must enforce the atom's `Nup=3`, `Ndown=1` occupation and occupied
subspace gauge.  The q-star proxy remains empirical: its accepted calibration
has leave-one-out maximum error 0.001912 eV/C, but the final eight-q result is
still required for publication.

## H2 Adapter

The H/H2 adapter uses the same candidate generator and state evaluator with
`2 H - H2`, molecular geometries instead of q stars, and its own ordinary-SOS
promotion evidence.  Counterpoise is not part of this C workflow.  No H2
numerical threshold is copied into C.

## Acceptance And Provenance

Every stage writes `STATUS.json`, `PROVENANCE.json`, and a result JSON with
finite values and SHA256 identities.  Scheduler completion, program success,
numerical validity, and physical acceptance are separate fields.  The final C
report gives PBE binding, RPA correlation binding, total ordinary-SOS binding,
and the difference from 6.902326 eV/C.

## Tests

Unit tests must prove tangent projection, fixed-prefix preservation, deterministic
candidate ordering, family degradation rejection, monotone state transitions,
duplicate reuse, and terminal rejection.  Contract tests must prove that the C
adapter contains the frozen PBE, q-star, LibRPA, product-PCA, full-Coulomb and
acceptance settings.  No production code is added before its test fails.
