# C jY All-Channel Contraction Design

## Goal

Construct the smallest practical C LCAO basis whose ordinary all-band SOS RPA
correlation energy reproduces the accepted `spdf` jY mother result and hence the
fixed Delta-ST reference within `0.1 eV/C`.  The first production target is
diamond C.  Molecular transferability, BSSE, and GW-specific fitting are outside
this stage.

## Physical Definition

The accepted `spdf` jY calculation is the high-rank SOS mother space.  It has
496 AO/C and gives `Ec=-0.5091040675004793 Ha/C2`, only
`0.0732014929 eV/C` from the fixed Delta-ST reference
`-0.5144842779929139 Ha/C2`.  A compact basis therefore need not reproduce all
raw spherical-Bessel primitives equally.  It must reproduce the response-active
subspace of this mother calculation over all eight q stars, 64 k sectors, and
12 frequencies.

All retained `s`, `p`, `d`, and `f` radial contractions are variational.  The old
SG15 `3s3p2d` span is an initialization and comparison point, not a frozen
prefix.  The frozen-`spd`, single-`f` calculation is retained only as an
ablation.

## Objective And Constraints

For sector `a=(q,k)` let `C` be the shared radial contraction, `E_v,a` the
stored virtual mother embedding, and `K_a` the response covariance containing
the q, k, occupation, and frequency weights exactly once.  The response loss is

```
L_resp(C) = sum_a [Tr(K_a) - Tr((E_v,a C)^+ K_a (E_v,a C))]
            / sum_a Tr(K_a).
```

The target artifact is extended with the occupied mother embedding `E_o,a`.
The combined retained-frame embedding reconstructs the candidate overlap in
the represented mother space:

```
Y_a = [E_o,a; E_v,a] C,
S_a(C) = Y_a^dagger Y_a.
```

After Lowdin orthogonalization of `S_a(C)`, occupied capture is measured from
`E_o,a C S_a(C)^(-1/2)`.  Every accepted optimization step must satisfy a fixed
minimum occupied-capture floor.  An occupied residual is included to guide the
gradient, but the hard capture gate, not its weight, defines feasibility.

After an occupied-safe response contraction is obtained, refine and rank it by
the actual per-q/per-frequency RPA trace-log mismatch to the accepted jY mother.
The final acceptance quantity remains the ordinary LibRPA SOS energy, not the
snapshot loss.

## Rank Ladder

Evaluate the nested all-channel profiles `3s3p2d1f`, `4s4p3d2f`,
`5s5p4d3f`, and `6s6p5d4f`.  Promotion is sequential.  A larger profile is used
only when the smaller profile cannot meet the response, occupied, PBE, and RPA
gates.  No `g` channel is added before the `spdf` contraction limit is measured.

Before interpreting a failed compact profile as insufficient rank, compute the
independent-sector Ky Fan lower bound.  For each `(q,k)` covariance, reserve its
occupied rank and allow the remaining AO columns to span that sector's leading
response eigenvectors independently.  The discarded eigenvalue sum is a strict,
optimistic lower bound on the snapshot residual at that AO count.  If this bound
is already large, the AO rank is insufficient for the stored target.  If it is
small while the shared contraction remains inaccurate, the failure belongs to
the common atom-centred subspace, its parameterization, or its optimization.
Because different sectors are allowed different subspaces, this diagnostic
cannot itself produce an orbital file or establish RPA accuracy.

## Gates

1. Target integrity: 512 sectors, immutable hashes, finite covariance, exact
   mother-Pi reconstruction, and occupied plus virtual embeddings spanning the
   retained mother rank.
2. Contraction: all requested radial columns are free, response loss decreases,
   occupied capture stays above its floor, overlap remains finite, and the final
   coefficients equal the best accepted checkpoint.
3. Rank diagnosis: report the independent-sector residual lower bound separately
   from the achieved shared-contraction residual.  Never call the optimistic
   bound a realizable compact basis.
4. PBE: the compact candidate must not materially damage the accepted diamond
   PBE occupied manifold; the existing 10 meV comparison remains a diagnostic
   gate rather than a frozen-orbital constraint.
5. RPA refinement: all q stars and all 12 frequencies use the same response
   definition as the mother target; per-sector mismatches are retained so total
   energy cancellation cannot hide a bad channel.
6. Physical acceptance: ordinary all-band LCAO SOS RPA with PCA `1e-6` and qavg
   head/wing differs from Delta-ST by less than `0.1 eV/C`.

## Evidence Boundary

The jY mother establishes that the finite LCAO/SOS route can reach the target.
It does not prove that a particular compact rank can.  Galerkin contraction is
an optimizer and screening method; only the final ordinary SOS calculation is
the physical result.  Auxiliary-basis/LRI optimization remains separate.
