# SIAB Compact Local Response Basis Design

## 1. Goal

Replace the impractical `13s11p10d5f4g` H basis with a frozen, compact
sequence that preserves the H DZP DFT core, represents the physical H and H2
Delta-Sternheimer first-order wavefunctions, and suppresses diffuse radial
subspaces before any RPA energy is inspected.

The production acceptance target is

```text
D_raw ~= D_CP ~= D_Delta-ST
```

within `0.1 kcal/mol`. Counterpoise remains a final diagnostic. It is not a
training target and is not used to repair an unsuitable basis.

## 2. Evidence And Corrected Scope

The completed greedy campaign reached `167 AO/H` because it optimized the
capture of the part of the response that is representable in the current
`l<=4`, 25-radial-function primitive pool. Its stopping value of `0.999019`
does not mean that the full first-order wavefunction error is `0.1%`; the
remaining absolute normalized loss is about `9.88%` for H and `12.86%` for
H2. The all-band SOS result then showed severe high-band growth and the
full-size H+ghost producer exceeded one-node memory.

Those results are a diagnostic of the oversized basis, not a reason to make
counterpoise production the optimization mainline. This stage therefore does
not modify the H+ghost producer. It uses only the physical H and H2 targets to
construct and optimize orbitals.

## 3. Fixed And Variable Spaces

The fixed DFT core is the H `1s,2s,1p` DZP prefix. The H TZDP `3s2p` file is
the initial coefficient file, so its upper `3s,2p` columns remain variable and
provide a smooth DFT-compatible starting point. Added response orbitals are
initialized from Delta-ST residual modes and optimized with `st_dpsi_joint`.

The first compact campaign keeps the validated `100 Ry`, `8 bohr`, 25-column
Bessel coefficient representation and `l<=4`. This isolates the compression
and locality change. A later producer campaign may extend the primitive target
to `l<=8`; it must be a new immutable target because the present target files
contain no `l>4` blocks.

## 4. Radial Locality Metric

For angular channel `l`, let `B_l(r)` be the SIAB radial primitive row vector,
including the same cutoff smoothing used for the emitted numerical orbitals.
Define the full and tail radial metrics

```text
S_l = integral_0^Rcut B_l(r)^T B_l(r) r^2 dr
T_l = integral_Rloc^Rcut B_l(r)^T B_l(r) r^2 dr.
```

Let `C_f,l` contain fixed DZP columns and `C_v,l` contain variable columns.
Project the variable columns out of the fixed radial subspace,

```text
Cbar_v,l = C_v,l
           - C_f,l (C_f,l^T S_l C_f,l)^-1 C_f,l^T S_l C_v,l.
```

For a nonempty variable subspace, its rotation- and scale-invariant mean tail
fraction is

```text
tau_l = Tr[(Cbar_v,l^T S_l Cbar_v,l)^-1
           (Cbar_v,l^T T_l Cbar_v,l)] / n_v,l.
```

The AO-weighted locality loss is

```text
L_tail = sum_l (2l+1) n_v,l tau_l
         / sum_l (2l+1) n_v,l.
```

The implementation rejects a non-positive or over-conditioned projected
radial Gram matrix rather than adding a hidden diagonal shift. The fixed DZP
columns are never changed by this term.

## 5. Optimization Loss

The production loss becomes

```text
L = L_ST(H,H2)
    + w_dpsi L_dpsi
    + w_tail L_tail
    + DFT/dpsi constraint penalties.
```

`w_tail=0` is exactly backward compatible. A positive weight and `Rloc` are
explicit committed input parameters. Every accepted optimizer point records
the unweighted `L_tail` and weighted locality regularization separately from
the Sternheimer and dpsi components.

This is a locality constraint on the one-electron AO space. It is not a
counterpoise term and does not use H+ghost data.

## 6. Compact Candidate Frontier

The existing atomic residual generalized eigensolver remains the initializer.
For each `l`, it proposes the leading mode after projecting out the current
basis. Candidate gains are still evaluated with the exact full H and H2 AO
projectors and divided by the AO cost `2l+1`.

The compact driver differs from the old convergence driver in three ways:

1. it freezes every intermediate basis as a candidate instead of declaring
   `0.999` representable-space capture to be the required endpoint;
2. it stops at a committed `max_ao_per_atom` budget;
3. it records radial tail fractions and rejects an optimized step that exceeds
   the committed locality or overlap-condition gate.

The first production budget is `48 AO/H`. This includes the full TZDP initial
space and is less than one third of the failed `167 AO/H` basis. The complete
nested sequence up to the budget is frozen before any new SOS energy is run.

## 7. Physical Validation

The first validation pass runs all-band, full-Coulomb SOS for H and H2 for the
frozen compact candidates. It reports `D_raw`, AO count, radial tails, H/H2
response losses, and overlap condition. Energy does not change the already
frozen shell order or coefficients.

Only candidates with a promising raw result and acceptable overlap are run as
H+ghost counterpoise controls. The final basis must satisfy both

```text
abs(D_raw - D_Delta-ST) < 0.1 kcal/mol
abs(D_raw - D_CP) < 0.1 kcal/mol.
```

If no candidate passes, the next scientific change is to add a Pi-sensitive
target and/or extend the primitive target to `l<=8`. It is not to continue the
old `0.999` campaign beyond the AO budget.

## 8. Pi Loss Boundary

The current Sternheimer SIAB file stores response norms, primitive overlaps,
and first-order-wavefunction projections. It does not store the complete
auxiliary-index response matrix needed to compare

```text
Pi(iw) = V^(1/2) chi0(iw) V^(1/2).
```

Therefore this stage must not label ordinary first-order-wavefunction spillage
as a Pi loss. A later format version will add the auxiliary-channel contraction
needed for a direct Pi metric, with the same full-Coulomb whitening convention
as Delta-ST and LibRPA.

## 9. Verification And Commit Boundaries

1. Commit this design and its executable implementation plan.
2. Add radial-locality tests and implementation; verify old zero-weight results
   are byte-compatible at the loss-component level.
3. Add the AO-budget frontier and deterministic manifest tests.
4. Run the complete SIAB suite on `df_dcu normal` with the frozen source SHA.
5. Run the compact H/H2 optimizer and record actual candidate metrics.
6. Submit SOS only after the compact sequence is frozen.

