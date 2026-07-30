# SIAB Low-Frequency Sternheimer Guard Design

## 1. Goal

Improve the fixed-DZP H-TZDP optimization without changing the response target,
orbital count, or held-out physics test. The existing integrated Sternheimer
spillage remains the primary response loss. The optimizer additionally reports
the loss at every imaginary frequency and rejects candidates whose lowest-
frequency loss is worse than the initial TZDP value.

This is a controlled test of the observed failure mode: the current optimized
`3s2p` basis reduces the integrated Sternheimer loss from `0.4428607140` to
`0.4054568603`, but the lowest-frequency local loss increases from `0.247384`
to `0.252711`. The new guard must remove that regression while fixed
`1s,2s,1p` DZP orbitals and the original DFT/dpsi constraints remain unchanged.

The 0.74085-A H2 SOS-RPA binding energy and counterpoise correction remain
held-out validation quantities. They are not added to the training loss.

## 2. Why The Integrated Loss Remains Primary

For a Sternheimer reference row
\(\rho=(i,a,p)\), let \(i\) be the occupied state, \(a\) the Coulomb-whitened
auxiliary perturbation, and \(p\) the imaginary-frequency index. After
projecting both the target and candidate outside the fixed DZP space, define

\[
  \bar n_\rho
  =
  \langle\bar{\delta\psi}_\rho|
  \bar{\delta\psi}_\rho\rangle,
  \qquad
  r_\rho(C)
  =
  \bar n_\rho
  -
  \langle\bar{\delta\psi}_\rho|
  P_C|\bar{\delta\psi}_\rho\rangle .
\]

Here \(P_C\) is the metric projector onto the variable `3s,2p` orbital
subspace. The current loss is

\[
  L_{\rm ST}(C)
  =
  \frac{\sum_p w_p R_p(C)}
       {\sum_p w_p N_p},
  \qquad
  R_p(C)=\sum_{\rho\in p} f_i r_\rho(C),
  \qquad
  N_p=\sum_{\rho\in p} f_i\bar n_\rho ,
\]

where \(f_i\) is the occupation and \(w_p\) is the GreenX minimax integration
weight. This objective weights a relative frequency error by both its
quadrature weight and the response norm at that frequency. It therefore
retains the physical decay of the response at high frequency.

The alternative

\[
  \frac{1}{N_\omega}\sum_p\frac{R_p}{N_p}
\]

is not used as the production objective. Equal frequency weights would give a
small high-frequency response the same importance as the low-frequency
response and can move the optimization away from the RPA energy sensitivity.
It may be evaluated later as an explicit ablation, but it is outside this
implementation.

## 3. Frequency-Resolved Diagnostic

For every distinct frequency in the target, report

\[
  \ell_p(C)=\frac{R_p(C)}{N_p}.
\]

The grouping uses the exact frequency values stored in the versioned producer
file. The producer writes the same binary64 frequency for all rows belonging
to one minimax point, so no numerical clustering tolerance is introduced.
Rows at a frequency are weighted by occupation only; the common GreenX weight
\(w_p\) is applied only in the integrated loss.

The diagnostic result contains sorted vectors for:

- frequency in Hartree;
- occupation-weighted residual \(R_p\);
- occupation-weighted projected norm \(N_p\);
- local loss \(\ell_p\).

All entries must be finite, every \(N_p\) must be positive, and every local
loss must be nonnegative up to the existing row-local roundoff policy. The
calculation continues to use complex `q` overlaps and real orbital
coefficients. A global phase rotation of any complex reference row must leave
all integrated and frequency-resolved losses unchanged.

## 4. Lowest-Frequency Guard

Let \(p_0\) be the smallest positive stored frequency and let \(C_0\) be the
initial unmodified TZDP coefficient set. Define

\[
  g_{\rm low}(C)
  =
  \max\left[
    0,\,
    \frac{\ell_{p_0}(C)}
         {\ell_{p_0}(C_0)}
    -1-\tau_{\rm low}
  \right],
\]

and add the differentiable penalty

\[
  L_{\rm low}(C)=\lambda_{\rm low}g_{\rm low}(C)^2
\]

to the existing `st_dpsi_joint` objective. The first production comparison
uses

```text
low_frequency_guard_weight = 10.0
low_frequency_guard_tolerance = 0.0
```

The initial lowest-frequency loss must be greater than the configured
numerical epsilon when the guard is active. Otherwise the input is rejected
because a relative guard would be undefined.

The penalty guides optimization, but acceptance is a separate hard condition:
an accepted candidate must satisfy

\[
  \ell_{p_0}(C)
  \le
  (1+\tau_{\rm low})\ell_{p_0}(C_0)
\]

within a `1e-12` relative numerical comparison allowance. This prevents the
optimizer from selecting a lower-total-loss point that pays a finite penalty
but still regresses at the lowest frequency.

## 5. Configuration And Compatibility

Add two optional keys to the existing optimizer loss configuration:

```json
{
  "low_frequency_guard_weight": 0.0,
  "low_frequency_guard_tolerance": 0.0
}
```

The zero weight is the default. With this default:

- the scalar integrated Sternheimer loss is numerically unchanged;
- candidate acceptance and selection are unchanged;
- existing inputs require no edits;
- `Spillage.dat` and `ORBITAL_RESULTS.txt` retain their existing field sets and
  values, so strict external parsers do not need edits.

When the weight is positive, the optimizer records two additional components:

```text
sternheimer_lowest_frequency
regularization_low_frequency
```

The saved baseline records `sternheimer_lowest_frequency`. The accepted-result
metadata also records the frequency value, initial local loss, final local
loss, tolerance, and guard weight. Unknown keys and negative or non-finite
values remain fatal input errors.

## 6. Code Boundaries

The implementation is limited to the SIAB repository:

- `sternheimer_spillage.py` computes frequency-resolved residuals and norms;
- `optimization_loss.py` validates guard options and composes the penalty;
- `opt_orbital_converge.py` enforces the hard acceptance condition and writes
  diagnostics;
- SIAB input readers continue to pass the normalized loss dictionary without
  a new file format;
- the fixed-DZP H example receives a separate guarded input and comparison
  runner.

No ABACUS producer, LibRPA reader, Coulomb kernel, minimax grid, or
Sternheimer equation is changed.

## 7. Test Strategy

Implementation follows test-driven development.

1. A synthetic two-frequency target verifies \(R_p\), \(N_p\), and
   \(\ell_p\) against direct formulas.
2. A phase-rotated complex target gives identical frequency-resolved losses.
3. The default zero guard reproduces the previous total loss and acceptance.
4. An active guard is zero at the baseline, positive for a regressed
   low-frequency candidate, and differentiable by finite differences.
5. A candidate with improved integrated loss but regressed lowest-frequency
   loss is rejected.
6. Invalid guard weights, tolerances, and zero baseline loss are rejected with
   explicit errors.
7. The complete SIAB regression suite remains green.

## 8. Physical A/B Validation

Run the same-size H optimization from the same initial TZDP coefficients and
the same H Sternheimer target in two lanes:

| Lane | Integrated ST loss | Low-frequency guard | Variable orbitals |
|---|---:|---:|---|
| Current control | unchanged current objective | off | `3s,2p` |
| Guarded candidate | unchanged current objective | on | `3s,2p` |

Both lanes freeze `1s,2s,1p` exactly and use the same DFT/dpsi data,
optimizer, radial primitives, random seed, and stopping criteria. Before the
held-out calculation, require:

- fixed-orbital coefficient differences below `1e-12`;
- guarded lowest-frequency loss no greater than the initial TZDP value;
- DFT and dpsi constraints satisfied;
- integrated ST loss no greater than `1.01 * 0.4054568603 =
  0.4095114289`, a one-percent relative allowance from the current optimized
  candidate;
- identical orbital count and radial cutoff.

Then run the established H2, H, and H+ghost SOS-RPA workflow with identical
20-A cell, 0.74085-A bond, orbital-independent auxiliary basis, full Coulomb,
16 fixed frequencies, and LibRPA executable. Report raw and counterpoise-
corrected binding energies. The guarded basis is physically better only if
the held-out H2 result moves toward the converged Delta-ST/FHI-aims reference
without degrading the DFT/dpsi gates. A successful software test without that
held-out improvement is recorded as a negative physics result, not as a basis
optimization success.
