# SIAB Source-Aware Projected-Pi Loss Design

## 1. Goal

Replace the diagonal first-order-wavefunction spillage as the primary SIAB
response objective with a loss that preserves the auxiliary-channel cross
contractions entering the independent-particle polarizability. The optimized
orbitals remain an atomic wavefunction basis: ABACUS still provides accurate
Sternheimer first-order wavefunctions, and SIAB still varies radial orbital
coefficients. This design does not fit a density basis or diagonalize
`chi0 * V`.

The first experiment keeps the current H basis size fixed: freeze DZP
`1s,2s,1p` and optimize only the remaining TZDP `3s,2p` orbitals. H and H2 are
equal-weight physical training families. H+ghost remains a held-out
counterpoise and basis-borrowing diagnostic and cannot affect the loss or
candidate selection.

The implementation begins with a source-only feasibility gate. It reuses the
existing expensive v1 Sternheimer response targets and computes only the
missing source overlaps in a much cheaper ABACUS run. Optimization code is
added only if the resulting projected response metric agrees with the known
held-out SOS-RPA trend.

## 2. Why The Existing Loss Is Insufficient

For a reference row `rho = (i,b,p)`, the existing v1 target stores

\[
  Q_{ibp,e} = \langle\delta\psi_{ib}(i\omega_p)|B_e\rangle,
  \qquad
  n_{ibp}=\langle\delta\psi_{ib}|\delta\psi_{ib}\rangle,
\]

where `i` is an occupied state, `b` is a full-Coulomb-whitened auxiliary
perturbation, `p` is an imaginary-frequency index, and `B_e` is a primitive
atomic orbital. The current loss minimizes the sum of row-local projection
residuals. It can improve every individual projected norm while changing the
relative phase and direction needed by a different source channel `a`.

The physical response instead contains

\[
  A_{ab,p}
  = \sum_i f_i
    \langle\psi_i\bar v_a|\delta\psi_{ib}(i\omega_p)\rangle,
  \qquad
  \Pi_p=A_p+A_p^\dagger .
\]

Consequently, a row-diagonal spillage is not guaranteed to rank bases in the
same order as the RPA response or correlation energy. The completed H+H2
compact-basis campaign already rules out simply adding more H2 rows to that
same objective.

## 3. Units, Kernel, And Complex Convention

The ABACUS SIAB producer constructs Coulomb-orthonormal perturbations

\[
  \bar v_a(\mathbf r)=\sum_\mu v_\mu(\mathbf r)W_{\mu a},
  \qquad W^\dagger V_{\rm full}W=I,
\]

using the global full-Ewald Coulomb matrix. Therefore `Pi` is already in the
symmetrized full-Coulomb representation. SIAB must not multiply the source or
response matrices by another Coulomb matrix or square root.

The sampled `bar v_a` is in Hartree. ABACUS multiplies it by two only when
forming the right-hand side of the Sternheimer equation because that solver's
Hamiltonian, eigenvalues, and frequency are in Rydberg. The source overlap
defined below must use the unscaled Hartree potential. No factor of two is
introduced in SIAB.

The canonical v1 producer convention is

\[
  Q_{ibp,e}=\langle\delta\psi_{ibp}|B_e\rangle,
  \qquad
  S_{ef}=\langle B_e|B_f\rangle .
\]

The new sidecar uses the matching row convention

\[
  D_{ia,e}=\langle\psi_i\bar v_a|B_e\rangle .
\]

Both `Q` and `D` are complex128. SIAB orbital coefficients remain real
float64. The sign of the response is carried by `delta psi`, whose equation
has the negative perturbing source; `D` itself receives no extra minus sign.

## 4. Projected Response Formula

Let the columns of `C` define the complete candidate AO basis, including the
fixed DZP and variable response orbitals, in the primitive basis. Define

\[
  G_C=C^\dagger S C.
\]

The least-squares projection of a Sternheimer response onto that AO space
gives

\[
  A^{(C)}_{ab,p}
  =\sum_i f_i
    (D_{ia}C)G_C^{-1}(Q_{ibp}C)^\dagger,
\]

and

\[
  \Pi^{(C)}_p=A^{(C)}_p+A^{(C)\dagger}_p.
\]

The primitive-space reference is the same expression with the identity basis:

\[
  A^{(B)}_{ab,p}
  =\sum_i f_i D_{ia}S^+Q_{ibp}^\dagger,
  \qquad
  \Pi^{(B)}_p=A^{(B)}_p+A^{(B)\dagger}_p.
\]

`S+` is a Hermitian eigendecomposition pseudoinverse. Eigenvectors are kept
when `lambda / lambda_max > 1e-12`; the retained rank and the result's
sensitivity to thresholds `1e-10`, `1e-11`, and `1e-12` are reported in the
feasibility analysis. Candidate `G_C` continues to use the existing Cholesky
and condition-number checks so a singular optimized AO basis is rejected.

For one physical family `F`, define

\[
  L_{\Pi,F}(C)=
  \frac{\sum_p w_p
    \|\Pi^{(C)}_{F,p}-\Pi^{(B)}_{F,p}\|_F^2}
       {\sum_p w_p\|\Pi^{(B)}_{F,p}\|_F^2}.
\]

The production response loss is the equal-family sum

\[
  L_\Pi(C)=L_{\Pi,{\rm H}}(C)+L_{\Pi,{\rm H_2}}(C).
\]

Frequency weights are the stored GreenX minimax weights. Occupation is
included exactly once in `A`. No H+ghost row enters either numerator or
denominator.

This is a source-aware projection diagnostic, not an assertion that a
least-squares projected response equals the result of solving a new Galerkin
Sternheimer equation in the candidate AO basis. If this metric fails the
physical ranking gate below, the next design must export the primitive
Hamiltonian and solve that Galerkin problem explicitly.

## 5. Source Sidecar Interface

ABACUS writes `STERNHEIMER_SIAB_SOURCE_V1.dat` with these canonical sections:

```text
<STERNHEIMER_SIAB_SOURCE_HEADER>
format_version 1
n_source ...
n_primitive ...
n_blocks ...
grid_volume_bohr3 ...
</STERNHEIMER_SIAB_SOURCE_HEADER>
<PRIMITIVE_BLOCKS>
...
</PRIMITIVE_BLOCKS>
<SOURCE_METADATA>
# occupied_state auxiliary_channel occupation norm
...
</SOURCE_METADATA>
<OVERLAP_D>
# row-major complex values: real imaginary
...
</OVERLAP_D>
<OVERLAP_S>
...
</OVERLAP_S>
<PROVENANCE_JSON>
...
</PROVENANCE_JSON>
```

Each source key `(occupied_state, auxiliary_channel)` occurs exactly once.
`norm` is `\langle psi_i bar v_a | psi_i bar v_a\rangle` and provides a
projection-capture diagnostic; it does not enter `L_Pi`.

Whenever `out_sternheimer_siab` is enabled, a normal production run writes
both the existing response v1 file and this sidecar. A new Boolean input
`sternheimer_siab_source_only`, defaulting to false, writes the sidecar and
returns before allocating response matrices or solving any first-order
equation. It is legal only with `out_sternheimer_siab = 1` and
`out_sternheimer_librpa = 0`.

The source uses the same occupied grid wavefunctions, primitive reciprocal
matrix, full-Coulomb whitening transform, and Hartree potentials already
constructed by the SIAB path. Source rows are distributed by the existing
channel ownership rule, gathered deterministically, sorted by their key, and
written once by rank zero. The source-only path still performs the zero-order
ABACUS calculation; it skips all frequency-dependent Sternheimer solves.

## 6. Pairing A Sidecar With Existing V1 Responses

The SIAB reader accepts a source sidecar only when all of the following hold:

- primitive blocks and offsets are exactly identical;
- the duplicate primitive overlap matrices agree within `1e-13` relative
  Frobenius norm and `1e-14` maximum absolute difference;
- cell, `ecut`, kernel, orbital manifest, pseudopotential manifest, auxiliary
  basis manifest, PCA threshold, whitening dimensions, and whitening-transform
  SHA256 agree;
- every unique `(occupied_state, auxiliary_channel)` required by the response
  rows is present exactly once in the source file;
- occupations agree within `1e-14` absolute tolerance;
- all metadata and matrix entries are finite.

The producer commit, executable SHA256, MPI rank count, and OpenMP thread count
are recorded but are not required to match because the source-only feature is
necessarily produced by a newer executable. Their mismatch is printed in the
analysis provenance.

For the transitional reuse of existing v1 targets, the zero-order eigenvalues
and SCF completion markers from the source-only run must also be compared with
the archived target-run logs before the pair is admitted to the physical
ranking gate. A future target produced by the new executable obtains `Q` and
`D` in one run and does not need this transitional log check.

Any failed pairing check is fatal. SIAB never guesses channel order, pads a
missing source, conjugates data heuristically, or silently falls back to the
old spillage.

## 7. SIAB Loss And Compatibility

Add a new loss mode `pi_dpsi_joint`; do not change the numerical behavior of
`st_only`, `st_dpsi_joint`, or the low-frequency guard. The new mode composes:

\[
  L_{\rm total}
  =\lambda_\Pi L_\Pi
   +\lambda_{\rm dpsi}L_{\rm dpsi}
   +\lambda_{\rm DFT}L_{\rm DFT}
   +L_{\rm existing\ regularization}.
\]

The existing normalization policy for DFT and dpsi constraints is retained.
The initial implementation uses `lambda_Pi = 1` after family normalization.
The ordinary integrated Sternheimer spillage remains a reported diagnostic
and a hard no-regression gate relative to the initial TZDP basis. Fixed DZP
coefficient columns must remain bitwise unchanged through optimization output.

H+ghost is loaded only by held-out SOS-RPA postprocessing. Supplying a ghost
source sidecar to `pi_dpsi_joint` is rejected rather than ignored so it cannot
accidentally influence a later selector.

## 8. Test And Numerical Gates

Implementation follows test-driven development and separates four gates.

### 8.1 Producer gate

- A synthetic complex grid verifies that `D` equals direct
  `sum_r conj(psi_i(r) * v_a(r)) * B_e(r) * DeltaOmega`.
- A potential expressed in Hartree and the same potential multiplied by two
  for the Ry right-hand side reproduce the expected no-extra-two convention.
- Serial and channel-MPI writers are byte-identical after deterministic row
  sorting.
- Source-only mode writes no response file and reports zero solved
  Sternheimer equations.

### 8.2 Reader and algebra gate

- Hand-constructed complex `D`, `Q`, `S`, and `C` reproduce a direct projected
  `A` and Hermitian `Pi` to `1e-13`.
- Independent phase rotations of primitive functions, response rows, and
  source rows leave `L_Pi` invariant when their paired quantities are rotated
  consistently.
- Occupation is counted once, frequency weight is counted once, and no factor
  two is missing from `A + A^dagger`.
- Mismatched blocks, overlap, whitening hash, occupations, or source keys are
  rejected with explicit errors.
- Existing v1-only optimization tests remain numerically unchanged.

### 8.3 Feasibility ranking gate

Generate source-only H and H2 sidecars with the exact archived target inputs,
then evaluate `L_Pi` for:

1. the initial TZDP basis;
2. the previous fixed-DZP joint basis;
3. the low-frequency-guarded basis.

The previous joint and guarded bases must both have lower `L_Pi` than initial
TZDP, consistent with their held-out counterpoise SOS-RPA improvement. Their
mutual order is reported but is not a hard gate because their binding-energy
difference is only about `0.011 kcal/mol`, below the `0.1 kcal/mol` resolution
used for this project. The rank and values must be stable to `1%` under the
three primitive-overlap pseudoinverse thresholds.

If this gate fails, stop. Do not optimize against `L_Pi`; proceed to a separate
Galerkin-Sternheimer design.

### 8.4 Same-size optimization and held-out physics gate

If the ranking gate passes, optimize only the original TZDP `3s,2p` columns
while freezing DZP `1s,2s,1p`. Require:

- fixed columns unchanged to `1e-12`;
- finite, well-conditioned candidate overlap;
- lower `L_Pi` than both initial TZDP and the previous joint basis;
- no regression of DFT, dpsi, or ordinary Sternheimer constraints relative to
  their configured initial limits;
- the same orbital count and radial cutoff as initial TZDP.

Finally run the established all-band, full-Coulomb, 16-frequency SOS-RPA
calculations for H2, H, and H+ghost in the 20-A cell. Report raw binding,
counterpoise binding, and BSSE. The new loss passes its physical gate only if
the counterpoise binding moves toward the converged Delta-ST/FHI-aims reference
by at least `0.1 kcal/mol` relative to the previous joint basis. SOS-RPA energy
and H+ghost data remain held out and are never optimizer inputs.

## 9. Code Boundaries And Commit Stages

ABACUS changes are limited to the molecular SIAB producer path:

- `sternheimer_siab_data.h`: source-row data type;
- new focused source writer/overlap files beside the existing v1 writer;
- `sternheimer_abacus_st_smoke.cpp`: source construction, source-only early
  return, and integration with existing channel ownership;
- ABACUS input declaration and validation files for
  `sternheimer_siab_source_only`;
- focused C++ unit and MPI regression tests.

SIAB changes are limited to:

- a strict source-sidecar reader and pairing validator;
- a focused projected-Pi evaluator separate from
  `sternheimer_spillage.py`;
- physical-family aggregation and the new `pi_dpsi_joint` loss mode;
- synthetic tests, feasibility analysis, and one same-size H campaign input.

Work is committed in independent stages: design, ABACUS source format,
source-only producer, SIAB reader/pairing, projected-Pi algebra, ranking gate,
optimizer integration, and held-out physics result. ABACUS compilation and all
numerical tests run only on `df_dcu` normal nodes. The research TeX records the
formula, provenance, result, and any failed gate after each completed stage.
