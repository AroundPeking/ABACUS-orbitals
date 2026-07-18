# H Sternheimer-SIAB Basis Optimization Design

## 1. Goal

Extend SIAB so that an existing H-TZDP basis can be optimized against accurate
Delta-Sternheimer first-order wavefunctions while retaining its ground-state
DFT quality. The new Sternheimer supervision is generated only from an
isolated H atom. The 0.74-A H2 RPA/SOS result is held out from the new loss.

The physics success criterion is that the H2 LCAO-SOS RPA@PBE binding energy
moves to within `0.1 kcal/mol` of the converged Delta-ST/FHI-aims reference,
without a significant regression of the PBE+EXX binding contribution.

## 2. Repository And Baseline

All SIAB changes are made in this repository:

```text
/Users/ghj/同步空间/AITP_project/sternheimer_abacus/ABACUS-orbitals
```

Development branch:

```text
codex/sternheimer-siab-h
```

The implementation starts from upstream commit `3b634f16`. The H basis is the
Dojo-NC-SR 8-au TZDP basis:

```text
Dojo-NC-SR/Orbitals_v2.0/H_TZDP/H_gga_8au_100Ry_3s2p.orb
Dojo-NC-SR/Orbitals_v2.0/H_TZDP/info/8/ORBITAL_RESULTS.txt
```

The orbital file SHA256 is
`7e398340398306a6baf1c61ea68944d81ed43667473fbcc290d6541c4a661d1c`.
The optimized basis contains three s and two p radial functions. The first s
function is SIAB level1 and remains fixed. The variable set is `2s`, `3s`,
`1p`, and `2p`.

## 3. Compared Optimization Modes

### 3.1 ST-only diagnostic

Minimize only the Sternheimer residual spillage:

\[
  \min_C \widehat L_{\mathrm{ST}}(C).
\]

This lane measures the maximum RPA improvement available from the chosen
Sternheimer target and variable orbital set. It is a diagnostic, not the
default production basis, because it does not prevent DFT quality from
degrading away from the initial TZDP optimum.

### 3.2 Constrained ST optimization

The production candidate minimizes the same Sternheimer loss while retaining
the original DFT-wavefunction and dpsi losses as inactive-until-needed
constraints:

\[
\begin{split}
  L_{\mathrm{total}}(C)
  ={}& \widehat L_{\mathrm{ST}}(C) \\
  &+ \lambda_0
  \left[\max\left(0,
  \widehat L_{\mathrm{DFT}}(C)-1-\tau_{\mathrm{DFT}}
  \right)\right]^2 \\
  &+ \lambda_1
  \left[\max\left(0,
  \widehat L_{\mathrm{dpsi}}(C)-1-\tau_{\mathrm{dpsi}}
  \right)\right]^2.
\end{split}
\]

Each normalized loss is

\[
  \widehat L_X(C)
  =
  \frac{L_X(C)}{\max(L_X(C_0),\epsilon)},
\]

where `C0` is the unmodified H-TZDP coefficient set. The initial H experiment
uses `tau_DFT = 0.05` and `tau_dpsi = 0.10`. Penalty strengths are increased
until both constraints are satisfied; they are not interpreted as physical
weights.

### 3.3 Rejected default: unrestricted weighted sum

A fixed sum

\[
  L_{\mathrm{ST}}
  +a L_{\mathrm{DFT}}
  +b L_{\mathrm{dpsi}}
\]

is not the default because `a` and `b` depend on unrelated numerical scales.
It may be retained as an explicit experimental mode, but no validation result
will be selected from it unless it also satisfies the constrained criteria.

## 4. Sternheimer Training Data

### 4.1 Independent reference vectors

For occupied state `i`, Coulomb-orthonormal auxiliary perturbation channel
`a`, and minimax frequency point `p`, ABACUS generates the complete
Delta-Sternheimer first-order wavefunction

\[
  Y_{iap}(\mathbf r)
  =\Delta\psi_{ia}(\mathbf r,i\omega_p).
\]

These vectors are independent SIAB reference wavefunctions. They are not fed
through the existing `linear` input, because that path represents the first
variation of the projection of an existing reference rather than an
additional reference vector.

The auxiliary perturbations are transformed with the same retained
full-Coulomb eigenspace used by LibRPA:

\[
  \widetilde v_a
  =\sum_\mu v_\mu (V_{\mathrm{full}}^{-1/2})_{\mu a}.
\]

ABACUS solves the response to `v_tilde_a` directly. This makes the ST loss
invariant to a nonsingular change of the original ABFS representation and
avoids requiring cross-channel wavefunction Gram matrices in SIAB.

### 4.2 Compact producer interface

Raw grid wavefunctions are not written. For every reference `rho=(i,a,p)`,
the producer writes

\[
  n_\rho=\langle Y_\rho|Y_\rho\rangle_{\mathcal G},
  \qquad
  Q^{\mathrm{ST}}_{\rho e}
  =\langle Y_\rho|B_e\rangle_{\mathcal G},
\]

plus occupation, minimax weight, frequency, occupied-state index, auxiliary
channel index, grid volume element, and the primitive-basis definition. It
also writes or references

\[
  S^B_{ee'}=\langle B_e|B_{e'}\rangle.
\]

The first implementation uses a dedicated `sternheimer` data source in the
SIAB JSON input and a tagged `sternheimer_matrix.dat` producer file. The file
contains a version number and explicit array dimensions. It does not overload
`orb_matrix.1.dat`.

The ABACUS producer is implemented and committed in the active Sternheimer
ABACUS repository. This repository owns only the reader, validation, loss,
optimization, and examples for that interface.

## 5. Removing The Fixed Level1 Space

Let `B` be the primitive spherical-Bessel basis, `C0` the fixed `1s` column,
and `C1` the variable upper-orbital columns. Define

\[
  S_{00}=C_0^\dagger S^B C_0,
  \quad
  S_{01}=C_0^\dagger S^B C_1,
  \quad
  S_{11}=C_1^\dagger S^B C_1.
\]

For each Sternheimer reference row `Qrho`, the target and candidate space are
both projected outside the fixed level1 space without reconstructing a grid
wavefunction:

\[
\begin{split}
  \bar n_\rho
  &=n_\rho
    -(Q_\rho C_0)S_{00}^{-1}(Q_\rho C_0)^\dagger,\\
  \bar q_\rho
  &=Q_\rho C_1
    -(Q_\rho C_0)S_{00}^{-1}S_{01},\\
  \bar S_{11}
  &=S_{11}-S_{10}S_{00}^{-1}S_{01}.
\end{split}
\]

The part represented by the variable upper orbitals is

\[
  p_\rho
  =\bar q_\rho\bar S_{11}^{-1}\bar q_\rho^\dagger.
\]

The normalized ST objective is

\[
  \widehat L_{\mathrm{ST}}
  =
  \frac{
    \sum_\rho f_i w_p(\bar n_\rho-p_\rho)
  }{
    \sum_\rho f_i w_p\bar n_\rho
  }.
\]

This is the trace loss of the weighted response covariance. It does not need
an explicit PCA for the first H experiment. PCA or rank-revealing
orthogonalization may later compress the reference rows without changing this
objective.

`S00` is fixed and precomputed. `Sbar11` is solved through a Hermitian
Cholesky factorization. A non-positive factorization or a condition number
above the configured limit stops the optimization with the offending step
and orbital block in the error message.

## 6. SIAB Changes

### 6.1 Backward-compatible input

Existing inputs containing only `origin` and `linear` retain identical loss
and gradient behavior. New fields are opt-in:

```json
{
  "file_list": {
    "origin": [".../orb_matrix.0.dat"],
    "linear": [[".../orb_matrix.1.dat"]],
    "sternheimer": [".../sternheimer_matrix.dat"]
  },
  "freeze_orbitals": [
    {"element": "H", "l": 0, "zeta": 1}
  ],
  "loss": {
    "mode": "st_constrained",
    "tau_dft": 0.05,
    "tau_dpsi": 0.10,
    "constraint_penalty_dft": 10.0,
    "constraint_penalty_dpsi": 10.0
  }
}
```

The old `opt_C_read` behavior remains supported. The new explicit freeze list
takes precedence only when present and is validated against the actual
`(element,l,zeta)` dimensions.

### 6.2 Loss components and logging

The current combined spillage calculation is split into named components:

```text
dft_origin
dft_dpsi
sternheimer
constraint_dft
constraint_dpsi
total
```

Every optimization log row records all six quantities, the current constraint
penalties, the condition number of each variable overlap block, and whether
the step is the best accepted constrained point. `ORBITAL_RESULTS.txt` records
the mode and baseline-normalized final components.

### 6.3 Gradient and freeze behavior

The ST loss uses PyTorch complex tensors and automatic differentiation. A
finite-difference test checks selected real coefficient directions. After
`backward()`, the exact `1s` coefficient column is zeroed by the explicit
freeze mask. The initial and final fixed columns must be bitwise identical.

## 7. H Training And H2 Validation

### 7.1 Training

The H atom is the only ST training system. It uses the exact pseudopotential,
primitive basis, H-TZDP reference, auxiliary basis, full-Coulomb eigenspace,
minimax frequencies, and spin convention used by the validated H/H2
Delta-ST calculation.

The constrained lane reuses the original DFT-origin and dpsi training set that
created this TZDP basis. That historical set contains H dimers and trimers, so
the experiment must not be described as a basis that has never seen molecular
ground-state data. The actual transfer test is narrower and explicit: no H2
Sternheimer vector, RPA correlation energy, or 0.74-A binding energy enters
the new optimization objective.

The historical `orb_matrix.0.dat` and `orb_matrix.1.dat` files are not stored
in this repository. Before the constrained run, they are regenerated from the
checked-in 8-au `SIAB_INPUT` structures with the same Dojo-NC-SR
pseudopotential and a version-recorded ABACUS executable. The regenerated
baseline must reproduce the checked-in `ORBITAL_RESULTS.txt` loss before it is
used as a constraint.

Two runs start from the same `C0`:

1. `st_only`;
2. `st_constrained` with the DFT-origin and dpsi constraints active.

The ST-only lane sees no molecular data. The constrained lane sees only the
historical DFT-origin/dpsi regularizer described above; it does not see the H2
RPA validation observable.

### 7.2 Engineering acceptance

The implementation is accepted when:

1. all legacy SIAB unit fixtures reproduce their previous loss and gradients;
2. the ST loss finite-difference gradient test passes;
3. the fixed `1s` coefficients remain bitwise unchanged;
4. both optimization modes lower `L_ST` from the common initial point;
5. the constrained mode satisfies both configured loss constraints;
6. all generated data report dimensions, kernel, frequency grid, PP, orbital,
   and source commit provenance.

### 7.3 Physics comparison

For the initial TZDP, ST-only result, and constrained result, run the same H
and H2 workflow and report the DFT-origin loss, dpsi loss, ST loss, H and H2
PBE+EXX energies, H and H2 RPA correlation energies, H2 SOS RPA@PBE binding
energy, and its difference from Delta-ST.

These cells are populated only from completed calculations. The preferred
basis is the constrained result if it reaches the `0.1 kcal/mol` RPA target.
If only ST-only reaches the target, the result is diagnostic evidence that the
fixed DFT tolerances or variable orbital set must be revised; it is not
automatically promoted as the production basis.

## 8. Commit Boundaries

Each stage is committed separately:

1. design document only;
2. versioned Sternheimer reader, data model, and parser tests;
3. level1-projected ST loss and finite-difference gradient tests;
4. explicit orbital freeze mask, named loss components, and constrained mode;
5. H-TZDP-8au example inputs and deterministic optimization smoke test;
6. H/H2 production results, comparison table, and documentation.

No commit combines interface work, optimizer behavior, and physics results.
ABACUS producer changes are committed separately in the Sternheimer ABACUS
repository before stage 5 begins.

## 9. Non-goals Of The First Experiment

- No H2 Sternheimer vector or RPA/binding-energy target is used for training;
  the constrained lane may reuse the historical ground-state dimer data.
- The orbital count and 8-au cutoff are not changed.
- The fixed `1s` level1 orbital is not reoptimized.
- The first experiment does not optimize derivatives of Sternheimer
  wavefunctions with respect to atomic displacement.
- The first experiment does not require PCA of the ST references.
- No periodic-solid or multi-element training path is added before the H/H2
  held-out comparison is complete.
