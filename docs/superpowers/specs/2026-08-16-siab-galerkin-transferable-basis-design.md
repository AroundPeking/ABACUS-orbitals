# SIAB Galerkin-RPA Transferable Basis Design

## 1. Status and decision

The previous source-projected RPA-sensitivity route is closed.  The formal
five-basis campaign at commit `3fafebb5` found no admissible blend among
`alpha={0,0.1,0.25,0.5,1}`.  Small alpha values failed to rank the first `f`
and `g` additions as improvements, while larger alpha values incorrectly
ranked the independently rejected second `f` and second `g` as further
improvements.  No production alpha or new optimized orbital was created.

This is a method stop, not evidence that the current H basis has reached its
accuracy limit.  The next route must evaluate the response that the candidate
AO space actually generates through a Galerkin Sternheimer solve.  It must not
continue optimizing a projection of a fixed Delta-ST response.

The current same-protocol five-point reference is:

| response basis | RMSE | MAE | maximum error |
|---|---:|---:|---:|
| response-only `3s4p3d` | 0.7757 | 0.6715 | 1.2880 |
| original TZDP | 2.4229 | 2.1771 | 3.6591 |

All values are kcal/mol relative to the 25-Ry, 20-Angstrom, 16-frequency,
PCA-`1e-4` Delta-ST curve.  The `0.7757` value is the best completed
same-protocol five-point result, not a lower bound.

## 2. Goal

Develop the smallest H response basis that simultaneously satisfies:

1. the uncorrected, raw H2 RPA binding curve agrees with same-protocol
   Delta-ST within `0.1 kcal/mol` at every release geometry;
2. the raw result is not made accurate by counterpoise error cancellation;
3. the isolated H response and the original DFT occupied space are not
   degraded;
4. a periodic H2 lattice scan remains accurate when intermolecular overlap is
   changed; and
5. production calculations use one ordinary raw SOS-RPA calculation, without
   a ghost calculation or a CP correction.

The optimization method, data contracts, and acceptance gates must later map
to periodic `(k,q,iomega)` response data.  This design implements and validates
only the molecular H/H2 cycle first.

## 3. Options considered

### 3.1 Direct fitting of the raw H2 binding energy

This can make one bond length agree with Delta-ST, but the historical
`3s2p2d1f1g` result demonstrates the failure mode:

| basis | raw binding | CP binding | BSSE |
|---|---:|---:|---:|
| `3s2p2d1f1g` | 108.958806 | 107.888474 | 1.070332 |

The apparent raw agreement contains substantial basis borrowing.  A scalar
raw-energy fit is therefore rejected as the training objective.

### 3.2 Penalizing all ghost borrowing in the gradient

This would reduce a CP diagnostic, but it can also suppress physical
multicenter virtual response.  Ghost data is therefore not a training family.
It is retained only as a development-time error-cancellation veto.

### 3.3 Candidate-space Galerkin Sternheimer optimization

This is the selected route.  It uses the candidate basis to construct its own
Hamiltonian, denominators, perturbed wavefunctions, response matrix, and RPA
integrand.  Multiple geometries and an isolated-atom family prevent a single
H2 geometry from dominating the optimization.

## 4. Candidate capacity, coefficient parent, and released basis

The first candidate-capacity test allows the radial layout

```text
3s4p3d1f1g = 46 AO per H.
```

The original DZP coefficient columns `1s,2s,1p` remain exactly fixed.  They
contribute `2+3=5` AO per H; the remaining `41` AO in the 46-AO capacity are
the optimizable response shells.  Thus 46 AO is the total candidate basis, not
46 response AO added to DZP or TZDP.  The 46-AO layout is a candidate capacity,
not the release size.  The release basis is obtained by deleting the least
important radial shell, reoptimizing the
remaining coefficients, and repeating until the next deletion fails a
physical gate.  Angular degeneracies are never split: one radial `l` shell is
added or removed with all `2l+1` magnetic components.

The coefficient parent used to shape those candidate orbitals is distinct
from the 46-AO candidate layout and is not the full 25-radial-function Bessel tensor.
For each `l`, SIAB constructs a compact, metric-orthonormal parent from:

1. every radial function in the starting basis;
2. the leading H/H2 Delta-ST residual covariance modes after the fixed DZP
   space is removed; and
3. enough additional positive-metric modes to pass the parent-convergence
   gate.

The parent is accepted only when adding one further positive-metric radial
mode in every available `l=0,...,4` channel changes the equilibrium Galerkin
RPA binding by less than `0.02 kcal/mol` and changes every normalized family
loss by less than `1%`.  This makes the profile small enough to store while
retaining an explicit parent-space convergence test.

## 5. Candidate-space Galerkin equations

Let `A_DZP` be the fixed DZP block inside the candidate basis and `P` the
compact response parent.  Candidate response orbitals are `R(C)=P C`, where
the same radial coefficient column is shared by all magnetic components of one
shell.  The total candidate basis is

```text
U(C) = [A_DZP, R(C)],
T(C) = diag(I_DZP, C).
```

ABACUS evaluates coefficient-independent parent matrices in the combined
parent `B=[A_DZP,P]`, using the frozen original-TZDP zero-order Hamiltonian.
SIAB contracts them as

```text
S(C)    = T(C)^H S_parent T(C),
H(C)    = T(C)^H H_parent T(C),
V_mu(C) = T(C)^H V_mu,parent T(C).
```

The reference occupied functions `psi_i^0` and eigenvalues `eps_i^0` are
obtained only from the original TZDP problem and normalized in the common grid
metric.  For each candidate, their coefficient vectors `c_i(C)` are the
metric-least-squares projections into `U(C)`.  The relative grid-metric
projection residual

```text
eta_i(C) = ||psi_i^0-U(C)c_i(C)||_W / ||psi_i^0||_W
```

must not exceed `1e-6`; otherwise the candidate is invalid.  After
S-orthonormalizing the projected occupied vectors, define their candidate
S-metric complement as `Q_C`.  Candidate response orbitals therefore improve
the virtual response without silently redefining the zero-order occupied
reference.  For every occupied state, auxiliary perturbation `mu`, and
imaginary frequency `omega_p`, solve

```text
Q_C^H [H(C) - eps_i^0 S(C) + i omega_p S(C)] Q_C x_muip
      = -Q_C^H V_mu(C) c_i(C).
```

The conjugate-frequency branch is included explicitly when forming
`M_mu,nu(iomega_p)`.  With the existing full-Coulomb contract,

```text
Pi_C = V_full^(-1/2) M_C V_full^(-1/2),
F_C  = Tr[log(I-Pi_C) + Pi_C].
```

The implementation must verify the overlap rank, overlap condition, occupied
orthogonality, Hermiticity, negative-semidefinite response, and positive
definiteness of `I-Pi_C`.  It may not repair an invalid candidate by silently
clipping eigenvalues.

## 6. Training loss without a fitted scalar energy

For physical family `X` and frequency `p`, define normalized matrix and
integrand errors against the same-profile Delta-ST reference:

```text
e_Pi,X = sum_p w_p ||Pi_C,Xp-Pi_Delta,Xp||_F^2
         / sum_p w_p ||Pi_Delta,Xp||_F^2,

e_F,X  = sum_p w_p |F_C,Xp-F_Delta,Xp|^2
         / sum_p w_p |F_Delta,Xp|^2.
```

The two errors are normalized by their values at the starting `3s4p3d`
basis.  A fourth-order norm prevents either target from being hidden:

```text
l_X = [(e_Pi,X/e_Pi,X0)^4 + (e_F,X/e_F,X0)^4]^(1/4),
L_G = [sum_X l_X^4]^(1/4).
```

Checkpoint selection minimizes `L_G` only among candidates satisfying all of
these hard constraints:

```text
fixed 1s,2s,1p coefficient difference = 0,
maximum fixed-TZDP occupied projection residual <= 1e-6,
DFT loss / initial DFT loss <= 1.05,
dpsi loss / initial dpsi loss <= 1.10,
overlap condition <= 1e12,
isolated-H e_Pi and e_F both improve over the starting basis.
```

There is no radial-tail penalty and no ghost term in the gradient.  Frequency
errors are squared before quadrature, so positive and negative trace-log
errors at different frequencies cannot cancel during training.

## 7. Physical families and geometry split

The molecular cycle uses the existing 20-Angstrom box, 16 fixed GreenX
minimax frequencies, full Coulomb, Massidda treatment, all AO bands for SOS,
and `exx_pca_threshold=1e-4`.

Training families are:

```text
isolated H,
H2 at R = 0.60, 0.74085, and 1.20 Angstrom.
```

Held-out molecular gates are:

```text
H2 at R = 0.90 and 1.80 Angstrom.
```

The split includes a compressed bond, equilibrium, and a stretched bond in
training; the held-out geometries test interpolation and the approach to
dissociation.  No held-out energy or ghost result is read by the optimizer.

## 8. Auxiliary-basis contract

The differentiable inner optimization needs a fixed matrix dimension.  Its
common auxiliary space is generated once from the unoptimized 46-AO capacity
basis using
`exx_pca_threshold=1e-4`; the matching Delta-ST references are regenerated in
that same auxiliary space.  No manually chosen auxiliary threshold or second
auxiliary basis is introduced.

Every independent physical promotion and release calculation removes an
explicit `ABFS_ORBITAL` entry and regenerates the tested candidate's own
auxiliary basis with exactly `exx_pca_threshold=1e-4`.  Candidate SOS and its
Delta-ST comparator must use the same regenerated auxiliary basis, ordering,
full-Coulomb matrix, and frequency grid.  Thus the common parent auxiliary
space is an optimization coordinate system, not the production contract.

## 9. Staged execution and stop gates

### Stage A: existing-candidate response-only screen

At `R=0.74085 Angstrom`, run the current fixed-TZDP response-only Galerkin
producer for:

1. the accepted `3s4p3d` baseline;
2. the archived `3s2p2d1f1g` orbital; and
3. the unoptimized 46-AO `3s4p3d1f1g` capacity basis.

A new candidate advances to a full curve only if its absolute equilibrium
error improves by at least `0.1 kcal/mol` over the baseline `0.9096 kcal/mol`
error.  Otherwise the result is retained as a capacity diagnostic.

### Stage B: parent convergence and differentiable Galerkin gate

Generate the compact parent profiles for H and the three training geometries.
Verify direct complex solves against a spectral generalized-eigenvalue oracle
on a small fixture and on one real frequency.  Require matrix relative error
below `1e-10` and trace-log absolute error below `1e-10` before optimization.

The fixed-occupied optimization representation also needs an explicit bridge
to the ordinary production calculation.  For every promoted checkpoint:

1. regenerate the candidate PCA-`1e-4` auxiliary basis;
2. require the fixed-TZDP occupied projection residual in the candidate basis
   to remain below `1e-6`;
3. compare the direct Galerkin solve and the spectral resolvent expansion using
   the same projected fixed-TZDP occupied states, candidate AO space, auxiliary
   ordering, and Coulomb matrix, requiring relative matrix error below `1e-10`;
4. run the ordinary candidate-basis SCF and require its occupied-projector
   distance from the fixed-TZDP occupied projector to be below `1e-6`; and
5. require the candidate-basis zero-order H2 binding contribution to differ
   from the fixed-TZDP value by less than `0.05 kcal/mol` at every release
   geometry.

Failure of either occupied-space gate means that the response-only optimizer
is not predicting the ordinary one-shot basis calculation.  Such a candidate
cannot advance by compensating its changed DFT reference with its RPA error.

### Stage C: 46-AO optimization

Optimize the unfrozen radial coefficients, record every family/frequency loss,
and publish only checkpoints satisfying the DFT, dpsi, isolated-H, overlap,
and dielectric gates.  If no checkpoint improves both isolated-H response
errors and total `L_G`, stop without SOS promotion.

### Stage D: raw molecular release gate

For all five bond lengths, regenerate the candidate PCA-`1e-4` auxiliary basis
and run an ordinary candidate-basis SCF, independent all-band SOS-RPA, and
same-auxiliary-space Delta-ST.  `D_raw` is the uncorrected total binding from
that ordinary SCF plus SOS-RPA chain.  The candidate passes only if

```text
max_R |D_raw(R)-D_Delta(R)| < 0.1 kcal/mol,
five-point RMSE < 0.1 kcal/mol.
```

Development-only H+ghost jobs are then run at the same five distances.  They
do not modify the reported production result.  They veto release if

```text
max_R |D_raw(R)-D_CP(R)| >= 0.1 kcal/mol.
```

Therefore raw agreement accompanied by a large CP correction is classified
as error cancellation, not success.

### Stage E: compression

Rank removable radial shells by the increase in `L_G` per removed AO, remove
one shell, and reoptimize.  A smaller basis replaces the current candidate
only after repeating the complete Stage-D gate.  Compression stops at the
last basis that passes every raw, Delta-ST, and development-only CP criterion.

### Stage F: periodic transfer gate

Before applying the method to another element, test the released H basis in an
insulating periodic H2 crystal at lattice scale factors `0.90`, `1.00`, and
`1.10`.  Use matched Delta-ST and raw SOS settings at every scale.  The basis
is transferable to the first solid pilot only if the raw-minus-Delta-ST energy
difference changes by less than `1 meV/H2` across the three scales and the
location of the discrete minimum is unchanged.  No molecular CP correction is
defined or applied in this periodic gate.

## 10. Provenance and reporting

Every stage records source commit, executable hash, orbital and pseudopotential
hashes, parent/profile hashes, auxiliary dimension and ordering, Coulomb hash,
frequency grid, Ecut and actual grid, solver tolerances, rank and condition
diagnostics, wall time, memory, and scheduler identity.  A completed solver or
rendered plot is not a physical pass until the corresponding numerical gate is
evaluated.

The research note must distinguish proposal, code test, running calculation,
completed numerical result, and physical promotion.  Failed candidates and
their inputs remain archived.

## 11. Explicit non-goals for the first implementation

- No periodic `(k,q)` Galerkin optimizer is implemented before the molecular
  H/H2 release gate passes.
- No direct fitting to a single total binding energy is allowed.
- No CP correction is part of a released production workflow.
- No basis larger than the 46-AO search layout is promoted without a separate
  design review.
- No claim of transferability is made from the H2 molecule alone.
