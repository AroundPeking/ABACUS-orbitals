# SIAB RPA-Sensitive Response Loss Design

## 1. Status and decision

This design starts from the selected H `3s2p2d1f1g` basis and keeps its size
fixed at 35 atomic orbitals per H. The fixed DZP columns remain `1s,2s,1p`;
only `3s,2p,2d,1f,1g` may change. No new `f`, `g`, or `h` shell is added in
this cycle.

The design has two distinct uses of counterpoise (CP):

- CP is an offline basis-development and release-certification test.
- A released basis is used in production through the uncorrected, raw
  all-band SOS-RPA calculation. Production does not run H+ghost or apply a CP
  correction.

The selected basis is not yet a released production basis. Its equilibrium
H2 values are

| basis | raw binding | CP binding | BSSE |
|---|---:|---:|---:|
| selected `3s2p2d1f1g` | 108.958806 | 107.888474 | 1.070332 |

in kcal/mol. The second-g candidate is rejected even though its raw value is
108.728861 kcal/mol, because its CP value is only 107.713459 kcal/mol and its
1.015402 kcal/mol BSSE shows that the apparent agreement is an error
cancellation. This is the direct reason not to optimize or select a basis
from the raw scalar binding energy alone.

## 2. Goal and scope

The immediate goal is to retrain the same `3s2p2d1f1g` radial space with a
response loss that ranks basis changes according to their importance for the
full-Coulomb RPA correlation energy. The loss must also make the isolated H
response no worse than the selected basis, so the molecular calculation does
not improve by borrowing the neighboring atom's basis.

The first cycle reuses the validated H and 0.74085-Angstrom H2 source-v1 and
response-v1 data. H+ghost remains outside the optimizer. New bond-length
targets are generated only after the revised loss passes the historical
ranking test and produces an equilibrium candidate that improves both CP
binding and BSSE.

This cycle does not:

- change ABACUS Sternheimer or Delta-Sternheimer equations;
- change the full-Coulomb whitening convention or GreenX frequency grid;
- change `exx_pca_threshold=1e-4`;
- add direct LibRPA energies or ghost data to the training gradient;
- add more AO shells before the fixed-size loss redesign is tested.

## 3. Why the current projected-Pi loss is insufficient

For physical family `F` and imaginary frequency `omega_p`, the current code
constructs a candidate symmetrized polarizability `Pi_Fp(C)` and its
primitive-space reference `Pi_Fp(ref)`. Define

```text
Delta_Pi_Fp(C) = Pi_Fp(C) - Pi_Fp(ref).
```

The existing family loss is

```text
L_base,F = sum_p w_p ||Delta_Pi_Fp||_F^2
           / sum_p w_p ||Pi_Fp(ref)||_F^2.
```

It preserves the full auxiliary-channel matrix, but after the family
normalization it treats all retained matrix directions according to their
plain Frobenius amplitude. The second-f and second-g controls lowered this
training objective without improving the independently computed RPAc-CP
binding. Therefore the missing information is not simply another AO shell;
the loss does not adequately distinguish response directions with different
RPA-energy sensitivity.

## 4. RPA-energy sensitivity

For one frequency, the RPA integrand in the existing full-Coulomb,
symmetrized representation is

```text
F(Pi) = Tr[log(I - Pi) + Pi].
```

Its first variation at the primitive reference is

```text
delta F = Tr[G delta Pi],
G = I - (I - Pi_ref)^(-1).
```

Thus a matrix direction with a large component in `G` changes the RPA
integrand more strongly than an equally large Frobenius error in a weak
screening direction. The direct scalar `delta F` is not a safe loss: positive
and negative errors can cancel between channels or frequencies. The design
therefore uses `G` to weight a positive matrix norm and reports the exact
trace-log difference only as a diagnostic.

Diagonalize the Hermitian primitive reference at every family and frequency,

```text
Pi_ref = U diag(lambda_a) U^H,
g_a = abs(1 - 1 / (1 - lambda_a)).
```

The input is accepted only if `I-Pi_ref` is positive definite within the
existing numerical threshold. Define a dimensionless positive sensitivity
matrix

```text
W = U diag(g_a / max_b(g_b)) U^H
```

when `max(g)>0`. A zero-sensitivity reference is rejected because it cannot
define an RPA-sensitive metric. The weighted matrix error is

```text
L_sens,F = sum_p w_p ||W_Fp^(1/2) Delta_Pi_Fp W_Fp^(1/2)||_F^2
           / sum_p w_p ||W_Fp^(1/2) Pi_Fp(ref) W_Fp^(1/2)||_F^2.
```

The original full-matrix loss is retained to prevent the optimizer from
discarding weak channels that have small first-order energy sensitivity:

```text
L_resp,F(alpha) = alpha L_base,F + (1-alpha) L_sens,F,
0 <= alpha <= 1.
```

`alpha` is not chosen from the new candidate's RPA energy. It is selected
once in the historical ranking gate described below and then frozen before
new optimization begins.

For diagnosis, the code also reports

```text
Delta_F_Fp = Tr[log(I-Pi_Fp(C)) + Pi_Fp(C)]
             - Tr[log(I-Pi_Fp(ref)) + Pi_Fp(ref)]
```

for every frequency. This quantity is not summed into the training loss,
because summing it would reintroduce cancellation.

## 5. Historical ranking gate before optimization

Before changing any orbital, evaluate `L_base`, `L_sens`, `L_resp`, and
`Delta_F` for these archived, independently tested candidates:

1. `3s2p2d`;
2. `3s2p2d1f`;
3. selected `3s2p2d1f1g`;
4. rejected `3s2p2d2f1g`;
5. rejected `3s2p2d1f2g`.

Scan the small frozen set

```text
alpha in {0.0, 0.1, 0.25, 0.5, 1.0}.
```

An `alpha` is admissible only if the response metric reproduces all four
independent physical decisions:

```text
3s2p2d1f       improves over 3s2p2d,
3s2p2d1f1g     improves over 3s2p2d1f,
3s2p2d2f1g     does not improve over selected 3s2p2d1f1g,
3s2p2d1f2g     does not improve over selected 3s2p2d1f1g.
```

If more than one `alpha` passes, choose the largest `alpha`, because it
retains the most of the already validated complete-matrix objective. If none
passes, stop. In that case the source-projected response is not sufficient
to guide this basis, and the next method must solve the candidate-space
Galerkin Sternheimer equation or use a richer response target. It is not
permitted to tune `alpha` against the second decimal place of the CP energy.

This gate uses old CP results only to validate the definition of a reusable
response metric. The new optimized basis remains unseen until `alpha` is
frozen.

## 6. Atomic self-completeness and training objective

Let `C0` be the selected `3s2p2d1f1g` coefficients and let
`L_resp,H(C0)` and `L_resp,H2(C0)` be the frozen baselines. The response
objective uses a fourth-order family norm,

```text
L_family(C) = (L_resp,H(C)^4 + L_resp,H2(C)^4)^(1/4).
```

Compared with an equal sum, this gives the larger normalized family error
the larger gradient without introducing an adjustable H/H2 family weight.
The isolated-atom gate is explicit:

```text
L_resp,H(C_final) < L_resp,H(C0).
```

Therefore an H2 improvement cannot be accepted by sacrificing the isolated
H response. No ghost center is needed in the gradient.

The full objective retains the current DFT/dpsi protection:

```text
L_total(C) = L_family(C)
             + 0.02 L_dpsi(C) / L_dpsi(C0)
             + 10 max(0, L_DFT(C)/L_DFT(C0) - 1.05)^2
             + 10 max(0, L_dpsi(C)/L_dpsi(C0) - 1.10)^2.
```

The `1s,2s,1p` coefficient columns remain bitwise fixed. There is no radial
tail penalty: diffuse radial response is physical information and may not be
suppressed merely to make the orbitals look compact.

The selected checkpoint must satisfy all of the following before any SOS job
is submitted:

```text
finite loss and finite gradients,
candidate overlap condition number <= 1e12,
fixed-DZP maximum coefficient difference = 0,
L_family(C_final) < L_family(C0),
L_resp,H(C_final) < L_resp,H(C0),
L_DFT(C_final) / L_DFT(C0) <= 1.05,
L_dpsi(C_final) / L_dpsi(C0) <= 1.10.
```

## 7. Independent physical gates

### 7.1 Iteration gate at the equilibrium geometry

The first new candidate uses H2, H, and H+ghost only in independent
postprocessing. Every case removes explicit `ABFS_ORBITAL` and regenerates
its own auxiliary basis with `exx_pca_threshold=1e-4`. The fixed contract is:

```text
20-Angstrom cubic cell,
H2 bond length 0.74085 Angstrom,
100 Ry,
16 fixed minimax frequencies,
all AO bands,
rpa_ccp_rmesh_times=5,
Massidda singularity correction,
LibRPA full Coulomb.
```

Relative to the selected basis, a development candidate is promoted only if

```text
D_CP(new) > 107.888474 kcal/mol,
BSSE(new) < 1.070332 kcal/mol,
abs(D_raw(new) - D_Delta-ST) is smaller than for the selected basis.
```

Raw binding alone cannot promote a candidate.

### 7.2 Release gate for production without CP

A basis is released for ordinary raw SOS-RPA only after testing H2 at bond
lengths

```text
0.60, 0.74085, 1.00, and 1.50 Angstrom.
```

At every geometry, under the same numerical contract, it must satisfy

```text
abs(D_raw - D_CP) < 0.1 kcal/mol,
abs(D_raw - D_Delta-ST) < 0.1 kcal/mol.
```

Adjacent Ecut and frequency-grid changes must also alter the reported binding
by less than 0.1 kcal/mol before a basis-error conclusion is made. Once these
release checks pass, the basis file and its provenance are frozen. Subsequent
production calculations use only the raw SOS-RPA path; CP is no longer part
of the user workflow.

## 8. Code boundaries and interfaces

The implementation is confined to SIAB and its H example:

- `projected_pi.py` owns the per-family RPA-sensitivity algebra and exact
  frequency diagnostics.
- `projected_pi_optimization.py` owns H/H2 aggregation and exposes both
  family losses to checkpoint selection.
- `optimization_loss.py` validates the new loss configuration without
  changing legacy modes.
- `opt_orbital_converge.py` logs and selects checkpoints using the atomic and
  family gates.
- a new analysis script evaluates all archived candidates before training.
- a new fixed-size input starts from the selected first-g coefficients.

The public result objects must expose, for each family and frequency:

```text
frequency, GreenX weight, base loss, sensitivity loss,
blended response loss, exact trace-log difference,
minimum eigenvalue of I-Pi_candidate,
maximum candidate-overlap condition number.
```

Invalid non-Hermitian response, a nonpositive `I-Pi`, a zero sensitivity
denominator, non-finite eigenvalues, or mismatched H/H2 frequency grids is a
fatal error with an explicit message. The implementation must not clip an
invalid response silently.

## 9. Test strategy

Implementation follows test-driven development.

1. A diagonal two-channel synthetic case verifies `g_a`, `W`, the weighted
   norm, and the exact trace-log difference analytically.
2. A nondiagonal complex Hermitian case verifies unitary covariance and
   gradients against centered finite differences.
3. Tests prove that channel permutation and common source/response phase do
   not change any loss.
4. Tests reject `lambda_max(Pi) >= 1`, a zero sensitivity denominator,
   non-Hermitian matrices, and non-finite inputs.
5. Existing projected-Pi and all legacy SIAB tests remain numerically
   unchanged when the new mode is not selected.
6. The archived five-basis analysis must pass the frozen ordering gate before
   an optimizer job can be submitted.
7. The remote optimizer gate records source commit, executable and input
   hashes, all loss components, fixed-column differences, condition number,
   resources, job ID, and wall time.
8. The independent SOS/CP gate records AO and auxiliary dimensions, all-band
   counts, Coulomb hashes, H2/H/H+ghost energies, raw/CP/BSSE decomposition,
   resources, and elapsed time.

## 10. Documentation and commit sequence

Every stage updates both the example README and the project TeX note. A stage
is not described as successful until its own test or numerical gate passes.
The commits are separated as follows:

1. design, equations, and frozen validation contract;
2. RED tests for sensitivity algebra and configuration;
3. minimal sensitivity implementation and local unit-test gate;
4. archived-candidate ranking analysis and frozen `alpha` decision;
5. fixed-size `3s2p2d1f1g` optimizer input and training result;
6. independent equilibrium SOS/CP result;
7. multi-geometry release certification, only if the equilibrium iteration
   gate passes.

Proposed work, queued jobs, completed solver output, and passed physical gates
must remain separate statuses in both documents.
