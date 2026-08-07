# Finite-AO Galerkin Sternheimer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently certify a differentiable finite-AO Galerkin Sternheimer response evaluator whose matrix output is algebraically identical to a full-virtual-state SOS evaluator.

**Architecture:** A focused PyTorch module validates finite-AO matrices, transforms the generalized problem to a Lowdin-orthonormal basis, solves Sternheimer equations in the orthogonal complement of the occupied subspace, and assembles the auxiliary response. A separate SOS function is retained only as an oracle. No optimizer or ABACUS producer behavior changes in this phase.

**Tech Stack:** Python 3.10, PyTorch 2.1 CPU complex128, `unittest`, df_dcu normal Slurm validation.

---

### Task 1: Freeze the matrix API with RED tests

**Files:**
- Create: `SIAB/tests/test_galerkin_sternheimer.py`
- Create later: `SIAB/opt_orb_pytorch_dpsi/galerkin_sternheimer.py`

- [ ] **Step 1: Write the analytic two-level RED test**

Use `S=I`, `H=diag(-0.5, 0.7)`, one occupied state with occupation 2, and a
Hermitian perturbation with `V_01=V_10=0.3`. At `omega=0.4`, require the
Galerkin result to have shape `(1, 1, 1)`, be Hermitian, and equal the explicit
closed form obtained from

```python
delta = -0.3 / (0.7 - (-0.5) + 0.4j)
expected_half = 2.0 * 0.3 * delta
expected = expected_half + expected_half.conjugate()
```

- [ ] **Step 2: Run the focused test remotely and verify RED**

Stage the tracked SIAB tree on df_dcu and submit one `normal` job with one
node, one process, 30 CPUs, 110610 MiB, and 24 hours. Run:

```bash
python -m unittest -v \
  test_galerkin_sternheimer.GalerkinSternheimerTest.test_two_level_analytic_response
```

Expected: import failure for `galerkin_sternheimer`; no production code exists.

- [ ] **Step 3: Commit the RED test and evidence**

Commit only the test, plan/spec, and immutable RED job evidence.

### Task 2: Implement validated Lowdin preparation

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/galerkin_sternheimer.py`
- Modify: `SIAB/tests/test_galerkin_sternheimer.py`

- [ ] **Step 1: Add RED validation tests**

Add one test per failure: wrong dtype, non-Hermitian `H`, non-Hermitian
`V_mu`, non-positive frequency, no virtual state, non-positive overlap
eigenvalue, and condition number above the configured limit.

- [ ] **Step 2: Run validation tests remotely and verify RED**

Expected: the analytic test and validation tests fail because the public API is
absent.

- [ ] **Step 3: Implement strict input validation and Lowdin transform**

Implement private helpers that:

```python
eigenvalue, eigenvector = torch.linalg.eigh(overlap)
threshold = relative_rank_tolerance * torch.max(eigenvalue)
if torch.any(eigenvalue <= threshold):
    raise RuntimeError("overlap is rank deficient")
x = eigenvector @ torch.diag(eigenvalue.rsqrt()) @ eigenvector.mH
```

Validate the measured overlap condition number before returning transformed
`H` and `V`.

- [ ] **Step 4: Implement the minimal analytic Galerkin solve**

Diagonalize transformed `H`, select positive occupations, form
`P=U_occ@U_occ.mH` and `Q=I-P`, solve
`Q@(Hbar-eps_i*I+1j*omega*I)@Q+P` against all perturbation right-hand sides,
and assemble `A + A.mH`.

- [ ] **Step 5: Run focused tests remotely and verify GREEN**

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 6: Commit the Galerkin implementation**

Use commit message `feat(siab): add finite-AO Galerkin Sternheimer solver`.

### Task 3: Add the independent SOS oracle and equivalence gates

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/galerkin_sternheimer.py`
- Modify: `SIAB/tests/test_galerkin_sternheimer.py`

- [ ] **Step 1: Write the dense-complex Galerkin/SOS RED test**

Construct deterministic dense complex matrices from a fixed seed, form
positive `S=B^H B+I`, Hermitian `H`, and Hermitian perturbations. Use two
occupied and two virtual states, three perturbations, and two frequencies.
Require relative Frobenius error below `1e-11` and maximum absolute error below
`1e-12`.

- [ ] **Step 2: Run the equivalence test remotely and verify RED**

Expected: import failure for `evaluate_sos_response`.

- [ ] **Step 3: Implement explicit finite-basis SOS**

Transform and diagonalize through the same validated preparation, then form

```python
denominator = energy_virtual - energy_i + 1j * omega
delta_virtual = -(u_virtual.mH @ v_mu @ u_i) / denominator
```

for every occupied state and perturbation. Assemble the same `A + A.mH` and
return the same result dataclass.

- [ ] **Step 4: Add Hermiticity and sign/factor assertions**

Check both evaluators independently, not only their difference.

- [ ] **Step 5: Run the focused suite remotely and verify GREEN**

Expected: all Galerkin/SOS tests pass.

- [ ] **Step 6: Commit the SOS oracle**

Use commit message `test(siab): certify Galerkin response against SOS`.

### Task 4: Add invariance and gradient gates

**Files:**
- Modify: `SIAB/tests/test_galerkin_sternheimer.py`

- [ ] **Step 1: Write occupied-gauge and AO-coordinate RED tests**

Use a degenerate occupied two-dimensional block for the gauge test. Apply a
unitary rotation inside that occupied block and require unchanged response.
For the AO-coordinate test, apply an invertible matrix `T` consistently:

```python
S2 = T.mH @ S @ T
H2 = T.mH @ H @ T
V2 = torch.einsum("pa,mpq,qb->mab", T.conj(), V, T)
```

Require the transformed response to match the original response.

- [ ] **Step 2: Run both tests and verify RED for any unsupported behavior**

The AO-coordinate test is expected to expose any incorrect Lowdin back
transformation or occupation ordering.

- [ ] **Step 3: Correct only the behavior exposed by RED**

Do not add optimizer coupling. Keep the response API matrix-only.

- [ ] **Step 4: Add a finite-gradient test**

Make one nondegenerate Hermitian perturbation tensor depend on a scalar
`torch.float64` parameter, evaluate `sum(abs(response)**2)`, call
`backward()`, and require a finite nonzero gradient.

- [ ] **Step 5: Run the focused suite and the existing projected-Pi tests remotely**

Run:

```bash
python -m unittest -v \
  test_galerkin_sternheimer \
  test_projected_pi \
  test_projected_pi_optimization
```

Expected: all tests pass with no warnings or skipped tests.

- [ ] **Step 6: Commit the invariance and gradient gates**

Use commit message `test(siab): harden Galerkin response invariants`.

### Task 5: Independent server certification and documentation

**Files:**
- Modify: `SIAB/example_H_sternheimer/projected_pi_loss/README.md`
- Add: compact immutable logs under `SIAB/example_H_sternheimer/projected_pi_loss/results/`
- Modify externally: `sternheimer_siab_project/main.tex`

- [ ] **Step 1: Create an immutable tracked archive**

Record commit SHA, archive SHA256, Python/PyTorch versions, source hashes,
thread settings, Slurm resources, and exact test command.

- [ ] **Step 2: Run the focused suite twice in one normal allocation**

Require both runs to have exit code zero and byte-identical normalized test
summaries. Runtime-only timestamps may be excluded explicitly.

- [ ] **Step 3: Record the scientific boundary**

State that this closes only the matrix-solver gate. It does not establish AO
completeness, optimize a basis, or provide an H2 RPA binding energy.

- [ ] **Step 4: Compile and inspect the TeX note**

Run `latexmk -g -xelatex -interaction=nonstopmode -halt-on-error main.tex`,
check unresolved references and changed-section overfull boxes, and render the
new pages for visual inspection.

- [ ] **Step 5: Merge into the standing SIAB branch and push**

After all checks pass, merge the staged commits into
`codex/sternheimer-siab-h`, verify Codex author/AroundPeking committer
attribution, and push the branch.

### Task 6: Start the separate ABACUS primitive-exporter plan

**Files:**
- Plan only in this phase: ABACUS branch `codex/sternheimer-siab-producer`

- [ ] **Step 1: Audit the current source-v1 writer**

Locate where primitive grid values, overlap, source projections, auxiliary
whitening, occupations, and MPI reductions are assembled.

- [ ] **Step 2: Write a separate producer design before C++ edits**

Specify versioned serialization for primitive `H^p` and `V^(p,mu)`, size and
MPI ownership, Hermiticity checks, hashes, and the fixed-TZDP SOS comparison.

- [ ] **Step 3: Do not implement the producer until the Python matrix gate is certified**

The producer consumes the certified API and must not redefine its sign,
occupation, or conjugation conventions.
