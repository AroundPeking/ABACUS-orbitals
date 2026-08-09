# SIAB Delta-ST Response Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize a compact fixed-DZP-plus-variable LCAO basis, parameterized by spherical-Bessel coefficients, against the immutable uniform-grid Delta-ST symmetric response.

**Architecture:** Reuse the exact frozen-occupied Galerkin solver and full-Coulomb transform. Assemble candidate AO columns from SIAB radial coefficient blocks, compute the compact response, and expose its weighted `Pi` error through the existing projected-Pi optimizer contract. Keep the Bessel correction parent out of the candidate response so the loss measures what the final LCAO/SOS basis itself captures.

**Tech Stack:** Python 3, PyTorch complex128/float64 autograd, SIAB coefficient/freeze infrastructure, `unittest`, server66, XeLaTeX.

---

### Task 1: Freeze the design

**Files:**
- Add: `docs/superpowers/specs/2026-08-09-siab-delta-st-response-compression-design.md`
- Add: `docs/superpowers/plans/2026-08-09-siab-delta-st-response-compression.md`

- [ ] Record the fixed occupied reference, Bessel contraction, compact Galerkin equation, full-Coulomb loss, freeze contract, and real H gate.
- [ ] Check the documents for placeholders, contradictory candidate/reference definitions, and accidental scalar-energy optimization.
- [ ] Commit and push with Codex author and AroundPeking committer.

### Task 2: RED radial assembly and differentiable response tests

**Files:**
- Modify: `SIAB/tests/test_sternheimer_spillage.py`
- Add: `SIAB/tests/test_delta_st_response_compression.py`

- [ ] Add a test that assembles one radial coefficient matrix across complete `m=-l,...,l` primitive blocks using `SternheimerPrimitiveGalerkinData`.
- [ ] Run it and require failure because the existing assembler assumes response-v1 `q` storage.
- [ ] Add a three-level response fixture and a test requiring a variable compact orbital to have finite nonzero gradient toward a missing response direction.
- [ ] Run it and require failure because the compression evaluator does not exist.
- [ ] Add a test that one small gradient step lowers the weighted full-frequency `Pi` loss.
- [ ] Add a test that a changed primitive-to-AO Hamiltonian cross block changes the loss or gradient.

### Task 3: GREEN compact Delta-ST evaluator

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/sternheimer_spillage.py`
- Add: `SIAB/opt_orb_pytorch_dpsi/delta_st_response_compression.py`

- [ ] Generalize primitive dimension/device discovery in `assemble_orbital_coefficients` without changing existing response-v1 ordering.
- [ ] Implement protocol validation by reusing `validate_parent_space_protocol` and the exact frozen-occupied solver.
- [ ] Cache `Pi_ref`, full-Coulomb metadata, frequency weights, and the weighted reference norm at construction.
- [ ] Implement expanded-matrix evaluation and SIAB radial-mapping evaluation.
- [ ] Return `ProjectedPiOptimizationResult` with per-frequency losses and candidate condition diagnostics.
- [ ] Run the new focused tests and the existing spillage, frozen-occupied, parent-space, projected-Pi, and loss/freeze suites.
- [ ] Commit and push after the focused and regression tests pass on server66.

### Task 4: RED/GREEN optimizer freeze integration

**Files:**
- Modify: `SIAB/tests/test_loss_and_freeze.py`
- Modify only if the test exposes an incompatibility: `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py`

- [ ] Add an integration test using `pi_dpsi_joint`, the new evaluator, and `freeze_orbitals=[1s,2s,1p]`.
- [ ] Require the variable column to move, the response loss to decrease, and every fixed column to remain bitwise equal.
- [ ] Run the test red; implement only the minimal adapter needed for green.
- [ ] Re-run all loss/freeze tests and commit separately if production code changes.

### Task 5: H real-data gradient gate on server66

**Files:**
- Add: `SIAB/example_H_sternheimer/delta_st_response_compression/run_h_gradient_gate.py`
- Add: `SIAB/example_H_sternheimer/delta_st_response_compression/README.md`
- Add remotely: immutable result directory under `/home/ghj/sternheimer_abacus_tests/`

- [ ] Load the accepted 100-Ry primitive sidecar, fixed-AO sidecar, grid Delta-ST reference, and full Coulomb matrix; verify protocol before any optimization.
- [ ] Load the original H `3s2p` Bessel coefficients and declare `1s,2s,1p` fixed, `3s,2p` variable.
- [ ] Evaluate the initial full-frequency `Pi` loss, maximum per-frequency loss, overlap condition, retained rank, and RPA-energy diagnostic.
- [ ] Backpropagate once, zero fixed gradients through the existing freeze helper, and require finite nonzero variable gradients and exact zero fixed gradients.
- [ ] Perform a bounded line search and accept the first step that lowers `Pi` loss while leaving fixed columns bitwise equal.
- [ ] Archive JSON diagnostics, coefficient hashes, input hashes, source revision, runtime, and memory.

### Task 6: Record the result

**Files:**
- Modify: `/Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project/sections/siab_primitive_galerkin_gate.tex`
- Add: compact H gradient-gate data under `/Users/ghj/同步空间/AITP_project/sternheimer_abacus/sternheimer_siab_project/data/`

- [ ] Add the Bessel-to-LCAO contraction formula and explain why the candidate loss excludes a Bessel correction parent.
- [ ] Record red/green commits, server66 tests, H protocol, initial loss, gradient gate, accepted step, and unresolved optimization work.
- [ ] Compile with XeLaTeX, check unresolved references and changed-section overfull boxes, render changed pages, and inspect them visually.
- [ ] Verify commit attribution and remote branch head before reporting completion.
