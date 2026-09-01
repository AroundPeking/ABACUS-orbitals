# Reusable Galerkin Binding-Basis Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one reusable Galerkin candidate-generation and physical-promotion workflow for H/H2 and C atom/diamond, then use the C adapter to generate the next controlled candidate.

**Architecture:** Add a tested tangent-gradient candidate-bank core beside the periodic Galerkin optimizer and a pure manifest state evaluator beside it. Keep all host paths and physical thresholds in a C adapter JSON plus a thin df runner, so the core contains no material- or cluster-specific values.

**Tech Stack:** Python 3, PyTorch CPU float64/complex128, unittest, JSON manifests, Slurm shell contracts, ABACUS reader-v1, LibRPA.

---

### Task 1: Tangent-space family gradients

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_candidates.py`
- Create: `SIAB/tests/test_periodic_galerkin_candidates.py`

- [ ] Write failing tests for fixed-prefix projection, Stiefel tangency, finite normalized family gradients, and deterministic family order.
- [ ] Run `python -m unittest -v test_periodic_galerkin_candidates` and require failures caused by the missing module.
- [ ] Implement tangent projection and family-gradient evaluation by reusing the prepared block contractions and `_global_pi_loss`.
- [ ] Re-run the focused test and `test_periodic_galerkin_fit`.

### Task 2: Deterministic Pareto candidate bank

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/periodic_galerkin_candidates.py`
- Modify: `SIAB/tests/test_periodic_galerkin_candidates.py`

- [ ] Write failing tests for weights `0.25,0.50,0.75`, fixed trust radius, exact fixed-prefix preservation, retraction, degradation gates, and stable orbital hashes.
- [ ] Verify the new tests fail for missing candidate-bank behavior.
- [ ] Implement the smallest candidate-bank builder satisfying the tests.
- [ ] Re-run both candidate and periodic Galerkin suites.

### Task 3: Manifest state evaluator

**Files:**
- Create: `SIAB/opt_orb_pytorch_dpsi/galerkin_binding_workflow.py`
- Create: `SIAB/tests/test_galerkin_binding_workflow.py`

- [ ] Write failing tests for the ordered states, one-next-action rule, terminal rejection, acceptance tolerance, and duplicate active/completed fingerprints.
- [ ] Verify RED.
- [ ] Implement the pure evaluator without scheduler calls or material constants.
- [ ] Verify GREEN and run all SIAB Python tests touched by the workflow.

### Task 4: C system adapter and df contracts

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/c_diamond.json`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/build_c_candidate_bank.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_candidate_bank_df.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/test_c_galerkin_binding_workflow.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/test_c_galerkin_binding_workflow_contract.sh`

- [ ] Write failing adapter and shell-contract tests for 22 AO/C, fixed `2s2p1d`, atom `3/1` occupation, PBE 10 meV, q2/q6, q6/q7/q8, eight q stars, product-PCA `1e-4`, all bands, full Coulomb, LibRPA `d4810f73`, and 0.1 eV/C acceptance.
- [ ] Verify RED.
- [ ] Implement the adapter loader, candidate-bank CLI and immutable one-rank df runner.
- [ ] Verify focused Python and shell tests.

### Task 5: Real read-only C gradient audit

**Files:**
- Generated remote artifact only under the immutable C campaign root.

- [ ] Verify no active or completed duplicate fingerprint exists.
- [ ] Deploy the exact committed source to df and submit one read-only/CPU Galerkin gradient-bank job from the accepted reverse-p3 orbital.
- [ ] Require `COMPLETED/0:0`, successful status/provenance, finite family losses and gradients, fixed-prefix identity, valid capture/condition, and deterministic candidate hashes.
- [ ] Select at most one candidate through the fixed held-out and physical promotion sequence; do not submit a second candidate concurrently.

### Task 6: Documentation and commit verification

**Files:**
- Modify the canonical `development_notes` TeX source outside this repository.

- [ ] Record the reusable workflow, its boundary, source commit, job IDs, hashes and stage result.
- [ ] Compile with XeLaTeX and visually inspect changed pages.
- [ ] Run the complete focused regression set and inspect `git diff`.
- [ ] Commit with Codex as author and AroundPeking as committer, verify attribution, and push the branch.
