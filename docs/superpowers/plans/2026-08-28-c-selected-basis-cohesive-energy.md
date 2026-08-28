# Selected C Basis Solid Binding-Energy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing selected-orbital C atom SOS/Delta-ST endpoints and compute the diamond RPA@PBE binding-energy difference between SOS and Delta-ST.

**Architecture:** Reuse the accepted selected orbital and fixed-triplet atom inputs, but regenerate every zero-order, auxiliary, Coulomb, SOS, and Delta-ST artifact under one new immutable campaign root.  A pure Python collector combines successful atom and diamond summaries and refuses mixed or incomplete provenance.

**Tech Stack:** Bash/Slurm, ABACUS reader-v1 and Delta-ST, LibRPA `d4810f73`, Python 3, shell and Python regression tests.

---

### Task 1: Binding-Energy Collector

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/collect_selected_c_binding_energy.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/test_collect_selected_c_binding_energy.py`

- [ ] Write tests for the two-C binding-energy formula, correlation-only and total energies, the `0.1 kcal/mol/C` gate, and rejection of missing or mismatched endpoint summaries.
- [ ] Run the test and verify that it fails because the collector does not exist.
- [ ] Implement strict summary parsing and the binding-energy calculation.
- [ ] Run the focused and full Python tests.

### Task 2: Selected-Atom Producer Contract

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_selected_c_atom_response_pca_producer_55d25e3c9.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/test_selected_c_atom_binding_contract.sh`

- [ ] Add failing contract assertions for fixed triplet occupations, 56 bands per spin, `135^3` FD8, 20 Angstrom box, response-aware product-PCA `1e-4`, 261 auxiliary functions, full Coulomb, and immutable selected-orbital provenance.
- [ ] Run the shell test and verify the missing producer failure.
- [ ] Implement the one-node atom producer and all output gates.
- [ ] Run shell syntax and contract tests.

### Task 3: Atom SOS and Frequency Extraction

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_selected_c_atom_sos_d4810f73.slurm`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/test_selected_c_atom_binding_contract.sh`

- [ ] Extend the failing contract test to require all-band spin-polarized SOS, one-q full Coulomb, six minimax points, finite `EcRPA`, and atom frequency-grid extraction.
- [ ] Implement the LibRPA atom SOS stage using only the successful atom producer.
- [ ] Verify shell syntax and contract tests.

### Task 4: Matched Atom Delta-ST and LibRPA

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_selected_c_atom_matched_delta_55d25e3c9.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_selected_c_atom_matched_delta_reader_d4810f73.slurm`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/test_selected_c_atom_binding_contract.sh`

- [ ] Extend the failing contract test to require the atom SOS frequency hash, `global_equation`, FD8, fixed occupations, all-equation convergence, residual gate, basis-order comparisons, and finite Delta-ST LibRPA energy.
- [ ] Implement the 16-node atom response stage and one-node LibRPA reader.
- [ ] Verify shell syntax and contract tests.

### Task 5: Duplicate-Safe Atom Release

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/release_selected_c_atom_binding_chain.sh`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/test_selected_c_atom_binding_contract.sh`

- [ ] Add a failing test for immutable chain records, directory locks, scheduler/result gates, `sbatch --test-only`, and refusal to duplicate an existing atom stage.
- [ ] Implement a staged release that submits only the next validated atom stage.
- [ ] Run all periodic C validation tests.

### Task 6: Remote Validation and First Unique Job

- [ ] Commit and push the completed code with Codex as author and AroundPeking as committer.
- [ ] Deploy the exact commit to df and run all relevant tests there.
- [ ] Check the scheduler and campaign root for an exact active/completed atom duplicate.
- [ ] Run `sbatch --test-only`, then submit exactly one selected-atom producer if no duplicate exists.
- [ ] Record its job ID and immutable provenance; do not release later stages before success.

### Task 7: Canonical TeX Record

**Files:**
- Modify: `/Users/ghj/同步空间/AITP_project/delta_st_rpa_project/development_notes/sections/periodic_siab_basis_optimization_output.tex`

- [ ] Record the corrected solid binding-energy definition and explain why the old atom result cannot be reused.
- [ ] Record code commit, remote deployment, job IDs, resources, and current gates.
- [ ] Force recompilation, render changed pages, and inspect them for overflow or misplaced material.
