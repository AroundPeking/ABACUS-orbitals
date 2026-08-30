# C Relaxed-DZP Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize a partially relaxed `3s3p2d` carbon basis, require its atom and diamond PBE energies to remain within `10 meV` of the original TZDP results, and validate its ordinary all-band SOS binding energy against Delta-ST.

**Architecture:** Extend the existing balanced atom-solid optimizer with one `relaxed_dzp` profile that starts from the original SG15 TZDP orbital, fixes only the first s and p radial profiles, and uses the complete initial candidate as the occupied-capture reference. Export a provenance-checked candidate, run a fail-closed atom-solid PBE gate, and release the threshold-only eight-q SOS chain only through an `afterok` dependency on that gate.

**Tech Stack:** Python 3.9, PyTorch 1.9, ABACUS `55d25e3c9`, LibRPA `d4810f73`, Bash/Slurm, `unittest`.

---

### Task 1: Add the partially relaxed optimizer profile

**Files:**
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/run_c_atom_solid_balanced_one_g_df.slurm`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/submit_c_atom_solid_balanced_one_g_df.sh`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/tests/test_c_atom_solid_balanced_job.py`

- [x] Add a failing test requiring profile `relaxed_dzp`, SG15 TZDP coefficient hash `b58a2183...`, `nu=3,3,2,0,0`, `fixed_nu=1,1,0,0,0`, and `occupied-capture-reference initial_candidate`.
- [x] Run `python3 -m unittest SIAB.example_C_sternheimer.periodic_basis_optimization.tests.test_c_atom_solid_balanced_job -v` and confirm the new test fails.
- [x] Implement the profile without changing `one_g` or `no_f`; preserve empty high-l layout columns and record the initial coefficient hash.
- [x] Run the focused test and the full periodic optimizer test directory.  The focused tests pass; dependency-bearing legacy tests require the pinned remote PyTorch/NumPy runtime and are covered by the pilot.
- [x] Commit with Codex as author and AroundPeking as committer.

### Task 2: Export one immutable relaxed candidate

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/export_relaxed_dzp_candidate.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/tests/test_export_relaxed_dzp_candidate.py`

- [ ] Write tests that require successful optimizer status, final orbital equality with the best checkpoint, finite and improved atom/solid losses, occupied-capture compliance, and the exact `3s3p2d` AO count.
- [ ] Implement atomic export of the orbital and `CANDIDATE.json` with hashes, source job, optimizer commit, `fixed_optimizer_nu=1,1,0,0,0`, and `auxiliary_basis_rule=product_pca_threshold_only`.
- [ ] Reject duplicate output directories, missing q3/stability gates, or any non-finite value.
- [ ] Run focused and periodic optimizer tests, then commit.

### Task 3: Add the atom-solid PBE preservation gate

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_relaxed_dzp_pbe_gate_55d25e3c9.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/collect_relaxed_dzp_pbe_gate.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/test_collect_relaxed_dzp_pbe_gate.py`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/test_relaxed_dzp_pbe_gate_contract.sh`

- [ ] Test the collector against the frozen references `-5.329967462866793 Ha` and `-11.030536886845258 Ha`, including passing and each independent failing inequality.
- [ ] Implement a single-node sequential C-atom and diamond PBE gate using the exact reference executable, PP, geometry, k mesh, spin/occupation state, and convergence settings.
- [ ] Require genuine SCF convergence, atom occupations `3/1`, finite energies, individual atom/solid errors at most `10 meV`, and PBE binding error at most `10 meV/C`.
- [ ] Write `PBE_GATE.json` and complete provenance only on success; return nonzero on a physical failure.
- [ ] Run Python and shell contract tests, then commit.

### Task 4: Add a threshold-only eight-q SOS release

**Files:**
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_relaxed_dzp_atom_sos_55d25e3c9_d4810f73.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_relaxed_dzp_solid_qstar_55d25e3c9.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_relaxed_dzp_solid_qstar_sos_d4810f73.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/run_relaxed_dzp_binding_collect.slurm`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/release_relaxed_dzp_validation.sh`
- Create: `SIAB/example_C_sternheimer/periodic_basis_optimization/ordinary_sos_validation/test_relaxed_dzp_validation_contract.sh`

- [ ] Test that all ABACUS inputs use `exx_pca_threshold 1e-4`, remove `rpa_pca_fixed_nu`, use all 22/44 AO bands, full periodic Coulomb, six frequencies, and the eight representatives `1,2,3,6,7,8,11,28`.
- [ ] Adapt the validated threshold candidate atom and q-star routes to consume `CANDIDATE.json` without a fabricated truncation manifest.
- [ ] Make the release duplicate-safe and dependent on `PBE_GATE.json` success; submit no 64-q array and no Delta-ST job.
- [ ] Collect zero-order, RPA-correlation, and total SOS binding contributions and the difference from `6.902326 eV/C`.
- [ ] Run all contract tests and commit.

### Task 5: Deploy and execute the controlled campaign

**Files:**
- Update after validation: `/Users/ghj/同步空间/AITP_project/delta_st_rpa_project/development_notes`

- [ ] Deploy the committed tree immutably to df and record the commit/hash.
- [ ] Submit one two-step `relaxed_dzp` pilot after fixed-prefix job `3159150` completes; validate both family losses, capture, condition, and checkpoint identity.
- [ ] Submit one 500-step production only if the pilot passes.
- [ ] Run q3, radial, overlap, and virtual-spectrum gates; export one candidate.
- [ ] Run the PBE gate and release the SOS chain only if it passes.
- [ ] Report the original and optimized C atom PBE energy, diamond PBE energy per C, PBE binding energy, RPA correlation binding contribution, total SOS binding energy, and Delta-ST difference.
- [ ] Update and compile the canonical TeX note, inspect changed pages, and commit validated documentation.
