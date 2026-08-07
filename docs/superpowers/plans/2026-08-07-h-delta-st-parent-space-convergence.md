# H Delta-ST Parent-Space Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task by task. For every behavior change, use test-driven development.

**Goal:** Establish, on server66, a full-grid H-atom Delta-ST reference and determine the radial and angular completeness needed for the uncompressed Bessel parent space to reproduce its full-Coulomb symmetric response and RPA correlation energy.

**Architecture:** ABACUS produces the immutable full-grid Delta-ST reference and versioned primitive Galerkin matrices. SIAB reads both products, validates their shared physical protocol, solves in selected primitive subspaces, transforms both responses with one numerical full-Ewald Coulomb matrix, and reports matrix and energy errors. The zero-order TZDP basis is fixed throughout; only the response parent space changes.

**Tech Stack:** ABACUS C++17, MPI/OpenMP, SIAB Python 3/PyTorch complex128, `unittest`, LibRPA v1 files, server66 direct execution, XeLaTeX research note.

---

### Task 1: Freeze and commit the protocol

**Files:**
- Add: `docs/superpowers/specs/2026-08-07-h-delta-st-parent-space-convergence-design.md`
- Add: `docs/superpowers/plans/2026-08-07-h-delta-st-parent-space-convergence.md`

- [ ] Commit the fixed physical setup, formulas, convergence axes, acceptance thresholds, and scientific boundary.
- [ ] Push `codex/galerkin-sternheimer` and verify Codex author/AroundPeking committer attribution.

### Task 2: Audit and certify the full-grid Delta-ST producer

**Files:**
- Inspect: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`
- Inspect: `source/source_lcao/module_ri/sternheimer_delta.cpp`
- Inspect: `source/source_io/module_parameter/read_input_item_output.cpp`
- Add remotely: immutable H one-frequency gate directory and manifest

- [ ] Sync commit `8663f4ef` to a new clean SIAB checkout on server66; preserve the existing dirty tree untouched.
- [ ] Record the ABACUS source revision, configured commit metadata, executable SHA256, linked LibRPA provenance, and all physical input hashes.
- [ ] Trace the `out_sternheimer_librpa + sternheimer_delta` path from input validation through `M` assembly and v1 writeout. Record the spin, occupation, `2Re`, volume-element, auxiliary-channel, and Coulomb conventions.
- [ ] Prepare the exact H 20-Angstrom, 100-Ry, TZDP 8-bohr, `exx_pca_thr=1e-4` one-frequency input by copying the accepted 16-frequency case and replacing only the frequency list/count.
- [ ] Run on server66 and require 33 auxiliary channels, matching channel order, one complete Hermitian `M`, and compatible `V_full` dimensions.
- [ ] Stop before the formal run if any protocol field differs.

### Task 3: Produce the immutable 16-frequency H Delta-ST reference

**Files:**
- Add remotely: `h_delta_st_ref_nf16_e100_b100_<revision>/`
- Add later: compact evidence under `SIAB/example_H_sternheimer/`

- [ ] Reuse the accepted fixed 16 frequencies and weights; do not regenerate them.
- [ ] Use the validated molecular frequency/channel MPI distribution, normal compute resources available on server66, and all cores per node.
- [ ] Run the complete ABACUS producer and preserve input, stdout/stderr, `OUT.ABACUS`, response v1 files, Coulomb v1 files, progress diagnostics, and timing/memory record.
- [ ] Validate all 16 blocks, strict frequency ordering, Hermiticity, finite values, residual diagnostics, channel count/order, and hashes.
- [ ] Calculate `Pi_ref` and `E_c,ref^RPA`; independently check the same fixed-AO trace-log path against its existing LibRPA result before accepting the energy routine.

### Task 4: Decouple primitive angular momentum from the fixed AO basis

**Files:**
- Modify in ABACUS: `source/source_io/module_parameter/input_parameter.h`
- Modify in ABACUS: `source/source_io/module_parameter/read_input_item_output.cpp`
- Modify in ABACUS: `source/source_lcao/module_ri/sternheimer_abacus_st_smoke.cpp`
- Modify/add ABACUS input and primitive writer tests

- [ ] Write a RED input test requiring default behavior to retain fixed-AO `nwl` and explicit `sternheimer_galerkin_primitive_lmax=3` to accept/export `l=0..3`.
- [ ] Run the focused test and verify the expected failure because the input does not exist.
- [ ] Add the parameter with default `-1`; resolve `-1` to the current fixed-AO `nwl`, reject values below `-1`, and pass the resolved value only to primitive construction.
- [ ] Write a RED primitive-construction test showing a fixed H `nwl=1` basis can produce a valid `l=2` block when explicitly requested.
- [ ] Implement the minimal construction change and make both tests GREEN.
- [ ] Run serial input tests and focused Sternheimer SIAB tests, then commit and push the ABACUS branch with verified attribution.
- [ ] Build the exact pushed revision in a new server66 build directory; never replace a running executable.

### Task 5: Add protocol-checked parent-space analysis

**Files:**
- Add: `SIAB/opt_orb_pytorch_dpsi/delta_st_parent_space.py`
- Add: `SIAB/tests/test_delta_st_parent_space.py`
- Modify only if needed: `SIAB/opt_orb_pytorch_dpsi/primitive_galerkin.py`

- [ ] Write RED parser/validation tests for frequency, auxiliary channel order, spin occupation, Coulomb dimensions, kernel label, and provenance mismatch.
- [ ] Implement fail-closed matching between the Delta-ST reference, primitive sidecar, and `V_full`.
- [ ] Write a two-level RED test for `V_full^{-1/2} M V_full^{-1/2}` and the discrete trace-log energy, including eigenvalue threshold handling.
- [ ] Implement complex128 symmetric-response and RPA-energy routines.
- [ ] Add a regression that reproduces the existing fixed-AO ABACUS-to-LibRPA RPA energy within the printed precision.
- [ ] Write a RED identity-parent test with occupations generated from the producer spin metadata rather than from the fixed nine-AO occupation array.
- [ ] Implement primitive block selection by radial index and `lmax`, then call the certified Galerkin solver.
- [ ] Run focused and existing Galerkin/primitive tests twice on server66; require identical normalized summaries.
- [ ] Commit and push SIAB with verified attribution.

### Task 6: Run radial and angular completeness scans

**Files:**
- Add remotely: immutable run directories for every accepted point
- Add: compact CSV/JSON summaries and plotting input under `SIAB/example_H_sternheimer/`

- [ ] Estimate primitive matrix memory before every point and reject settings that exceed the server66 per-node memory envelope.
- [ ] Radial scan at `lmax=1`: start from `bessel_nao_ecut=100 Ry`, then increase one point at a time until two adjacent points change `E_c^RPA` by less than 0.01 kcal/mol and the response error no longer improves materially.
- [ ] Angular scan at the accepted radial setting for `lmax=1,2,3,4`.
- [ ] For every point record primitive count/rank/condition, all-frequency and maximum per-frequency `Pi` errors, `E_c^RPA`, wall time, memory, revision, binary hash, and input hashes.
- [ ] Require the matrix and energy acceptance criteria simultaneously. If `lmax=4` fails, document the failure and extend the scan rather than optimizing a compact basis.

### Task 7: Document the physical result

**Files:**
- Modify: `sternheimer_siab_project/sections/siab_primitive_galerkin_gate.tex`
- Modify: `sternheimer_siab_project/main.tex` only if section placement changes
- Add: convergence figures and machine-readable data under `sternheimer_siab_project/data/`

- [ ] Present the immutable reference protocol before the numerical results.
- [ ] Add one radial-convergence figure and one angular-convergence figure with both response and RPA-energy errors.
- [ ] Add a table containing resources, wall time, memory, rank/condition, and acceptance status for every point.
- [ ] State explicitly whether the uncompressed parent space passes and what this permits next: compact AO optimization and then H2 transferability/BSSE tests.
- [ ] Compile with XeLaTeX, check unresolved references and changed-section overfull boxes, render the changed pages, and inspect them visually.
- [ ] Commit/push each completed code phase and the final documentation phase separately.
