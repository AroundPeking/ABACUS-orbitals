# C Atomic Delta-ST Response Equivalence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the accepted fixed-occupation and field-seeded free-occupation zero-field C triplet states produce the same FD8 Delta-Sternheimer RPA response before either state is used as SIAB training data.

**Architecture:** Reuse only the final `fixed_zero_restart` and `dir0/free_restart2` artifacts from the passed PBE gate. Generate one immutable six-point GreenX grid from the union of their occupied-to-virtual transition windows, run both response branches with byte-identical physical assets and frequency data, process both with the same full-Coulomb LibRPA executable, and compare rotationally invariant response spectra and trace-log quantities.

**Tech Stack:** Python 3 standard library plus NumPy for binary reader-v1 analysis, Bash/Slurm for server 66 execution, ABACUS FD8 Delta-ST, LibRPA Sternheimer reader-v1, unittest for local contract tests.

---

### Task 1: Freeze the response contract

**Files:**
- Create: `SIAB/example_C_sternheimer/delta_response_gate/response_contract.py`
- Create: `SIAB/example_C_sternheimer/delta_response_gate/tests/test_response_contract.py`
- Modify: `docs/superpowers/specs/2026-08-21-c-atom-reference-equivalence-design.md`

- [x] **Step 1: Write failing tests for the accepted phase names, exact parameter contract, integer occupations, and union transition window.**

  The fixed source must be `fixed/fixed_zero_restart`, the free source must be `dir0/free_restart2`, and both must contain $N_\uparrow=3$, $N_\downarrow=1$. The union window is the minimum same-spin occupied-to-virtual gap and maximum same-spin transition across both files, converted from eV to Ha.

- [x] **Step 2: Run `python3 -m unittest SIAB.example_C_sternheimer.delta_response_gate.tests.test_response_contract -v` and verify it fails because `response_contract.py` does not exist.**

- [x] **Step 3: Implement the parser and constants.**

  Freeze `nfreq=6`, FD order 8, solver tolerance `1e-6`, PCA `1e-4`, `nx=ny=nz=135`, `ecutwfc=30 Ry`, `exx_ccp_rmesh_times=rpa_ccp_rmesh_times=1`, Gamma only, full-Ewald Coulomb, `nbands=22`, and the two accepted phase names. Reject fractional occupations, absent virtual states, spin-count mismatch, and a non-positive transition window.

- [x] **Step 4: Run the focused test and the existing PBE gate suite.**

  Run:

  ```bash
  python3 -m unittest SIAB.example_C_sternheimer.delta_response_gate.tests.test_response_contract -v
  python3 -m unittest discover -s SIAB/example_C_sternheimer/pbe_reference_gate/tests -v
  ```

  Expected: all tests pass and the PBE gate behavior remains unchanged.

- [x] **Step 5: Correct the stale fixed-route wording in the design specification.**

  Replace the obsolete `fixed_cold -> fixed_restart` response source with `fixed_field_seed -> fixed_zero_restart`; explicitly state that only `fixed_zero_restart` and `dir0/free_restart2` enter the Delta-ST gate.

### Task 2: Build immutable input and frequency-grid preparation

**Files:**
- Create: `SIAB/example_C_sternheimer/delta_response_gate/prepare_response_gate.py`
- Create: `SIAB/example_C_sternheimer/delta_response_gate/tests/test_prepare_response_gate.py`
- Create: `SIAB/example_C_sternheimer/delta_response_gate/generate_frequency_grid.py`
- Create: `SIAB/example_C_sternheimer/delta_response_gate/tests/test_generate_frequency_grid.py`

- [x] **Step 1: Write failing tests for restart staging and GreenX output parsing.**

  Require four phase-local restart files, exact source hashes, no symlinks in prepared branch inputs, identical PP/orbital/STRU/KPT/frequency hashes across branches, `ocp=1` only for the fixed branch, `ocp=0` only for the free branch, and exactly six positive ordered `(omega, weight)` rows.

- [x] **Step 2: Run both focused test modules and verify the expected missing-module failures.**

- [x] **Step 3: Implement preparation without overwriting an existing formal root.**

  Copy the final PBE outputs `wfs1_nao.txt`, `wfs2_nao.txt`, `chgs1.cube`, and `chgs2.cube` into each branch. Render zero-field response `INPUT` files with `rpa=1`, `out_librpa_reader_version=1`, `out_sternheimer_librpa=1`, `sternheimer_delta=1`, `sternheimer_delta_max_states=0`, `sternheimer_delta_norm_tol=1e-10`, explicit shared frequency file, and branch-specific occupation control.

- [x] **Step 4: Implement GreenX grid generation from the union transition window.**

  Invoke the standalone GreenX executable from the pinned LibRPA feature build with the explicit union transition window and `nfreq=6`; parse the six frequency/weight rows and write one `fixed_frequency_grid.dat` plus a JSON manifest containing the source `eig_occ.txt` hashes, window, GreenX hash, and grid hash.

- [x] **Step 5: Run focused and full local tests.**

  Expected: deterministic prepared inputs and manifests, with all negative-path fixtures rejected.

### Task 3: Run ABACUS and LibRPA with duplicate protection

**Files:**
- Create: `SIAB/example_C_sternheimer/delta_response_gate/run_response_branch_server66.slurm`
- Create: `SIAB/example_C_sternheimer/delta_response_gate/run_librpa_branch_server66.slurm`
- Create: `SIAB/example_C_sternheimer/delta_response_gate/submit_response_gate_server66.sh`
- Create: `SIAB/example_C_sternheimer/delta_response_gate/tests/test_hpc_contract.py`

- [x] **Step 1: Write failing static tests for scheduler resources, runtime hashes, duplicate guards, restart evidence, and success gates.**

  ABACUS uses partition `640`, six nodes, one MPI rank and 48 OpenMP threads per node, 180000 MB per node, exclusive allocation, and the 24-hour limit. LibRPA uses one full node after successful completion of both ABACUS array tasks. The submitter must reject any active/prior matching job, claim, result, failure, or receipt.

- [x] **Step 2: Implement the two-branch Slurm runner.**

  Recheck restart-load log messages, PBE convergence, integer occupations, `STERNHEIMER_CHI0.dat` success, FD8, six frequencies, all equations converged, maximum residual at most `1e-6`, six reader-v1 response files, and one Hermitian full-Coulomb matrix. Record executable/source/asset hashes and scheduler records.

- [x] **Step 3: Implement the LibRPA runner.**

  Use `task=sternheimer_rpa`, the full-Coulomb prefix, `sqrt_coulomb_threshold=1e-5`, Gamma inclusion, the same response/frequency files, and no head/wing replacement. Require finite per-frequency trace-log rows and one finite total `EcRPA` with negligible imaginary part.

- [x] **Step 4: Implement duplicate-safe submission and run all static tests plus `bash -n`.**

- [x] **Step 5: Correct the molecular-report audit before production starts.**

  Source inspection showed that the atomic Gamma fallback writes
  `sternheimer_delta`, `ccp_rmesh_times`, and `occupied_bands`, whereas the
  periodic path writes different field names.  Cancel the untouched first
  submission, replace the mixed-field checks, and separate the ABFS CCP
  `Fock, alpha=1, singularity_correction=limits` perturbation potential from
  the reader-v1 full-Coulomb response metric in the completion record.  Verify
  the actual `v1_coulomb_full_iq_1_rank0.dat` file and the LibRPA full-Coulomb
  prefix instead of requiring a periodic-only report line.

### Task 4: Audit physical equivalence

**Files:**
- Create: `SIAB/example_C_sternheimer/delta_response_gate/audit_response_gate.py`
- Create: `SIAB/example_C_sternheimer/delta_response_gate/tests/test_audit_response_gate.py`

- [x] **Step 1: Write failing tests for reader-v1 parsing, Coulomb assembly, $\Pi=V^{-1/2}MV^{-1/2}$, eigenvalue matching, trace-log parsing, and energy acceptance.**

- [x] **Step 2: Implement the audit.**

  Hermitize $V$ and $M$, keep the same positive Coulomb subspace selected by `sqrt_coulomb_threshold=1e-5`, construct $\Pi$, and compare sorted eigenvalues at each frequency. Require relative differences below `1e-3`, per-frequency LibRPA integrand differences below `1e-3`, equal `naux` and identical frequency/Coulomb provenance, and $\lvert\Delta E_c^{\rm RPA}\rvert<0.1$ kcal/mol.

- [x] **Step 3: Run focused tests and the complete local suite.**

### Task 5: Execute and record the gate

**Files:**
- Create after validated execution: `SIAB/example_C_sternheimer/delta_response_gate/results/DELTA_RESPONSE_GATE_RESULT.md`
- Modify: `SIAB/example_C_sternheimer/delta_response_gate/README.md`
- Modify: `docs/superpowers/specs/2026-08-21-c-atom-reference-equivalence-design.md`

- [ ] **Step 1: Stage an immutable source archive on server 66 and run local/remote preflight tests.**

- [ ] **Step 2: Query `squeue` and `sacct`; submit exactly once only if no matching work exists.**

- [ ] **Step 3: Validate scheduler, ABACUS, response, LibRPA, and physical-equivalence gates separately.**

- [ ] **Step 4: Record exact jobs, paths, hashes, resources, wall times, transition window, `naux`, residuals, response-spectrum errors, trace-log errors, and both (E_c^{\rm RPA}) values.**

- [ ] **Step 5: Commit the accepted result and decide the SIAB reference.**

  Only a passed gate authorizes the free zero-field C state as the rotationally independent SIAB training reference. A failed gate blocks basis optimization and triggers a three-determinant ensemble design rather than looser thresholds.
