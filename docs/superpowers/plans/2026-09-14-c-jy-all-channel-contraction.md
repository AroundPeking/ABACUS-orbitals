# C jY All-Channel Contraction Implementation Plan

> Execute in the existing isolated worktree
> `/Users/ghj/.config/superpowers/worktrees/ABACUS-orbitals/c-response-metric-20260913`.
> Do not compile ABACUS or LibRPA locally.

## Task 1: Persist The Occupied Mother Embedding

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/periodic_available_mother.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/response_target.py`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_available_mother.py`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_jy_angular_ladder.py`
- Modify: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_jy_joined_ladder.py`
- Test: `SIAB/tests/test_response_target.py`
- Test: target-builder contract tests under `SIAB/tests/`

1. Add a failing test that requires the target consumer to receive an occupied
   embedding whose rows, together with the virtual embedding, reconstruct the
   retained mother metric.
2. Run the focused Python test and verify the expected interface failure.
3. Extend the read-only mother callback and target NPZ/manifest schema without
   changing the response covariance or Pi definition.
4. Run focused and surrounding target tests.

## Task 2: Add Occupied-Safe Shared Radial Loss

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/response_radial_fit.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/response_fit_iteration.py`
- Test: `SIAB/tests/test_response_radial_fit.py`
- Test: `SIAB/tests/test_response_fit_iteration.py`

1. Add failing tests for exact occupied capture, capture loss under an all-free
   contraction, and rejection of a trial below the hard capture floor.
2. Verify the tests fail because the occupied embedding/capture API is absent.
3. Implement candidate overlap and occupied capture from the combined retained
   frame, a normalized occupied residual for gradient guidance, and a hard
   minimum-capture constraint in line-search acceptance.
4. Preserve backward compatibility for response-only diagnostic artifacts.
5. Run focused tests, finite-difference gradient tests, and the full Python test
   suite for the optimizer modules.

## Task 3: Implement The All-Channel Rank-Ladder Runner

**Files:**
- Add: `SIAB/example_C_sternheimer/periodic_basis_optimization/galerkin_binding_workflow/run_c_jy_all_channel_contraction.py`
- Add/modify runner tests under `SIAB/tests/`
- Modify immutable-source manifest helpers if required.

1. Add failing contract tests that reject a frozen prefix, incomplete occupied
   targets, non-512-sector input, and a profile with missing mother channels.
2. Implement sequential profiles `3321`, `4432`, `5543`, and `6654`, with all
   radial columns free and explicit response/occupied diagnostics.
3. Write immutable initial/final coefficients, progress, best-checkpoint hash,
   per-profile feasibility, and provenance.  Label the old frozen-`spd` run as an
   ablation only.
4. Run static and Python tests locally; do not run native compilation.

## Task 4: Rebuild Targets And Run The Occupied-Safe Contraction On DF

**Files:** remote immutable deployment and new run roots only.

1. Commit and push the tested code with Codex as author and AroundPeking as
   committer.
2. Deploy immutably on `df`; verify source manifest hashes.
3. Refresh live CPU partitions and choose a compatible available partition.
4. Rebuild the 512 enriched target files from the existing accepted jY source
   data only.  Do not run Delta-ST, ABACUS, or LibRPA.
5. Validate all target hashes and metric reconstruction.
6. Run one sequential all-channel rank-ladder contraction.  Do not duplicate a
   matching active or completed job.

## Task 5: Direct Full-q/Frequency RPA Refinement

**Files:** optimizer/refinement module and tests selected after Task 4 evidence.

1. Reuse the existing global RPA trace-log implementation and write failing
   tests for per-q/per-frequency jY-reference mismatch and occupied feasibility.
2. Refine only feasible rank-ladder candidates, retaining per-sector errors.
3. Require response and actual RPA-reference errors to improve without losing
   occupied capture.  Promote rank only when necessary.

## Task 6: Physical LCAO SOS Validation

1. Freeze exactly one compact candidate and its hash.
2. Run the PBE occupied-manifold gate.
3. Run ordinary all-band LCAO SOS RPA using 12 frequencies, PCA `1e-6`, full
   periodic Coulomb, and qavg head/wing under the frozen production binaries.
4. Report basis size, PBE result, `Ec`, per-q diagnostics, and error versus
   Delta-ST.  Accept only below `0.1 eV/C`; otherwise return to the next rank or
   a measured missing angular channel.

## Task 7: Documentation And Delivery

1. Update the standalone canonical basis-optimization TeX note to replace the
   single-f main route with the all-channel jY contraction route.
2. Compile and visually inspect changed PDF pages.
3. Verify commits, authorship, remote push, job provenance, and no local native
   build artifacts.
