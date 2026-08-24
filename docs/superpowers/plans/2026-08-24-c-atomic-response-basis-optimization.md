# C Atomic Response-Basis Optimization Implementation Plan

> **Execution rule:** Preserve the accepted FD8 target and 20-step checkpoint
> hashes. Do not run SOS or promote a basis in this campaign.

**Goal:** Converge the existing C `3s3p2d` response-only optimization, resolve
the remaining atomic Sternheimer response by angular momentum, and construct
one deterministic next-shell seed.

**Architecture:** Add a self-contained continuation campaign beside the
existing short gradient gate. A Python contract prepares and audits the
continuation, a spectrum analyzer reuses SIAB's native radial residual
eigensolver and coefficient I/O, and one df Slurm job runs the optimizer and,
only after a convergence pass, the spectrum analysis. Every stage is guarded
by fixed hashes and writes machine-readable provenance.

**Stack:** Python 3, PyTorch, existing SIAB Sternheimer modules, unittest,
Slurm on df `p1`, LaTeX development note.

---

## Task 1: Freeze The Continuation Contract

**Files:**
- Create: `SIAB/example_C_sternheimer/atomic_basis_optimization/SIAB_INPUT.tzdp_continuation.json`
- Create: `SIAB/example_C_sternheimer/atomic_basis_optimization/continuation_campaign.py`
- Create: `SIAB/example_C_sternheimer/atomic_basis_optimization/tests/test_continuation_campaign.py`

1. Write failing tests for the exact `3s3p2d0f0g` shell count, frozen DZP
   prefix, Adam `lr=1e-3`, `max_steps=3000`, and immutable target/checkpoint
   hashes.
2. Run:

   ```bash
   python3 -m unittest SIAB/example_C_sternheimer/atomic_basis_optimization/tests/test_continuation_campaign.py
   ```

   Confirm failure because the campaign module and template do not exist.
3. Implement template validation, SHA256 validation, nonempty-output refusal,
   and INPUT/manifest generation from the 20-step checkpoint.
4. Re-run the targeted tests and keep them green.

## Task 2: Make Convergence A Machine Gate

**Files:**
- Modify: `SIAB/example_C_sternheimer/atomic_basis_optimization/continuation_campaign.py`
- Modify: `SIAB/example_C_sternheimer/atomic_basis_optimization/tests/test_continuation_campaign.py`

1. Add failing tests for parsing accepted Sternheimer rows, bitwise preservation
   of the five DZP columns, condition-limit rejection, the final-100-step
   relative-best-loss criterion, the 51-step nonimprovement stop, and the
   `CONTINUE_REQUIRED` state.
2. Verify the tests fail for the missing audit behavior.
3. Implement the smallest parser/auditor that emits one of
   `TZDP_CONVERGED` or `CONTINUE_REQUIRED`. Define the final-window quantity as
   `(best_before_last_100 - best_all) / best_before_last_100`; detect the
   built-in stop only when fewer than 3000 optimizer rows exist and the final
   51 accepted losses do not improve the previous best.
4. Re-run the targeted tests.

## Task 3: Rank Residual Angular Channels And Seed One Shell

**Files:**
- Create: `SIAB/example_C_sternheimer/atomic_basis_optimization/analyze_residual_spectrum.py`
- Create: `SIAB/example_C_sternheimer/atomic_basis_optimization/tests/test_residual_spectrum_analysis.py`

1. Add failing unit tests using synthetic spectra for per-channel weight,
   capture fractions, `lambda_1/(2l+1)` ranking, 1% tie refusal, and exact
   one-column shell append.
2. Verify failure because the analyzer does not exist.
3. Implement analysis helpers, then connect them to
   `radial_residual_spectrum_many`, `read_optimizer_coefficients`, and
   `write_optimizer_coefficients` for C with 31 Bessel rows and `l=0,...,4`.
4. Emit `RESIDUAL_SPECTRUM.json` and, only for a unique winner,
   `ORBITAL_RESULTS.next_shell_seed.txt`. Assert all old columns are unchanged
   and exactly one selected channel gains one column.
5. Re-run both new test modules and existing SIAB response-selection tests.

## Task 4: Add A Reproducible df Runner

**Files:**
- Create: `SIAB/example_C_sternheimer/atomic_basis_optimization/run_tzdp_continuation_df.slurm`
- Create: `SIAB/example_C_sternheimer/atomic_basis_optimization/submit_tzdp_continuation_df.sh`
- Create: `SIAB/example_C_sternheimer/atomic_basis_optimization/README.md`
- Modify: `SIAB/example_C_sternheimer/atomic_basis_optimization/tests/test_continuation_campaign.py`

1. Add failing static tests for df `p1`, one task, 8 CPUs, 16 GiB, no debug or
   exclusive request, fixed source-commit verification, duplicate refusal,
   `sbatch --test-only`, and spectrum execution only after `TZDP_CONVERGED`.
2. Implement the runner and submitter. Record Python/PyTorch versions and
   `/usr/bin/time -v`; never overwrite a prior output or receipt.
3. Run the complete new test directory and the pre-existing C gate tests.
4. Commit and push this executable campaign with Codex author and AroundPeking
   committer attribution.

## Task 5: Execute The Accepted Continuation

**Remote inputs:**
- Target: `/data/home/df_iopcas_ghj/gw/260823/c-atom-siab-reference-fixedocc-p1-8cf890e8/OUT.C_SIAB_REFERENCE/sternheimer_matrix.dat`
- Checkpoint: `/data/home/df_iopcas_ghj/gw/260824/c-siab-atomic-gradient-2816daad/runs/atomic_gradient_gate_df/ORBITAL_RESULTS.txt`

1. Sync the committed branch to df and verify the remote source commit.
2. Create a new campaign directory without altering the accepted parent.
3. Run `sbatch --test-only`; submit once only if no receipt, result, or live job
   already exists.
4. Monitor scheduler state separately from optimizer convergence. On
   completion, audit hashes, convergence state, loss trajectory, resources,
   and residual-spectrum artifacts. Do not select a shell if the status is
   `CONTINUE_REQUIRED`.

## Task 6: Archive And Document The Physical Result

**Files:**
- Modify: `/Users/ghj/同步空间/AITP_project/delta_st_rpa_project/development_notes/sections/c_atom_delta_siab_gate.tex`
- Create: `/Users/ghj/同步空间/AITP_project/delta_st_rpa_project/development_notes/data/c_atomic_response_basis_optimization_20260824/`

1. Archive compact inputs, manifests, convergence data, spectrum report,
   selected seed if present, Slurm accounting, and hashes.
2. Add the convergence trajectory and the `l=0,...,4` spectrum table to the C
   SIAB section. State explicitly that the seed is not a validated basis and
   that projected-Pi/C-C/SOS validation remains pending.
3. Rebuild the development-note PDF and visually inspect all affected pages for
   table overflow, missing symbols, and ordering.
4. Commit code-side documentation updates and report the exact physical gate
   reached; do not claim the C basis is complete.
