# H DZP-Core Sternheimer Response Basis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the checked-in H-DZP `2s1p` subspace exactly while optimizing the H-TZDP-only `3s2p` shells against the existing Sternheimer target, then support deterministic appended response shells when matching primitive blocks exist in the target.

**Architecture:** Treat the first two H s columns and first H p column as a fixed DFT core. `SternheimerSpillage` projects every response vector outside that full fixed subspace and optimizes only the remaining columns. Existing TZDP columns initialize the first response shell; extra same-l zeta columns are seeded deterministically, while a new angular channel is accepted only when the producer target contains every required `(atom,l,m)` primitive block.

**Tech Stack:** Python 3, PyTorch float64/complex128, `unittest`, SIAB JSON inputs, existing version-1 Sternheimer matrix reader.

---

### Task 1: Make the H campaign use a fixed DZP core

**Files:**
- Modify: `SIAB/tests/test_h_sternheimer_smoke.py`
- Modify: `SIAB/example_H_sternheimer/INPUT.st_only`
- Modify: `SIAB/example_H_sternheimer/INPUT.st_constrained`
- Modify: `SIAB/example_H_sternheimer/run_st_only.py`
- Modify: `SIAB/example_H_sternheimer/README.md`
- Modify: `docs/superpowers/specs/2026-07-18-sternheimer-siab-h-basis-design.md`

- [x] **Step 1: Write a failing configuration test**

Require both H inputs to freeze exactly `H/0/1`, `H/0/2`, and `H/1/1`, corresponding to `1s`, `2s`, and `1p`.

- [x] **Step 2: Run the focused test and verify the old level1-only input fails**

Run:

```bash
python3 -m unittest SIAB.tests.test_h_sternheimer_smoke.ExampleInputTest.test_inputs_match_exact_campaign_contract -v
```

Expected: `FAIL` because the existing inputs contain only the `1s` freeze specification.

- [x] **Step 3: Update both inputs and campaign validation**

Use this fixed core in both JSON files:

```json
"freeze_orbitals": [
  {"element": "H", "l": 0, "zeta": 1},
  {"element": "H", "l": 0, "zeta": 2},
  {"element": "H", "l": 1, "zeta": 1}
]
```

Change the runner to compare all three coefficient columns bitwise and all three exported radial functions against the checked-in H-TZDP orbital. Record the fixed core as an array in `campaign_summary.json`; reject a campaign if any fixed column or radial function changes.

- [x] **Step 4: Add an optimization regression test**

Run a deterministic synthetic optimization with the full DZP freeze set and assert:

```python
torch.equal(initial_s[:, :2], final_s[:, :2])
torch.equal(initial_p[:, :1], final_p[:, :1])
not torch.equal(initial_s[:, 2], final_s[:, 2])
not torch.equal(initial_p[:, 1], final_p[:, 1])
final_st < initial_st
```

- [x] **Step 5: Run the H Sternheimer test module**

Run:

```bash
python3 -m unittest SIAB.tests.test_h_sternheimer_smoke -v
```

Expected: all tests pass.

- [x] **Step 6: Commit the fixed-DZP stage**

```bash
git add SIAB/tests/test_h_sternheimer_smoke.py \
  SIAB/example_H_sternheimer/INPUT.st_only \
  SIAB/example_H_sternheimer/INPUT.st_constrained \
  SIAB/example_H_sternheimer/run_st_only.py \
  SIAB/example_H_sternheimer/README.md \
  docs/superpowers/specs/2026-07-18-sternheimer-siab-h-basis-design.md \
  docs/superpowers/plans/2026-07-19-h-dzp-st-response-basis.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m "feat(siab): preserve H DZP during ST optimization"
```

### Task 2: Support deterministic appended response shells

**Files:**
- Modify: `SIAB/tests/test_h_sternheimer_smoke.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/IO/func_C.py`
- Modify: `SIAB/example_H_sternheimer/README.md`

- [x] **Step 1: Write failing initialization tests**

Read the checked-in H-TZDP coefficients into `Nu={"H": [4,3]}` and require:

```python
loaded == {("H", 0, 0), ("H", 0, 1), ("H", 0, 2),
           ("H", 1, 0), ("H", 1, 1)}
```

The appended `4s` and `3p` columns must be finite, nonzero, deterministic for a fixed seed, and different for a different seed.

- [x] **Step 2: Run the tests and verify the missing explicit contract fails**

Run:

```bash
python3 -m unittest SIAB.tests.test_h_sternheimer_smoke.AppendedResponseShellTest -v
```

Expected: `FAIL` until the reader exposes and validates which requested columns were absent from the initialization file.

- [x] **Step 3: Add explicit initialization metadata and validation**

Return a small immutable initialization record containing loaded and appended `(element,l,zeta)` zero-based indices. Reject an initialization file that contains a column outside the requested `Nu`, a duplicate column, or a primitive-count mismatch. Keep existing callers compatible by retaining tuple unpacking as `(C, loaded_indices)` where no appended metadata is requested.

- [x] **Step 4: Verify angular-channel gating**

Add a test requesting `Nu={"H": [3,2,1]}` from an s/p-only target and require `_sternheimer_info_element` to raise an error identifying missing `H/l=2` primitive blocks. Add a matching synthetic d-block target and verify initialization/evaluation succeeds.

- [x] **Step 5: Run focused and full SIAB tests**

Run:

```bash
python3 -m unittest SIAB.tests.test_h_sternheimer_smoke -v
python3 -m unittest discover -s SIAB/tests -v
```

Expected: all tests pass.

- [x] **Step 6: Commit the response-shell extension stage**

```bash
git add SIAB/tests/test_h_sternheimer_smoke.py \
  SIAB/opt_orb_pytorch_dpsi/IO/func_C.py \
  SIAB/example_H_sternheimer/README.md
GIT_AUTHOR_NAME=Codex GIT_AUTHOR_EMAIL=codex@openai.com \
GIT_COMMITTER_NAME=AroundPeking \
GIT_COMMITTER_EMAIL=gonghuanjing@iphy.ac.cn \
git commit -m "feat(siab): initialize appended ST response shells"
```

### Task 3: Run and record the fixed-DZP H campaign

**Files:**
- Modify: `SIAB/example_H_sternheimer/README.md`
- Modify: `sternheimer_siab_project/main.tex` in the parent project only after the campaign artifacts exist.

- [x] **Step 1: Run the deterministic optimizer against the canonical target**

Run outside the Git tree:

```bash
python3 SIAB/example_H_sternheimer/run_st_only.py \
  --target /Users/ghj/同步空间/AITP_project/sternheimer_abacus/results/siab_h_option1_20260719/producer_21311439/sternheimer_matrix.dat \
  --output /Users/ghj/同步空间/AITP_project/sternheimer_abacus/results/siab_h_dzp_core_20260719
```

Expected: the summary reports all three DZP columns bitwise unchanged and a final ST loss no larger than the initial loss.

- [x] **Step 2: Plot and inspect every radial orbital**

Plot initial and final `1s`, `2s`, `3s`, `1p`, and `2p` on identical axes. Confirm the fixed DZP curves overlap exactly and report, without hiding, whether the variable response orbitals remain oscillatory.

- [x] **Step 3: Record numerical results**

Record the target hash, code commit, initial/final loss, condition number, wall time, coefficient hashes, and radial comparison. Do not claim improved RPA physics before the held-out H2 SOS calculation is run.

- [x] **Step 4: Commit the campaign record**

Commit only the concise README/TeX record and plot; keep raw campaign output outside Git.

### Task 4: Keep dpsi active while optimizing the ST response space

**Files:**
- Modify: `SIAB/opt_orb_pytorch_dpsi/optimization_loss.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/opt_orbital_converge.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/IO/func_C.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/main.py`
- Create: `SIAB/example_H_sternheimer/INPUT.st_dpsi_joint`
- Modify: focused tests and this README

- [x] **Step 1: Add a failing differentiable-loss test**

Require a normalized dpsi term to contribute a nonzero gradient even while the
DFT and dpsi hard constraints are satisfied.

- [x] **Step 2: Implement and validate `st_dpsi_joint`**

Use `joint_dpsi_weight * dpsi/dpsi_initial`, preserve both existing hinge
constraints, require linear dpsi input data, and select the feasible candidate
with minimum total loss.

- [x] **Step 3: Propagate named metadata and update campaign contracts**

Write `regularization_dpsi` to both `Spillage.dat` and
`ORBITAL_RESULTS.txt`. Keep pure-ST validation strict by requiring this value to
be zero in `st_only` campaigns.

- [x] **Step 4: Restore or regenerate the historical DFT/dpsi matrices**

df_dcu `normal` array job `21315279` regenerated the three
`orb_matrix.0.dat` and three `orb_matrix.1.dat` files from equilateral H3 at
0.7, 0.9, and 1.3 Angstrom. Every task used 30 MPI ranks, completed in under
40 seconds, passed the SCF/tag checks, and recorded matrix hashes. The producer
used immutable ABACUS commit `80a606f57a26`; the exact six hashes are recorded
in `SIAB/example_H_sternheimer/README.md`.

- [x] **Step 5: Run the first physical fixed-DZP joint campaign**

Evaluate the regenerated DFT/dpsi matrices and the canonical atomic ST target
from the same initial TZDP coefficients. Then freeze `1s,2s,1p`, optimize only
`3s,2p` with `lambda_dpsi=0.1`, and reject a result that violates either hard
DFT/dpsi tolerance. Plot all radial functions before interpreting H2 RPA.

df_dcu `normal` job `21315288` completed from source commit `d41f975e`.
It reduced ST, DFT, and dpsi losses simultaneously; both hinge penalties stayed
zero. The joint `3s,2p` orbitals have 2 and 1 radial nodes, compared with 3 and
10 in the ST-only result. Exact values and output hashes are recorded in the
example README.

- [x] **Step 6: Run the held-out H2/H SOS-RPA transfer test**

Replace only the H orbital in the validated 20-Angstrom H2/H SOS producer,
regenerate the ABFS and full-Coulomb data, and run LibRPA at 16 frequencies.
Recompute PBE, PBE-xc, EXX, and RPA correlation contributions for both H2 and
spin-polarized H; do not reuse the original TZDP non-correlation binding term.

The same-producer calculations completed for the checked-in TZDP, joint
`3s2p`, and joint `4s3p` bases. Their H2 RPA@PBE binding energies are
106.5361, 106.7371, and 106.8576 kcal/mol, respectively. The `4s3p` result is
still about 1.86 kcal/mol below the Thesis/FHI-aims reference.

### Task 5: Remove the angular-momentum floor before further H2 tuning

**Files:**
- Modify: the ABACUS Sternheimer-SIAB producer to generate an explicit
  primitive angular cutoff independent of the input orbital `Lmax`
- Modify: `SIAB/example_H_sternheimer/README.md`
- Modify: focused target-contract tests and joint H inputs

- [x] **Step 1: Quantify the current representability floor**

Decompose the fixed-DZP residual by auxiliary perturbation angular momentum.
The current target carries 8 s, 18 p, and 15 d channels, but has only s/p
primitive blocks. The joint `4s3p` loss is 0.328766; the complete current s/p
primitive space can reach only 0.301781. Missing d response contributes
0.301108 of the total loss.

- [x] **Step 2: Add complete d primitive blocks to the producer contract**

Generate 25 radial primitives for every `l=2`, `m=-2,-1,0,1,2` block without
adding those functions to the fixed DFT core. Reject incomplete magnetic
multiplets in both writer and reader tests.

ABACUS commit `efc128f33531` implements the independent output cutoff. RED job
`21315686` and GREEN jobs `21315747`/`21315765` verify primitive generation and
INPUT parsing. The immutable Release build is from job `21315778`; formal H
target job `21315811` completed all 225 primitive columns in `03:52:39`.

- [x] **Step 3: Select d shell count from the atomic residual spectrum**

Whiten each angular block with its primitive overlap, diagonalize the weighted
residual covariance, and choose `n_s,n_p,n_d` from a declared cumulative
capture threshold. Do not use the H2 binding energy to select the shell count.

The overlap-whitened shared-radial eigensolver and threshold selector are
implemented and passed 107/107 SIAB tests in job `21315899`. This checklist
item remains open until the completed 225-column atomic target is analyzed and
the numerical `n_d` is recorded.

Before inspecting the new atomic spectrum or any new H2 energy, the production
selection threshold is fixed at 99% cumulative d-channel residual weight. The
analysis will also report 90%, 95%, and 99.9% shell counts, but they are
diagnostics and must not be substituted after seeing the held-out result.

- [x] **Step 4: Optimize only with the joint DFT+dpsi path**

Keep `1s,2s,1p` bitwise fixed. Retain `st_only` only for regression/ablation;
never use its `.orb` in DFT or RPA production calculations.

The synthetic `lmax=2` joint-gradient regression passed as normal-node job
`21315966`: the d coefficient receives independently nonzero ST and DFT+dpsi
gradients, changes during a short `st_dpsi_joint` optimization, and leaves all
three frozen DZP columns bitwise unchanged. The real optimization still waits
for the completed target and the 99% spectrum-selected `n_d`.

Normal job `21316098` repeated the gradient audit with the three real
`lmaxmax=2` H3 matrix pairs. It parsed `Nl=3`, `Ne=25`, `Nu=[4,3,1]` and found
nonzero d-column gradient norms `1.0922811097e-6` (DFT origin) and
`1.9509412056e-5` (dpsi). This closes the real-data regularizer gate; it does
not select `n_d` or use H2.

Normal job `21316263` is pre-submitted with dependency `afterok:21315811`.
Once the target passes its own 225-column checks, this one-full-node job applies
the fixed 99% rule, initializes d from the atomic spectrum, and runs the real
joint optimization from source commit `d37e5570`. It stops before H2 so the
atomic gates can be reviewed without leaking the held-out result into basis
selection.

Target job `21315811` completed in `03:52:39`; the 9,064,244-byte matrix hash
is `866ac27a7b0456f332f40048e52e8370a27c85c12b566fdb8189757ae48c2c1b`.
The dependent preparation initially failed its `1e-8` magnetic-overlap gate.
Audit `21317474` localized the deviation to diagonal d blocks
(`3.6776163e-5`) with cross-mixing `6.15e-15`; s/p remain rotationally
consistent to numerical precision. The predeclared d counts are `2,2,3,5` at
`90%,95%,99%,99.9%`, so the production choice is three d shells.

Commit `829bac21` makes the measured finite-grid magnetic-channel tolerance
`1e-4` explicit in the spectrum JSON and keeps the material-anisotropy
rejection. Regression job `21317505` passed 112/112. Rerun `21317536` selected
`Nu.H=[4,3,3]` and reduced the ST loss to `0.02600043`, but its first two d
radial functions retained `20` and `10` significant nodes.

A relative overlap-rank audit showed that the original `1e-10` cutoff retained
21 modes and amplified near-null directions through `S^(-1/2)`. Cutoffs
`1e-8,1e-6,1e-4,1e-3` retained `20,19,18,17` modes; only `1e-4` and `1e-3`
gave the expected `0,1,2` d-node hierarchy. All choices still selected three d
shells and changed the leading response eigenvalues by only about `1e-5`
relative. This leaves the 99% physical selection unchanged.

Commit `b2a1685a` exposes and records the production rank cutoff `1e-4`. RED
job `21317792`, focused GREEN job `21317796`, and full regression job `21317811`
established the change; the latter passed 114/114. Formal job `21317844`
completed in `00:10:14`, kept `1s,2s,1p` bitwise fixed, and returned zero hinge
penalties. Its final ST, DFT-origin, DFT-dpsi, and weighted dpsi regularization
losses are `0.02638453`, `1.97545e-6`, `7.19839e-6`, and `0.01194913`.
The final d functions have `0,1,2` main nodes and the maximum ST condition is
`8.6065`, compared with `382.6418` in the pathological run. The atomic gate is
therefore closed with the smooth candidate only.

- [ ] **Step 5: Re-run the held-out gate once**

After the atomic ST and DFT/dpsi gates pass, run all H2/H bands with one common
fixed ABS cross test and the standard regenerated-ABS result. Report both so
the wavefunction-space improvement is separated from auxiliary-basis change.
