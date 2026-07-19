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

- [ ] **Step 6: Run the held-out H2/H SOS-RPA transfer test**

Replace only the H orbital in the validated 20-Angstrom H2/H SOS producer,
regenerate the ABFS and full-Coulomb data, and run LibRPA at 16 frequencies.
Recompute PBE, PBE-xc, EXX, and RPA correlation contributions for both H2 and
spin-polarized H; do not reuse the original TZDP non-correlation binding term.
