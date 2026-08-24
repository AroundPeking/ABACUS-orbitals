# Generic Two-Family Projected-Pi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the H/H2 name restriction from the two-family projected-Pi optimizer while preserving its validated numerical objective.

**Architecture:** Keep the existing exactly-two-family contract and use input declaration order as the canonical family order. Store the two names in the optimization adapter and replace every H/H2 lookup with ordered-family iteration; make the loader validate structure rather than literal names.

**Tech Stack:** Python 3.10, PyTorch 2.2, `unittest`, existing SIAB response-v1/source-v1 readers.

---

### Task 1: Generalize The Optimization Adapter

**Files:**
- Modify: `SIAB/tests/test_projected_pi_optimization.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/projected_pi_optimization.py`

- [ ] **Step 1: Write the failing generic-family test**

Add a test that constructs the existing two fixture pairs under new names:

```python
def test_accepts_any_two_unique_physical_family_names(self):
    result = self.adapter(
        ("C_atom", self.h), ("C2", self.h2)
    ).evaluate(coefficients(self.coefficient))
    self.assertEqual(tuple(result.family_results), ("C_atom", "C2"))
```

Also require one, three, and duplicate names to raise an `exactly two unique`
error, and require reversed input order to remain reversed in
`family_results`.

- [ ] **Step 2: Run the focused test and verify RED**

Run on the df SIAB virtual environment:

```bash
python -m unittest test_projected_pi_optimization.py
```

Expected: the new `C_atom/C2` test fails with the current H/H2-only error.

- [ ] **Step 3: Implement ordered generic names**

Replace `_PHYSICAL_FAMILIES` with an instance tuple derived from normalized
pairs:

```python
if len(items) != 2 or len(set(names)) != 2 or any(not name for name in names):
    raise ValueError("projected-Pi optimization requires exactly two unique family names")
self._family_names = names
ordered = items
```

Use `self._family_names` for the fourth-order norm. For ordinary mode, compare
the two named family frequency values and weights, and average their
per-frequency losses without referring to H or H2.

- [ ] **Step 4: Run the test and verify GREEN**

Run:

```bash
python -m unittest test_projected_pi_optimization.py
```

Expected: all adapter tests pass and the existing H/H2 values are unchanged.

- [ ] **Step 5: Commit the adapter change**

Commit only the adapter and its test with message:

```text
Generalize projected Pi optimization family names
```

### Task 2: Generalize The Sternheimer Input Loader

**Files:**
- Modify: `SIAB/tests/test_main_sternheimer.py`
- Modify: `SIAB/opt_orb_pytorch_dpsi/main.py`

- [ ] **Step 1: Write the failing C-family loader test**

Reuse the existing source/response/audit fixtures, but declare entries named
`C_atom` and `C2`. Assert:

```python
self.assertEqual(
    tuple(name for name, _ in loaded.projected_pi_pairs),
    ("C_atom", "C2"),
)
self.assertEqual(
    tuple(family.name for family in loaded.families),
    ("C_atom", "C2"),
)
```

Add a reversed-order assertion and retain failures for one, three, duplicate,
ghost, missing-source, and missing-audit inputs.

- [ ] **Step 2: Run the loader test and verify RED**

Run:

```bash
python -m unittest test_main_sternheimer.py
```

Expected: the C-family fixture fails with the H/H2-only validation message.

- [ ] **Step 3: Implement structural validation**

Replace the literal set comparison with:

```python
family_names = tuple(entry.family for entry in entries)
if len(entries) != 2 or len(set(family_names)) != 2 or any(
    not name for name in family_names
):
    raise ValueError(
        f"{projected_pi_label} requires exactly two unique physical families"
    )
```

Load response, source, audit, pairs, and `ResponseTargetFamily` values by
iterating over `entries` in declaration order. Do not sort or rename them.

- [ ] **Step 4: Run loader and adapter regressions**

Run:

```bash
python -m unittest \
  test_main_sternheimer.py \
  test_projected_pi_optimization.py \
  test_projected_pi.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the loader change**

Commit only the loader and its test with message:

```text
Accept generic projected Pi input families
```

### Task 3: Freeze The Interface Gate

**Files:**
- Modify: `SIAB/example_C_sternheimer/atomic_basis_optimization/README.md`
- Verify: `docs/superpowers/specs/2026-08-24-siab-generic-two-family-projected-pi-design.md`

- [ ] **Step 1: Run the focused remote suite**

Run the projected-Pi, loader, response-selection, and C atomic campaign tests
in the immutable df virtual environment. Require no failures or warnings from
the changed interface.

- [ ] **Step 2: Record the C campaign boundary**

Document that the software gate accepts names `C_atom/C2`, but no physical C
projected-Pi result exists until both strict response/source/audit pairs are
produced with the frozen 20 Angstrom, FD8, 16-frequency, full-Coulomb,
SG15/TZDP, PCA `1e-4` definition.

- [ ] **Step 3: Commit and push**

Commit with message:

```text
Document C projected Pi production gate
```

Verify Codex author, AroundPeking committer, clean worktree, and remote branch
identity before beginning ABACUS source-v1 production.
